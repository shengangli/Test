# pylint:skip-file
"""
customentrypoint.py

PMR custom entrypoint (structuredquery / streaming) doing document RAG against a
Bedrock Knowledge Base via RetrieveAndGenerateStream.

Built to work without every framework detail confirmed up front:
  * question key      -> probes likely keys, then deep-searches the payload
  * config key names  -> accepts several spellings, plus env fallback
  * logger type       -> works with structlog or a plain logger
  * event shape       -> defensively pulls text from any known field path
  * aioboto3/boto3    -> uses aioboto3 if installed, else boto3 on a thread

Contract:
  * yield RAW text only (no json.dumps, no newline)
  * stream MUST end with __STOP_CODON__
"""
from __future__ import annotations

import asyncio
import os
import queue
import threading
import uuid
from typing import Any, AsyncGenerator, Optional

from customentrypointbase import CustomEntrypointBase
from fastapi import Request

STOP_CODON = "__STOP_CODON__"

# Likely keys holding the user's question. Probed in order.
QUESTION_KEYS = (
    "prompt", "question", "text", "query", "user_question",
    "user_input", "message", "content", "input",
)

# Accepted spellings for each config value.
CFG_ALIASES = {
    "region": ("AWSRegion", "AWS_REGION", "aws_region", "region", "BEDROCK_REGION"),
    "kb_id": ("KNOWLEDGE_BASE_ID", "KnowledgeBaseId", "knowledge_base_id",
              "BEDROCK_KNOWLEDGE_BASE_ID", "kb_id"),
    "model_arn": ("GENERATION_MODEL_ARN", "MODEL_ARN", "ModelArn",
                  "model_arn", "BEDROCK_MODEL_ARN"),
    "model_id": ("GENERATION_MODEL_ID", "MODEL_ID", "ModelId", "model_id",
                 "modelId", "BEDROCK_MODEL_ID"),
    "profile": ("AWS_PROFILE", "aws_profile", "AWSProfile"),
    "data_source_id": ("DATA_SOURCE_ID", "DataSourceId", "data_source_id",
                       "BEDROCK_DATA_SOURCE_ID"),
    "timeout": ("BEDROCK_TIMEOUT_SECONDS", "bedrock_timeout"),
}

DEFAULT_REGION = "us-east-1"
DEFAULT_TIMEOUT = 60
ERROR_MESSAGE = "申し訳ありません。回答の生成中にエラーが発生しました。"

_CLIENT_LOCK = threading.Lock()
_CLIENTS: dict = {}


class CustomEntryPoint(CustomEntrypointBase):
    domain_dict = None
    logger_obj = None

    def __init__(self, domain_dict, logger_obj):
        self.domain_dict = domain_dict
        self.logger_obj = logger_obj

    # ------------------------------------------------------------------ #
    async def execute_custom_entrypoint(
        self,
        app_name: str,
        request: Request,
        request_payload: dict,
        app_config_dict: dict,
    ) -> AsyncGenerator[str, None]:
        req_id = (request_payload or {}).get("client_request_id") or uuid.uuid4().hex

        try:
            question = self._extract_question(request_payload)
            if not question:
                self._log("error",
                          f"[{app_name}][{req_id}] no question in payload; "
                          f"keys={list((request_payload or {}).keys())}")
                yield "質問が見つかりませんでした。"
                return

            self._log("info", f"[{app_name}][{req_id}] Q({len(question)} chars)")

            got_output = False
            async for chunk in self._stream(question, app_config_dict, req_id):
                if chunk:
                    got_output = True
                    yield chunk

            if not got_output:
                self._log("error", f"[{app_name}][{req_id}] empty response")
                yield "該当する情報が見つかりませんでした。"

        except asyncio.CancelledError:
            self._log("info", f"[{app_name}][{req_id}] cancelled by client")
            raise

        except Exception as exc:  # noqa: BLE001
            self._log("error", f"[{app_name}][{req_id}] failed: {exc!r}")
            yield ERROR_MESSAGE

        finally:
            # Always terminate - clients block forever without this.
            yield STOP_CODON

    # ------------------------------------------------------------------ #
    # Question extraction
    # ------------------------------------------------------------------ #
    @classmethod
    def _extract_question(cls, payload: Optional[dict]) -> Optional[str]:
        if not isinstance(payload, dict):
            return None

        # 1) direct hits at top level
        for key in QUESTION_KEYS:
            val = payload.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()

        # 2) one level down (e.g. chat.prompt, request.text)
        for container in payload.values():
            if isinstance(container, dict):
                for key in QUESTION_KEYS:
                    val = container.get(key)
                    if isinstance(val, str) and val.strip():
                        return val.strip()

        # 3) last resort: longest natural-language-looking string that is not
        #    an id / timestamp / flag field.
        skip = ("id", "datetime", "flag", "name", "window", "transaction")
        best = None
        for key, val in payload.items():
            if any(s in key.lower() for s in skip):
                continue
            if isinstance(val, str) and len(val.strip()) > 5:
                if best is None or len(val) > len(best):
                    best = val.strip()
        return best

    # ------------------------------------------------------------------ #
    # Bedrock
    # ------------------------------------------------------------------ #
    async def _stream(
        self, question: str, cfg: dict, req_id: str
    ) -> AsyncGenerator[str, None]:

        if os.getenv("USE_MOCK_BEDROCK", "false").lower() == "true":
            for ch in "これはモック回答です。ストリーミング経路の確認用です。":
                await asyncio.sleep(0.01)
                yield ch
            return

        region = self._cfg(cfg, "region") or DEFAULT_REGION
        kb_id = self._cfg(cfg, "kb_id")
        model_arn = self._resolve_model_arn(cfg, region)
        ds_id = self._cfg(cfg, "data_source_id")
        timeout = int(self._cfg(cfg, "timeout") or DEFAULT_TIMEOUT)

        missing = [n for n, v in (("kb_id", kb_id), ("model_arn", model_arn)) if not v]
        if missing:
            raise RuntimeError(
                f"missing config: {', '.join(missing)} "
                "(expected from app_config_dict / conf/domain.conf)"
            )

        kb_conf: dict[str, Any] = {"knowledgeBaseId": kb_id, "modelArn": model_arn}
        if ds_id:
            kb_conf["retrievalConfiguration"] = {
                "vectorSearchConfiguration": {
                    "filter": {"equals": {
                        "key": "x-amz-bedrock-kb-data-source-id",
                        "value": ds_id,
                    }}
                }
            }

        params = {
            "input": {"text": question},
            "retrieveAndGenerateConfiguration": {
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": kb_conf,
            },
        }

        profile = self._cfg(cfg, "profile")
        async for chunk in self._invoke(params, region, timeout, req_id, profile):
            yield chunk

    async def _invoke(self, params, region, timeout, req_id, profile=None):
        """Prefer aioboto3 (native async); fall back to boto3 on a thread."""
        try:
            import aioboto3  # noqa: F401
            has_aio = True
        except ImportError:
            has_aio = False

        if has_aio:
            import aioboto3
            session = (aioboto3.Session(profile_name=profile)
                       if profile else aioboto3.Session())
            async with session.client(
                "bedrock-agent-runtime", region_name=region
            ) as client:
                resp = await client.retrieve_and_generate_stream(**params)
                first = True
                async for event in resp["stream"]:
                    if first:
                        self._log("info", f"[{req_id}] first event: {event}")
                        first = False
                    text = self._text_from(event)
                    if text:
                        yield text
            return

        # --- boto3 fallback: iterate the blocking stream off the event loop --
        import boto3
        from botocore.config import Config as BotoConfig

        cache_key = f"{region}|{profile or ''}"
        with _CLIENT_LOCK:
            client = _CLIENTS.get(cache_key)
            if client is None:
                sess = (boto3.Session(profile_name=profile)
                        if profile else boto3.Session())
                client = sess.client(
                    "bedrock-agent-runtime",
                    region_name=region,
                    config=BotoConfig(read_timeout=timeout, connect_timeout=10,
                                      retries={"max_attempts": 2,
                                               "mode": "standard"}),
                )
                _CLIENTS[cache_key] = client

        loop = asyncio.get_running_loop()
        q: queue.Queue = queue.Queue(maxsize=256)
        done = object()

        def pump():
            try:
                resp = client.retrieve_and_generate_stream(**params)
                first = True
                for event in resp["stream"]:
                    if first:
                        self._log("info", f"[{req_id}] first event: {event}")
                        first = False
                    text = self._text_from(event)
                    if text:
                        q.put(text)
            except BaseException as exc:  # noqa: BLE001
                q.put(exc)
            finally:
                q.put(done)

        loop.run_in_executor(None, pump)
        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is done:
                break
            if isinstance(item, BaseException):
                raise item
            yield item

    def _resolve_model_arn(self, cfg: Optional[dict], region: str):
        """
        The generation model comes from the APPLICATION config, not from the
        developer. Accept either a full ARN or a bare model id, and build the
        ARN from the id when only that is supplied.
        """
        arn = self._cfg(cfg, "model_arn")
        if arn:
            return arn
        model_id = self._cfg(cfg, "model_id")
        if model_id:
            if str(model_id).startswith("arn:"):
                return model_id
            return f"arn:aws:bedrock:{region}::foundation-model/{model_id}"
        return None

    @staticmethod
    def _text_from(event: Any) -> str:
        """Pull text from an event, tolerating several possible shapes."""
        if isinstance(event, str):
            return event
        if not isinstance(event, dict):
            return ""

        # {"output": {"text": "..."}}
        out = event.get("output")
        if isinstance(out, dict) and isinstance(out.get("text"), str):
            return out["text"]
        if isinstance(out, str):
            return out

        # {"chunk": {"text"/"bytes": ...}}
        chunk = event.get("chunk")
        if isinstance(chunk, dict):
            if isinstance(chunk.get("text"), str):
                return chunk["text"]
            raw = chunk.get("bytes")
            if isinstance(raw, (bytes, bytearray)):
                try:
                    return raw.decode("utf-8")
                except Exception:  # noqa: BLE001
                    return ""

        # {"text": "..."} / {"delta": {"text": "..."}}
        if isinstance(event.get("text"), str):
            return event["text"]
        delta = event.get("delta")
        if isinstance(delta, dict) and isinstance(delta.get("text"), str):
            return delta["text"]

        return ""

    # ------------------------------------------------------------------ #
    # Config + logging
    # ------------------------------------------------------------------ #
    def _cfg(self, app_config_dict: Optional[dict], name: str):
        """Look up a value by any of its accepted key spellings."""
        for key in CFG_ALIASES.get(name, (name,)):
            for source in (app_config_dict, self.domain_dict):
                if isinstance(source, dict) and source.get(key) not in (None, ""):
                    return source[key]
            env = os.getenv(key)
            if env not in (None, ""):
                return env
        return None

    def _log(self, level: str, message: str) -> None:
        if not self.logger_obj:
            return
        fn = getattr(self.logger_obj, level, None) or getattr(
            self.logger_obj, "info", None
        )
        if fn:
            try:
                fn(message)
            except Exception:  # noqa: BLE001
                pass
