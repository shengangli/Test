# pylint:skip-file
"""
customentrypoint.py

PMR custom entrypoint (structuredquery / streaming) doing document RAG against a
Bedrock Knowledge Base via RetrieveAndGenerateStream.

Verified against live AWS (2026-07):
  * region: ap-northeast-1 (Tokyo) - KB 4SFRKUP5WR lives here
  * model:  jp. inference-profile ARN (account-scoped), supplied via config
  * the equivalent non-streaming call was proven working via CLI

Contract (verified against framework + test clients):
  * yield RAW text only - no json.dumps(), no newline
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

# Question keys, ordered by likelihood. chat.prompt / question confirmed from
# the retrieval-lambda's extract_question (boss's handler.py).
QUESTION_KEYS = ("prompt", "question", "text", "query", "user_question")

# Accepted spellings for each config value (app_config_dict -> domain_dict -> env).
CFG_ALIASES = {
    "region": ("aws_region", "AWSRegion", "AWS_REGION", "region", "BEDROCK_REGION"),
    "kb_id": ("knowledge_base_id", "KNOWLEDGE_BASE_ID", "KnowledgeBaseId",
              "BEDROCK_KNOWLEDGE_BASE_ID", "kb_id"),
    "model_arn": ("generation_model_arn", "GENERATION_MODEL_ARN", "MODEL_ARN",
                  "ModelArn", "model_arn", "BEDROCK_MODEL_ARN"),
    "data_source_id": ("data_source_id", "DATA_SOURCE_ID", "DataSourceId",
                       "BEDROCK_DATA_SOURCE_ID"),
    "timeout": ("BEDROCK_TIMEOUT_SECONDS", "bedrock_timeout"),
}

DEFAULT_REGION = "ap-northeast-1"   # Tokyo - confirmed home of the KB
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
    # Question extraction (keys confirmed against retrieval-lambda's logic)
    # ------------------------------------------------------------------ #
    @classmethod
    def _extract_question(cls, payload: Optional[dict]) -> Optional[str]:
        if not isinstance(payload, dict):
            return None

        for key in QUESTION_KEYS:
            val = payload.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()

        # nested, e.g. chat.prompt (the 1UI shape the Lambda also handles)
        for container in payload.values():
            if isinstance(container, dict):
                for key in QUESTION_KEYS:
                    val = container.get(key)
                    if isinstance(val, str) and val.strip():
                        return val.strip()

        # chat-style messages list: last user message wins
        messages = payload.get("messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                if (isinstance(message, dict)
                        and message.get("role") == "user"
                        and isinstance(message.get("content"), str)
                        and message["content"].strip()):
                    return message["content"].strip()

        return None

    # ------------------------------------------------------------------ #
    # Bedrock RetrieveAndGenerateStream
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
        model_arn = self._cfg(cfg, "model_arn")   # full ARN incl. jp. inference profile
        ds_id = self._cfg(cfg, "data_source_id")
        timeout = int(self._cfg(cfg, "timeout") or DEFAULT_TIMEOUT)

        missing = [n for n, v in (("knowledge_base_id", kb_id),
                                  ("generation_model_arn", model_arn)) if not v]
        if missing:
            raise RuntimeError(
                f"missing config: {', '.join(missing)} "
                "(expected from domain.conf custom_entrypoint_config)"
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

        async for chunk in self._invoke(params, region, timeout, req_id):
            yield chunk

    async def _invoke(self, params, region, timeout, req_id):
        """Prefer aioboto3 (native async, already in requirements);
        fall back to boto3 on a worker thread."""
        try:
            import aioboto3
            has_aio = True
        except ImportError:
            has_aio = False

        if has_aio:
            import aioboto3
            session = aioboto3.Session()
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

        # --- boto3 fallback: blocking stream consumed off the event loop ---
        import boto3
        from botocore.config import Config as BotoConfig

        with _CLIENT_LOCK:
            client = _CLIENTS.get(region)
            if client is None:
                client = boto3.client(
                    "bedrock-agent-runtime",
                    region_name=region,
                    config=BotoConfig(read_timeout=timeout, connect_timeout=10,
                                      retries={"max_attempts": 2,
                                               "mode": "standard"}),
                )
                _CLIENTS[region] = client

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

    @staticmethod
    def _text_from(event: Any) -> str:
        """Pull text from a stream event, tolerating the known shapes.
        (Non-streaming response nests text at output.text - streaming events
        are expected to carry an output/text delta; confirm with first-event log.)"""
        if isinstance(event, str):
            return event
        if not isinstance(event, dict):
            return ""

        out = event.get("output")
        if isinstance(out, dict) and isinstance(out.get("text"), str):
            return out["text"]
        if isinstance(out, str):
            return out

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
        """Look up a value by any accepted spelling: app_config_dict ->
        domain_dict -> environment."""
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
