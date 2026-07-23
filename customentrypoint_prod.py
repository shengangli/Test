# pylint:skip-file
"""
customentrypoint.py

PMR custom entrypoint (structuredquery / streaming route) performing document RAG
against an Amazon Bedrock Knowledge Base via RetrieveAndGenerateStream.

Flow:
    1UI --(WebSocket)--> PMR --> execute_custom_entrypoint()
        --> bedrock-agent-runtime.retrieve_and_generate_stream()
        --> yield text chunks --> PMR --> 1UI

Contract notes (verified against the framework):
  * yield RAW text only - no json.dumps(), no trailing newline.
  * the stream MUST terminate with __STOP_CODON__; clients block until they see it.
  * the framework wraps yielded text into `custom_entrypoint_response`.
"""
from __future__ import annotations

import asyncio
import functools
import os
import queue
import threading
import uuid
from typing import Any, AsyncGenerator, Optional

import boto3
from botocore.config import Config as BotoConfig
from customentrypointbase import CustomEntrypointBase
from fastapi import Request

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

STOP_CODON = "__STOP_CODON__"

# Ordered candidates for locating the user's question in request_payload.
# TODO: once confirmed against a test-case JSON, reduce this to the single key.
QUESTION_KEYS = ("prompt", "question", "text", "query", "user_question")

# Config keys (resolved from app_config_dict, then domain_dict, then env).
CFG_REGION = "AWSRegion"
CFG_KB_ID = "KNOWLEDGE_BASE_ID"
CFG_MODEL_ARN = "GENERATION_MODEL_ARN"
CFG_DATA_SOURCE_ID = "DATA_SOURCE_ID"   # optional - filters to one data source
CFG_TIMEOUT = "BEDROCK_TIMEOUT_SECONDS"

DEFAULT_REGION = "us-east-1"
DEFAULT_TIMEOUT = 60

USER_FACING_ERROR = (
    "申し訳ありませんが、回答の生成中にエラーが発生しました。"
    "しばらくしてからもう一度お試しください。"
)

# Module-level client cache. Creating a boto3 client is expensive (~100ms+) and
# clients are thread-safe, so we build one per region and reuse it.
_CLIENT_LOCK = threading.Lock()
_CLIENTS: dict[str, Any] = {}


def _get_client(region: str, timeout: int):
    """Return a cached bedrock-agent-runtime client for the given region."""
    with _CLIENT_LOCK:
        client = _CLIENTS.get(region)
        if client is None:
            client = boto3.client(
                "bedrock-agent-runtime",
                region_name=region,
                config=BotoConfig(
                    read_timeout=timeout,
                    connect_timeout=10,
                    retries={"max_attempts": 2, "mode": "standard"},
                ),
            )
            _CLIENTS[region] = client
        return client


class ConfigError(RuntimeError):
    """Raised when a required configuration value is missing."""


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #


class CustomEntryPoint(CustomEntrypointBase):
    domain_dict = None
    logger_obj = None

    def __init__(self, domain_dict, logger_obj):
        self.domain_dict = domain_dict
        self.logger_obj = logger_obj

    # ------------------------------------------------------------------ #
    # Main entrypoint - PMR calls this once per request.
    # ------------------------------------------------------------------ #
    async def execute_custom_entrypoint(
        self,
        app_name: str,
        request: Request,
        request_payload: dict,
        app_config_dict: dict,
    ) -> AsyncGenerator[str, None]:
        # Correlate logs to the caller's request where possible.
        request_id = (
            (request_payload or {}).get("client_request_id")
            or uuid.uuid4().hex
        )
        log_ctx = {"app_name": app_name, "request_id": request_id}

        try:
            question = self._extract_question(request_payload)
            if not question:
                self._log("error", "no question found in request_payload", **log_ctx)
                yield "質問が見つかりませんでした。"
                return

            self._log(
                "info",
                f"question received (len={len(question)})",
                **log_ctx,
            )

            emitted = False
            async for chunk in self._stream_answer(
                question, app_config_dict, log_ctx
            ):
                if chunk:
                    emitted = True
                    yield chunk          # RAW text - no json.dumps, no newline

            if not emitted:
                self._log("error", "bedrock returned no content", **log_ctx)
                yield "該当する情報が見つかりませんでした。"

        except ConfigError as exc:
            # Misconfiguration: log loudly, surface a generic message.
            self._log("error", f"configuration error: {exc}", **log_ctx)
            yield USER_FACING_ERROR

        except asyncio.CancelledError:
            # Client disconnected mid-stream. Do not emit; let it propagate.
            self._log("info", "stream cancelled by client", **log_ctx)
            raise

        except Exception as exc:  # noqa: BLE001 - must never break the stream
            self._log("error", f"unhandled error: {exc!r}", **log_ctx)
            yield USER_FACING_ERROR

        finally:
            # The terminator MUST always be sent, on every path except
            # cancellation, or the client will wait forever.
            yield STOP_CODON

    # ------------------------------------------------------------------ #
    # Question extraction
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_question(payload: Optional[dict]) -> Optional[str]:
        if not isinstance(payload, dict):
            return None

        for key in QUESTION_KEYS:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        chat = payload.get("chat")
        if isinstance(chat, dict):
            for key in ("prompt", "text", "question"):
                value = chat.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

        return None

    # ------------------------------------------------------------------ #
    # Bedrock streaming
    # ------------------------------------------------------------------ #
    async def _stream_answer(
        self,
        question: str,
        app_config_dict: dict,
        log_ctx: dict,
    ) -> AsyncGenerator[str, None]:
        if os.getenv("USE_MOCK_BEDROCK", "false").lower() == "true":
            async for chunk in self._mock_stream():
                yield chunk
            return

        region = self._cfg(app_config_dict, CFG_REGION, DEFAULT_REGION)
        kb_id = self._require(app_config_dict, CFG_KB_ID)
        model_arn = self._require(app_config_dict, CFG_MODEL_ARN)
        data_source_id = self._cfg(app_config_dict, CFG_DATA_SOURCE_ID)
        timeout = int(self._cfg(app_config_dict, CFG_TIMEOUT, DEFAULT_TIMEOUT))

        kb_config: dict[str, Any] = {
            "knowledgeBaseId": kb_id,
            "modelArn": model_arn,
        }

        # Optional: restrict retrieval to a single data source within the KB.
        if data_source_id:
            kb_config["retrievalConfiguration"] = {
                "vectorSearchConfiguration": {
                    "filter": {
                        "equals": {
                            "key": "x-amz-bedrock-kb-data-source-id",
                            "value": data_source_id,
                        }
                    }
                }
            }

        params = {
            "input": {"text": question},
            "retrieveAndGenerateConfiguration": {
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": kb_config,
            },
        }

        client = _get_client(region, timeout)

        # boto3 is synchronous and its EventStream blocks on iteration. Running
        # it inline would stall the PMR event loop for every other connection,
        # so we consume it on a worker thread and hand chunks back via a queue.
        async for chunk in self._iter_blocking_stream(client, params, log_ctx):
            yield chunk

    async def _iter_blocking_stream(
        self,
        client,
        params: dict,
        log_ctx: dict,
    ) -> AsyncGenerator[str, None]:
        loop = asyncio.get_running_loop()
        chunks: queue.Queue = queue.Queue(maxsize=256)
        sentinel = object()

        def _pump() -> None:
            """Runs on a worker thread; pushes text chunks onto the queue."""
            try:
                response = client.retrieve_and_generate_stream(**params)
                for event in response["stream"]:
                    text = self._text_from_event(event)
                    if text:
                        chunks.put(text)
            except BaseException as exc:  # noqa: BLE001 - forwarded to caller
                chunks.put(exc)
            finally:
                chunks.put(sentinel)

        worker = loop.run_in_executor(None, _pump)

        try:
            while True:
                item = await loop.run_in_executor(None, chunks.get)
                if item is sentinel:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            # Ensure the worker is drained/finished before we return.
            with contextlib_suppress():
                await worker

    @staticmethod
    def _text_from_event(event: dict) -> str:
        """
        Pull the text delta out of one stream event.

        RetrieveAndGenerateStream emits several event types (output, citation,
        guardrail, metadata). We only forward generated text.

        TODO: confirm against a real response - print one raw event and verify
        the field path before shipping to DEV.
        """
        if not isinstance(event, dict):
            return ""

        output = event.get("output")
        if isinstance(output, dict):
            text = output.get("text")
            if isinstance(text, str):
                return text

        return ""

    async def _mock_stream(self) -> AsyncGenerator[str, None]:
        text = (
            "これはモックの回答です。ナレッジベースに接続せずに"
            "ストリーミング経路を検証しています。"
        )
        for token in text:
            await asyncio.sleep(0.01)
            yield token

    # ------------------------------------------------------------------ #
    # Config helpers
    # ------------------------------------------------------------------ #
    def _cfg(self, app_config_dict: Optional[dict], key: str, default=None):
        """Resolve config: app_config_dict -> domain_dict -> environment."""
        for source in (app_config_dict, self.domain_dict):
            if isinstance(source, dict) and source.get(key) not in (None, ""):
                return source[key]
        env_value = os.getenv(key)
        return env_value if env_value not in (None, "") else default

    def _require(self, app_config_dict: Optional[dict], key: str):
        value = self._cfg(app_config_dict, key)
        if not value:
            raise ConfigError(f"{key} is not configured")
        return value

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    def _log(self, level: str, message: str, **context) -> None:
        if not self.logger_obj:
            return
        emit = getattr(self.logger_obj, level, None)
        if emit is None:
            return
        try:
            emit(message, **context)
        except TypeError:
            # Logger does not accept structured kwargs - fall back to a string.
            suffix = " ".join(f"{k}={v}" for k, v in context.items())
            emit(f"{message} {suffix}".strip())


# Small local helper so we don't add an import just for one suppress().
class contextlib_suppress:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return True
