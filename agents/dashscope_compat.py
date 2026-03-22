"""DashScope-compatible client that mimics Anthropic's messages API shape.

This adapter keeps the tutorial code focused on harness patterns by preserving
`response.content` blocks and `response.stop_reason == "tool_use"` behavior.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from openai import OpenAI


@dataclass
class TextBlock:
    type: str
    text: str


@dataclass
class ToolUseBlock:
    type: str
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class AnthropicLikeResponse:
    content: list[Any]
    stop_reason: str


class _MessagesAdapter:
    def __init__(self, openai_client: OpenAI):
        self._client = openai_client

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AnthropicLikeResponse:
        oai_messages = _to_openai_messages(messages, system)
        oai_tools = _to_openai_tools(tools or [])

        req: dict[str, Any] = {
            "model": model,
            "messages": oai_messages,
        }
        if oai_tools:
            req["tools"] = oai_tools
        if max_tokens is not None:
            req["max_tokens"] = max_tokens
        if temperature is not None:
            req["temperature"] = temperature
        req.update(kwargs)

        completion = self._client.chat.completions.create(**req)
        msg = completion.choices[0].message

        blocks: list[Any] = []
        if msg.content:
            blocks.append(TextBlock(type="text", text=msg.content))

        tool_calls = msg.tool_calls or []
        for tc in tool_calls:
            call_id = tc.id or f"call_{uuid.uuid4().hex[:12]}"
            name = tc.function.name
            args = tc.function.arguments or "{}"
            try:
                parsed = json.loads(args)
                if not isinstance(parsed, dict):
                    parsed = {"_value": parsed}
            except json.JSONDecodeError:
                parsed = {"_raw": args}

            blocks.append(
                ToolUseBlock(
                    type="tool_use",
                    id=call_id,
                    name=name,
                    input=parsed,
                )
            )

        stop_reason = "tool_use" if tool_calls else "end_turn"
        return AnthropicLikeResponse(content=blocks, stop_reason=stop_reason)


class AnthropicCompat:
    """Compatibility facade for tutorial scripts originally using Anthropic SDK."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        resolved_api_key = (
            api_key
            or os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        resolved_base_url = (
            base_url
            or os.getenv("DASHSCOPE_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        self._openai = OpenAI(api_key=resolved_api_key, base_url=resolved_base_url)
        self.messages = _MessagesAdapter(self._openai)


def _to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for t in tools:
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
        )
    return out


def _to_openai_messages(
    messages: list[dict[str, Any]], system: str | None
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        if isinstance(content, list):
            if role == "assistant":
                assistant = _assistant_blocks_to_openai(content)
                out.append(assistant)
            elif role == "user":
                out.extend(_user_parts_to_openai(content))
            else:
                out.append({"role": role, "content": json.dumps(content, ensure_ascii=False)})
            continue

        out.append({"role": role, "content": str(content)})

    return out


def _assistant_blocks_to_openai(blocks: list[Any]) -> dict[str, Any]:
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for block in blocks:
        btype = _block_get(block, "type")
        if btype == "text":
            txt = _block_get(block, "text")
            if txt:
                text_parts.append(str(txt))
        elif btype == "tool_use":
            call_id = _block_get(block, "id") or f"call_{uuid.uuid4().hex[:12]}"
            name = _block_get(block, "name") or "unknown_tool"
            args = _block_get(block, "input") or {}
            tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": str(name),
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                }
            )

    assistant_msg: dict[str, Any] = {
        "role": "assistant",
        "content": "\n".join(text_parts) if text_parts else None,
    }
    if tool_calls:
        assistant_msg["tool_calls"] = tool_calls
    return assistant_msg


def _user_parts_to_openai(parts: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    pending_text: list[str] = []

    def flush_pending_text() -> None:
        if pending_text:
            out.append({"role": "user", "content": "\n".join(pending_text)})
            pending_text.clear()

    for part in parts:
        ptype = _block_get(part, "type")
        if ptype == "tool_result":
            flush_pending_text()
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": str(_block_get(part, "tool_use_id") or ""),
                    "content": str(_block_get(part, "content") or ""),
                }
            )
        elif ptype == "text":
            pending_text.append(str(_block_get(part, "text") or ""))
        else:
            pending_text.append(str(part))

    flush_pending_text()
    return out


def _block_get(block: Any, key: str) -> Any:
    if isinstance(block, dict):
        return block.get(key)
    return getattr(block, key, None)
