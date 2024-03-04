"""Chatbot callbacks."""

import sys
from collections.abc import Sequence
from typing import Any
from uuid import UUID

import tiktoken
from langchain.adapters.openai import convert_message_to_dict
from langchain_community.callbacks import OpenAICallbackHandler
from langchain_community.callbacks.openai_info import (
    get_openai_token_cost_for_model,
)
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import (
    ChatGenerationChunk,
    GenerationChunk,
    LLMResult,
)

encoding = tiktoken.get_encoding("cl100k_base")

token_counts: dict[UUID, int] = {}


def num_tokens_from_messages(
    messages: Sequence[dict], model="gpt-3.5-turbo-0613"
):
    """Returns the number of tokens used by a list of messages."""
    num_tokens: int = 0
    for message in messages:
        cost: int = 4
        for key, value in message.items():
            cost += len(encoding.encode(value))
            if key == "name":  # pragma: no cover
                cost += -1  # role is always required and always 1 token
        num_tokens += cost
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens


class StreamCallbackHandler(OpenAICallbackHandler):
    """Chatbot callback handler."""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Collect token usage."""
        run_id: UUID = kwargs["run_id"]
        with self._lock:
            token_counts[run_id] = token_counts.get(run_id, 0) + 1
        return super().on_llm_new_token(token, **kwargs)

    def on_chat_model_start(self, *args: Any, **kwargs: Any) -> Any:
        """Calculate the cost of the prompt."""
        messages: list[list[BaseMessage]] = kwargs.get("messages", args[1])
        prompt_tokens = num_tokens_from_messages(
            [convert_message_to_dict(message) for message in messages[0]]
        )
        prompt_cost = get_openai_token_cost_for_model(
            "gpt-3.5-turbo-0613", prompt_tokens
        )
        with self._lock:
            self.total_cost += prompt_cost
            self.total_tokens += prompt_tokens
            self.prompt_tokens += prompt_tokens
        return super().on_chat_model_start(*args, **kwargs)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Calculate generation cost."""
        run_id: UUID = kwargs["run_id"]
        with self._lock:
            self.total_tokens += token_counts.get(run_id, 0)
            self.completion_tokens += token_counts.get(run_id, 0)
            self.total_cost += get_openai_token_cost_for_model(
                "gpt-3.5-turbo-0613", token_counts.get(run_id, 0)
            )
            self.successful_requests += 1
        return super().on_llm_end(response, **kwargs)


class AlexisCallback(BaseCallbackHandler):
    """Alexis agent callback handler."""

    def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: GenerationChunk | ChatGenerationChunk | None = None,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Collect token usage."""
        sys.stdout.write(token)
        sys.stdout.flush()
        return super().on_llm_new_token(
            token,
            chunk=chunk,
            run_id=run_id,
            parent_run_id=parent_run_id,
            **kwargs,
        )
