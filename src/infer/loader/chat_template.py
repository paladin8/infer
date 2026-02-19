"""Jinja2-based chat template renderer.

Formats a list of chat messages into a prompt string using per-model
Jinja2 templates.  The returned string includes special token text
(e.g. ``<|begin_of_text|>``) and should be tokenized with
``tokenizer.encode(prompt, add_special_tokens=False)``.

These are simplified templates for the educational runtime.  They cover
the standard chat roles (system, user, assistant) but intentionally omit
model-specific features like Llama's date/knowledge-cutoff injection
and tool-call formatting.
"""

from __future__ import annotations

import jinja2

# ---------------------------------------------------------------------------
# Template strings
#
# Templates use string concatenation for precise whitespace control.
# All tag boundaries use {%- -%} / {{- -}} to strip surrounding whitespace.
# Newlines within the output come from \\n in the Python source, which
# becomes \n in the Jinja2 string literal and is interpreted as a newline.
# ---------------------------------------------------------------------------

# Llama 3 Instruct format:
#   <|begin_of_text|><|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>...
_LLAMA_TEMPLATE = (
    "{{- '<|begin_of_text|>' -}}"
    "{%- for message in messages -%}"
    "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'"
    " + message['content'] + '<|eot_id|>' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}"
    "{%- endif -%}"
)

# Qwen 3 ChatML format:
#   <|im_start|>{role}\n{content}<|im_end|>\n...
_QWEN3_TEMPLATE = (
    "{%- for message in messages -%}"
    "{{- '<|im_start|>' + message['role'] + '\\n'"
    " + message['content'] + '<|im_end|>\\n' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{{- '<|im_start|>assistant\\n' -}}"
    "{%- endif -%}"
)

# Gemma 3 format:
#   <bos><start_of_turn>{role}\n{content}<end_of_turn>\n...
# System messages are folded into the first user turn (no <start_of_turn>system).
_GEMMA3_TEMPLATE = (
    "{{- '<bos>' -}}"
    "{%- if messages[0]['role'] == 'system' -%}"
    "{%- set first_user_prefix = messages[0]['content'] + '\\n\\n' -%}"
    "{%- set loop_messages = messages[1:] -%}"
    "{%- else -%}"
    "{%- set first_user_prefix = '' -%}"
    "{%- set loop_messages = messages -%}"
    "{%- endif -%}"
    "{%- for message in loop_messages -%}"
    "{%- if message['role'] == 'assistant' -%}"
    "{{- '<start_of_turn>model\\n' + message['content'] + '<end_of_turn>\\n' -}}"
    "{%- else -%}"
    "{{- '<start_of_turn>' + message['role'] + '\\n'"
    " + (first_user_prefix if loop.first else '')"
    " + message['content'] + '<end_of_turn>\\n' -}}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{{- '<start_of_turn>model\\n' -}}"
    "{%- endif -%}"
)

# ---------------------------------------------------------------------------
# Compiled templates
# ---------------------------------------------------------------------------

_ENV = jinja2.Environment(undefined=jinja2.StrictUndefined)

_TEMPLATES: dict[str, jinja2.Template] = {
    "llama": _ENV.from_string(_LLAMA_TEMPLATE),
    "qwen3": _ENV.from_string(_QWEN3_TEMPLATE),
    "gemma3_text": _ENV.from_string(_GEMMA3_TEMPLATE),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_chat_template(
    messages: list[dict[str, str]],
    model_type: str,
    *,
    add_generation_prompt: bool = True,
) -> str:
    """Render chat messages into a formatted prompt string.

    Args:
        messages: List of message dicts, each with ``"role"`` and
            ``"content"`` keys.  Roles: ``"system"``, ``"user"``,
            ``"assistant"``.
        model_type: One of ``"llama"``, ``"qwen3"``, ``"gemma3_text"``.
        add_generation_prompt: Whether to append the assistant turn
            header at the end (for prompting the model to generate).

    Returns:
        The formatted prompt string, including special token text.

    Raises:
        ValueError: If *model_type* is not supported.
    """
    template = _TEMPLATES.get(model_type)
    if template is None:
        raise ValueError(
            f"No chat template for model_type: {model_type!r}. "
            f"Supported types: {sorted(_TEMPLATES.keys())}"
        )
    return template.render(messages=messages, add_generation_prompt=add_generation_prompt)
