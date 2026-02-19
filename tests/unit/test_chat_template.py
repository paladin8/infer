"""Tests for the chat template renderer."""

from __future__ import annotations

import pytest

from infer.loader.chat_template import render_chat_template

# ---------------------------------------------------------------------------
# Llama 3
# ---------------------------------------------------------------------------


class TestLlamaTemplate:
    def test_single_user_message(self) -> None:
        messages = [{"role": "user", "content": "Hello"}]
        result = render_chat_template(messages, "llama")
        expected = (
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "Hello<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        assert result == expected

    def test_system_and_user(self) -> None:
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        result = render_chat_template(messages, "llama")
        assert result.startswith("<|begin_of_text|><|start_header_id|>system<|end_header_id|>")
        assert "You are helpful.<|eot_id|>" in result
        assert "Hi<|eot_id|>" in result
        assert result.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")

    def test_multi_turn(self) -> None:
        messages = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]
        result = render_chat_template(messages, "llama")
        # Verify all roles present.
        assert "<|start_header_id|>system<|end_header_id|>" in result
        assert "<|start_header_id|>user<|end_header_id|>" in result
        assert "<|start_header_id|>assistant<|end_header_id|>" in result
        # Verify generation prompt at end.
        assert result.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")

    def test_no_generation_prompt(self) -> None:
        messages = [{"role": "user", "content": "Hello"}]
        result = render_chat_template(messages, "llama", add_generation_prompt=False)
        assert not result.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")
        assert result.endswith("Hello<|eot_id|>")


# ---------------------------------------------------------------------------
# Qwen 3
# ---------------------------------------------------------------------------


class TestQwen3Template:
    def test_single_user_message(self) -> None:
        messages = [{"role": "user", "content": "Hello"}]
        result = render_chat_template(messages, "qwen3")
        expected = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        assert result == expected

    def test_system_and_user(self) -> None:
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        result = render_chat_template(messages, "qwen3")
        assert "<|im_start|>system\nYou are helpful.<|im_end|>" in result
        assert "<|im_start|>user\nHi<|im_end|>" in result
        assert result.endswith("<|im_start|>assistant\n")

    def test_no_generation_prompt(self) -> None:
        messages = [{"role": "user", "content": "Hello"}]
        result = render_chat_template(messages, "qwen3", add_generation_prompt=False)
        assert not result.endswith("<|im_start|>assistant\n")
        assert result.endswith("Hello<|im_end|>\n")


# ---------------------------------------------------------------------------
# Gemma 3
# ---------------------------------------------------------------------------


class TestGemma3Template:
    def test_single_user_message(self) -> None:
        messages = [{"role": "user", "content": "Hello"}]
        result = render_chat_template(messages, "gemma3_text")
        expected = "<bos><start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\n"
        assert result == expected

    def test_bos_token_present(self) -> None:
        messages = [{"role": "user", "content": "Hi"}]
        result = render_chat_template(messages, "gemma3_text")
        assert result.startswith("<bos>")

    def test_assistant_mapped_to_model(self) -> None:
        messages = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
            {"role": "user", "content": "Q2"},
        ]
        result = render_chat_template(messages, "gemma3_text")
        assert "<start_of_turn>model\nA<end_of_turn>" in result
        assert "<start_of_turn>assistant" not in result

    def test_system_message_folded_into_user(self) -> None:
        messages = [
            {"role": "system", "content": "You are a pirate."},
            {"role": "user", "content": "Hello"},
        ]
        result = render_chat_template(messages, "gemma3_text")
        # No system turn token.
        assert "<start_of_turn>system" not in result
        # System content prepended to first user turn.
        assert "<start_of_turn>user\nYou are a pirate.\n\nHello<end_of_turn>" in result

    def test_no_system_message(self) -> None:
        messages = [{"role": "user", "content": "Hello"}]
        result = render_chat_template(messages, "gemma3_text")
        assert "<start_of_turn>user\nHello<end_of_turn>" in result

    def test_no_generation_prompt(self) -> None:
        messages = [{"role": "user", "content": "Hello"}]
        result = render_chat_template(messages, "gemma3_text", add_generation_prompt=False)
        assert not result.endswith("<start_of_turn>model\n")
        assert result.endswith("Hello<end_of_turn>\n")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_unsupported_model_type(self) -> None:
        with pytest.raises(ValueError, match="No chat template"):
            render_chat_template([{"role": "user", "content": "Hi"}], "unsupported")


# ---------------------------------------------------------------------------
# Parity against transformers
# ---------------------------------------------------------------------------

# Llama 3 is excluded: HF's template always injects "Cutting Knowledge Date" /
# "Today Date" into the system prompt, which our simplified template omits by design.
_DEV_MODELS = [
    ("Qwen/Qwen3-1.7B", "qwen3"),
    ("google/gemma-3-1b-it", "gemma3_text"),
]


@pytest.mark.slow
@pytest.mark.parametrize("model_id,model_type", _DEV_MODELS, ids=["qwen3", "gemma3"])
def test_parity_with_transformers(model_id: str, model_type: str) -> None:
    """Verify our rendered output matches transformers.apply_chat_template."""
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as exc:
        pytest.skip(f"Could not load tokenizer for {model_id}: {exc}")

    # Use conversations WITH explicit system messages so Llama's HF template
    # doesn't inject its default "Cutting Knowledge Date" system prompt.
    conversations = [
        # System + user
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ],
        # System + multi-turn
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ],
    ]

    for messages in conversations:
        ours = render_chat_template(messages, model_type, add_generation_prompt=True)
        ref = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        assert ours == ref, (
            f"Template mismatch for {model_type}:\n  Ours: {ours!r}\n  Ref:  {ref!r}"
        )
