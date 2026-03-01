"""Integration sanity-check: structured JSON output with a real model.

Loads Llama-3.2-1B-Instruct and generates with a JSON schema constraint,
verifying the output is valid JSON matching the schema.
"""

from __future__ import annotations

import asyncio
import json

import pytest
import torch

from infer.engine.config import EngineConfig
from infer.engine.engine import Engine
from infer.engine.request import StepOutput
from infer.engine.sampler import SamplingParams


def _drain_engine(engine: Engine, queue: asyncio.Queue[StepOutput]) -> str:
    """Run engine steps until the request finishes, return concatenated text."""
    pieces: list[str] = []
    while engine.has_work():
        engine.step()
        while not queue.empty():
            out = queue.get_nowait()
            pieces.append(out.text_delta)
            if out.finished:
                return "".join(pieces)
    return "".join(pieces)


@pytest.mark.slow
def test_structured_json_boolean() -> None:
    """JSON schema {type: boolean} should produce 'true' or 'false'."""
    config = EngineConfig(
        model="meta-llama/Llama-3.2-1B-Instruct",
        batching_mode="continuous",
        kv_cache_backend="paged",
        max_seq_len=256,
        max_batch_size=1,
        seed=42,
    )
    try:
        engine = Engine(config)
    except Exception as exc:
        pytest.skip(f"Could not load model: {exc}")

    queue: asyncio.Queue[StepOutput] = asyncio.Queue()
    params = SamplingParams(temperature=0.0, max_new_tokens=10)
    response_format = {"type": "json_schema", "schema": {"type": "boolean"}}

    engine.add_request(
        "bool-1",
        "Is the sky blue? Answer true or false:",
        params,
        queue,
        response_format=response_format,
    )
    text = _drain_engine(engine, queue)

    print(f"\n  JSON boolean output: {text!r}")
    parsed = json.loads(text)
    assert isinstance(parsed, bool), f"Expected bool, got {type(parsed).__name__}: {parsed!r}"

    del engine
    torch.cuda.empty_cache()


@pytest.mark.slow
def test_structured_json_object() -> None:
    """JSON schema for {name: string, age: integer} should produce valid JSON."""
    config = EngineConfig(
        model="meta-llama/Llama-3.2-1B-Instruct",
        batching_mode="continuous",
        kv_cache_backend="paged",
        max_seq_len=256,
        max_batch_size=1,
        seed=42,
    )
    try:
        engine = Engine(config)
    except Exception as exc:
        pytest.skip(f"Could not load model: {exc}")

    queue: asyncio.Queue[StepOutput] = asyncio.Queue()
    params = SamplingParams(temperature=0.0, max_new_tokens=50)
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
    }
    response_format = {"type": "json_schema", "schema": schema}

    engine.add_request(
        "obj-1",
        "Extract: Alice is 30 years old. Output JSON:",
        params,
        queue,
        response_format=response_format,
    )
    text = _drain_engine(engine, queue)

    print(f"\n  JSON object output: {text!r}")
    parsed = json.loads(text)
    assert isinstance(parsed, dict), f"Expected dict, got {type(parsed).__name__}"
    assert "name" in parsed, f"Missing 'name' key: {parsed}"
    assert "age" in parsed, f"Missing 'age' key: {parsed}"
    assert isinstance(parsed["name"], str), (
        f"'name' should be str, got {type(parsed['name']).__name__}"
    )
    assert isinstance(parsed["age"], int), (
        f"'age' should be int, got {type(parsed['age']).__name__}"
    )

    del engine
    torch.cuda.empty_cache()


@pytest.mark.slow
def test_structured_regex() -> None:
    """Regex pattern (yes|no) should produce exactly 'yes' or 'no'."""
    config = EngineConfig(
        model="meta-llama/Llama-3.2-1B-Instruct",
        batching_mode="continuous",
        kv_cache_backend="paged",
        max_seq_len=256,
        max_batch_size=1,
        seed=42,
    )
    try:
        engine = Engine(config)
    except Exception as exc:
        pytest.skip(f"Could not load model: {exc}")

    queue: asyncio.Queue[StepOutput] = asyncio.Queue()
    params = SamplingParams(temperature=0.0, max_new_tokens=10)
    response_format = {"type": "regex", "pattern": "(yes|no)"}

    engine.add_request(
        "regex-1", "Is water wet? Answer yes or no:", params, queue, response_format=response_format
    )
    text = _drain_engine(engine, queue)

    print(f"\n  Regex output: {text!r}")
    assert text in ("yes", "no"), f"Expected 'yes' or 'no', got {text!r}"

    del engine
    torch.cuda.empty_cache()
