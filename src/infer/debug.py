"""Layer-by-layer activation diff tooling.

Compares our model's intermediate activations against a HuggingFace
``transformers`` reference model.  Intended for interactive debugging
and integration tests, not production inference.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass
class LayerDiff:
    """Activation diff for a single layer."""

    layer_index: int
    max_abs_error: float
    mean_abs_error: float
    our_norm: float
    ref_norm: float


@dataclass
class ModelDiff:
    """Full model activation diff report."""

    embedding_diff: LayerDiff
    layer_diffs: list[LayerDiff]
    final_norm_diff: LayerDiff
    logits_diff: LayerDiff


def _compute_diff(ours: Tensor, ref: Tensor, index: int) -> LayerDiff:
    """Compute error metrics between two tensors."""
    diff = (ours.float() - ref.float()).abs()
    return LayerDiff(
        layer_index=index,
        max_abs_error=diff.max().item(),
        mean_abs_error=diff.mean().item(),
        our_norm=ours.float().norm().item(),
        ref_norm=ref.float().norm().item(),
    )


def compare_models(
    our_model: nn.Module,
    ref_model: nn.Module,
    input_ids: Tensor,
) -> ModelDiff:
    """Compare activations between our model and a reference model.

    Registers forward hooks on both models to capture intermediate
    activations at each layer boundary.  Runs the same input through
    both and computes per-layer error metrics.

    Args:
        our_model: Our model instance (LlamaModel, Qwen3Model, or
            Gemma3Model).
        ref_model: HuggingFace transformers model
            (``AutoModelForCausalLM``).
        input_ids: Token IDs, shape ``[batch, seq_len]``.

    Returns:
        A :class:`ModelDiff` with per-layer error metrics.
    """
    our_layers: list[Tensor] = []
    ref_layers: list[Tensor] = []
    our_embed: list[Tensor] = []
    ref_embed: list[Tensor] = []
    our_norm_out: list[Tensor] = []
    ref_norm_out: list[Tensor] = []
    handles: list[torch.utils.hooks.RemovableHandle] = []

    num_layers = len(list(our_model.layers))  # type: ignore[union-attr, arg-type]

    # --- Hook helpers ---
    def _make_capture_hook(
        storage: list[Tensor],
    ) -> Callable[[nn.Module, tuple[Any, ...], Any], None]:
        def hook(_module: nn.Module, _input: tuple[Any, ...], output: Any) -> None:
            if isinstance(output, tuple):
                storage.append(output[0].detach())
            else:
                storage.append(output.detach())

        return hook

    def _make_pre_hook(
        storage: list[Tensor],
    ) -> Callable[[nn.Module, tuple[Any, ...]], None]:
        def hook(_module: nn.Module, args: tuple[Any, ...]) -> None:
            if len(args) > 0:
                storage.append(args[0].detach())

        return hook

    try:
        # Embedding: pre-hook on first layer captures post-embedding input.
        handles.append(
            our_model.layers[0].register_forward_pre_hook(_make_pre_hook(our_embed))  # type: ignore[union-attr, index]
        )
        handles.append(
            ref_model.model.layers[0].register_forward_pre_hook(  # type: ignore[union-attr, index]
                _make_pre_hook(ref_embed)
            )
        )

        # Per-layer post-hooks.
        for i in range(num_layers):
            handles.append(
                our_model.layers[i].register_forward_hook(  # type: ignore[union-attr, index]
                    _make_capture_hook(our_layers)
                )
            )
            handles.append(
                ref_model.model.layers[i].register_forward_hook(  # type: ignore[union-attr, index]
                    _make_capture_hook(ref_layers)
                )
            )

        # Final norm post-hook.
        handles.append(
            our_model.norm.register_forward_hook(_make_capture_hook(our_norm_out))  # type: ignore[union-attr]
        )
        handles.append(
            ref_model.model.norm.register_forward_hook(  # type: ignore[union-attr]
                _make_capture_hook(ref_norm_out)
            )
        )

        # Forward pass.
        with torch.no_grad():
            our_logits = our_model(input_ids)
            ref_output = ref_model(input_ids)
            ref_logits = ref_output.logits if hasattr(ref_output, "logits") else ref_output

    finally:
        for h in handles:
            h.remove()

    # Compute diffs.
    embedding_diff = _compute_diff(our_embed[0], ref_embed[0], -1)

    layer_diffs = []
    for i in range(min(len(our_layers), len(ref_layers))):
        layer_diffs.append(_compute_diff(our_layers[i], ref_layers[i], i))

    final_norm_diff = _compute_diff(our_norm_out[0], ref_norm_out[0], -2)
    logits_diff = _compute_diff(our_logits, ref_logits, -3)

    return ModelDiff(
        embedding_diff=embedding_diff,
        layer_diffs=layer_diffs,
        final_norm_diff=final_norm_diff,
        logits_diff=logits_diff,
    )


def format_diff(diff: ModelDiff) -> str:
    """Format a :class:`ModelDiff` as a readable table.

    Args:
        diff: The diff to format.

    Returns:
        A multi-line string with aligned columns.
    """
    header = f"{'Layer':<14s} {'Max Abs Err':>12s} {'Mean Abs Err':>13s} {'Our Norm':>12s} {'Ref Norm':>12s}"
    sep = "-" * len(header)
    lines = [header, sep]

    def _row(label: str, d: LayerDiff) -> str:
        return (
            f"{label:<14s} {d.max_abs_error:>12.4e} {d.mean_abs_error:>13.4e} "
            f"{d.our_norm:>12.1f} {d.ref_norm:>12.1f}"
        )

    lines.append(_row("embed", diff.embedding_diff))
    for d in diff.layer_diffs:
        lines.append(_row(f"layer_{d.layer_index}", d))
    lines.append(_row("final_norm", diff.final_norm_diff))
    lines.append(_row("logits", diff.logits_diff))

    return "\n".join(lines)
