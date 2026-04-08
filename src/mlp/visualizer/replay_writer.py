"""Write replay_manifest.json and replay_steps.jsonl for the lesson visualizer."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mlp.visualizer.replay_schemas import (
    REPLAY_SCHEMA_VERSION,
    ReplayManifest,
    ReplayStep,
    TocEntry,
    replay_step_to_jsonable,
)

_REPLAY_STEP_FIELDS = frozenset(ReplayStep.model_fields.keys())

ExplanationFn = Callable[[dict[str, Any]], str]


def default_explanation(event: dict[str, Any]) -> str:
    phase = event.get("phase", "")
    layer = event.get("layer")
    toc = event.get("toc_id", "")
    if phase == "init":
        return (
            "Training run recorded for step-by-step replay. Each arrow key moves one "
            "micro-step through forward pass, loss, backward pass, and the optimizer."
        )
    if phase == "forward_input":
        return (
            "Forward pass starts from the input features for the highlighted example."
        )
    if phase == "forward_layer":
        li = "" if layer is None else f" {int(layer)}"
        return (
            f"Linear layer{li}: we compute z = Wᵀa + b (shown as weighted sums along edges). "
            "Each edge carries one term aⱼ·Wⱼᵢ into neuron i."
        )
    if phase == "activation":
        return (
            "Nonlinearity squashes pre-activations z into activations a. "
            "This is where the network can bend the decision boundary."
        )
    if phase == "loss":
        return (
            "Loss compares model probabilities to the true label. "
            "A wrong prediction pushes larger error and stronger learning signals."
        )
    if phase == "backward_layer":
        return (
            "Backpropagation routes influence backward: local derivatives chain together "
            "so we know how sensitive the loss is to each weight and bias."
        )
    if phase == "optimizer":
        opt = event.get("optimizer_name") or event.get("optimizer", "sgd")
        if opt == "rmsprop":
            return (
                "RMSprop scales each update by a running average of squared gradients, "
                "so frequent directions take smaller effective steps."
            )
        return (
            "Gradient descent nudges every parameter opposite the gradient, "
            "scaled by the learning rate η."
        )
    if phase == "batch_end":
        return "End of this batch; weights are updated for the next example or batch."
    if phase == "epoch_end":
        return "End of epoch: the full training set was visited once in shuffled order."
    return toc.replace("_", " ")


def tabular_explanation(event: dict[str, Any]) -> str:
    """Plain-language copy for real (tabular) feature runs — non-engineer audience."""
    phase = event.get("phase", "")
    layer = event.get("layer")
    if phase == "init":
        anchor = event.get("lesson_anchor_train_index")
        base = (
            "You will see one fixed training example move through the network: first predictions "
            "flow left to right, then we measure the mistake, then learning flows right to left, "
            "and finally the knobs (weights) get a small adjustment."
        )
        if anchor is not None:
            base += (
                f" The replay always follows training row index {int(anchor)} (order in the training "
                "matrix before shuffling). When that row is not in the current minibatch, forward/loss/"
                "backward micro-steps are skipped for that batch, but the optimizer still updates from "
                "the full batch."
            )
        return base
    if phase == "forward_input":
        return (
            "Each number here is one input feature for this patient record (scaled). "
            "They are the starting point for everything that follows."
        )
    if phase == "forward_layer":
        li = "" if layer is None else f" {int(layer) + 1}"
        return (
            f"Layer{li}: each neuron adds up weighted signals from the previous column, like a vote "
            "where some connections count more than others. The thickness and color of lines hint at "
            "how strong each link is."
        )
    if phase == "activation":
        return (
            "A gentle curve (or hinge for ReLU) keeps numbers in a useful range so the network can "
            "represent non-straight patterns, not just straight lines."
        )
    if phase == "loss":
        return (
            "Here we score how far the model’s guess is from the true label. "
            "Bigger mismatch means the model will try harder to adjust on the next update."
        )
    if phase == "backward_layer":
        return (
            "Learning walks backward: we ask how much each connection contributed to the mistake, "
            "so we know which weights deserve the most blame."
        )
    if phase == "optimizer":
        opt = event.get("optimizer_name") or event.get("optimizer", "sgd")
        if opt == "rmsprop":
            return (
                "The optimizer nudges each weight. With RMSprop, frequent big swings are smoothed out "
                "so updates stay steadier."
            )
        return (
            "The optimizer takes a small step opposite the error signal for each weight — "
            "like walking downhill on a hazy landscape."
        )
    if phase == "batch_end":
        return "That batch is done; the weights you see are now the updated ones for the next step."
    if phase == "epoch_end":
        return "One full pass over the training data is complete. The next epoch shuffles the order again."
    return default_explanation(event)


class LessonReplayWriter:
    """Collects ReplayStep records and writes manifest + JSONL."""

    def __init__(
        self,
        out_dir: Path,
        *,
        run_id: str,
        explain: ExplanationFn | None = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.run_id = run_id
        self._explain: ExplanationFn = explain or default_explanation
        self.steps: list[ReplayStep] = []
        self._toc_first: dict[str, int] = {}
        self.batches_per_epoch: list[int] = []
        self._current_epoch_batches: int = 0

    def emit_raw(self, event: dict[str, Any]) -> None:
        phase = event["phase"]
        toc_id = str(event.get("toc_id", phase))
        idx = len(self.steps)
        if toc_id not in self._toc_first:
            self._toc_first[toc_id] = idx
        explanation = str(event.get("explanation") or self._explain(event))
        math_val = event.get("math")
        math = None if math_val is None else str(math_val)
        payload = {
            k: v
            for k, v in event.items()
            if k in _REPLAY_STEP_FIELDS
            and k
            not in (
                "step_index",
                "explanation",
                "math",
            )
        }
        step = ReplayStep(
            step_index=idx,
            explanation=explanation,
            math=math,
            **payload,
        )
        self.steps.append(step)

    def note_batch_completed(self) -> None:
        self._current_epoch_batches += 1

    def end_epoch(self) -> None:
        self.batches_per_epoch.append(self._current_epoch_batches)
        self._current_epoch_batches = 0

    def write(
        self,
        manifest: ReplayManifest,
    ) -> ReplayManifest:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        toc_entries = [
            TocEntry(toc_id=tid, label=_toc_label(tid), step_index=self._toc_first[tid])
            for tid in sorted(self._toc_first.keys(), key=lambda t: self._toc_first[t])
        ]
        m = manifest.model_copy(
            update={
                "total_micro_steps": len(self.steps),
                "toc_entries": toc_entries,
                "batches_per_epoch": self.batches_per_epoch,
            },
        )
        if m.schema_version != REPLAY_SCHEMA_VERSION:
            raise ValueError("schema mismatch")
        manifest_path = self.out_dir / "replay_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(m.model_dump_json(indent=2))
        steps_path = self.out_dir / manifest.steps_file
        with open(steps_path, "w", encoding="utf-8") as f:
            for s in self.steps:
                f.write(json.dumps(replay_step_to_jsonable(s), allow_nan=False))
                f.write("\n")
        return m


def _toc_label(toc_id: str) -> str:
    mapping = {
        "init": "Initialization",
        "forward_input": "Forward: inputs",
        "loss": "Loss",
        "optimizer": "Optimizer update",
        "batch_end": "Batch end",
        "epoch_end": "Epoch end",
    }
    if toc_id in mapping:
        return mapping[toc_id]
    if toc_id.startswith("fwd_l"):
        return f"Forward: layer {toc_id.removeprefix('fwd_l')}"
    if toc_id.startswith("act_l"):
        return f"Activation: layer {toc_id.removeprefix('act_l')}"
    if toc_id.startswith("bp_l"):
        return f"Backward: layer {toc_id.removeprefix('bp_l')}"
    return toc_id.replace("_", " ").title()
