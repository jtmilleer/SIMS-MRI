"""Phase-2 registration wrapper that freezes non-optimized Phase-1 params.

This module is an additive drop-in around :mod:`phase2_registration`.  It does
not alter the Phase-2 forward graph, use ``torch.no_grad()``, detach tensors, or
enter inference mode.  It only sets ``requires_grad=False`` on parameters of the
Phase-1 modules that the production Phase-2 entrypoint already switches to
``eval()`` while optimizing only the registration model.

Invoke with the same arguments as the original entrypoint:

    PYTHONPATH=src:. python -m sims_mri.training.phase2_registration_freeze_frozen ...
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Callable

import torch.nn as nn

from IDIR.networks import networks
from multi_contrast_inr.model import MLPv1
from sims_mri.hash_encoder_wrapper_freenerf import HashEncoderWrapper


@dataclass(frozen=True)
class FreezeEvent:
    """Audit record for one eval-triggered parameter freeze."""

    module_type: str
    module_id: int
    reason: str
    param_count: int
    trainable_before: int
    trainable_after: int


@dataclass
class FreezeEvalHandle:
    """Owns process-local eval() monkeypatches and freeze audit events."""

    restore_callbacks: list[Callable[[], None]] = field(default_factory=list)
    events: list[FreezeEvent] = field(default_factory=list)
    restored: bool = False

    def restore(self) -> None:
        """Restore patched class methods once, in reverse install order."""

        if self.restored:
            return
        for restore in reversed(self.restore_callbacks):
            restore()
        self.restored = True


def _module_type(module: nn.Module) -> str:
    cls = type(module)
    return f"{cls.__module__}.{cls.__qualname__}"


def freeze_module_parameters(module: nn.Module, *, reason: str) -> FreezeEvent:
    """Freeze module parameters while preserving gradients through operations.

    Parameters are treated as constants by autograd after this call, but the
    module's forward operations remain differentiable with respect to their
    inputs.  Stale ``.grad`` tensors are cleared to avoid carrying Phase-1 or
    earlier eval-call gradients into Phase 2.
    """

    param_count = 0
    trainable_before = 0
    for param in module.parameters():
        param_count += param.numel()
        if param.requires_grad:
            trainable_before += param.numel()
        param.requires_grad_(False)
        param.grad = None

    trainable_after = sum(param.numel() for param in module.parameters() if param.requires_grad)
    return FreezeEvent(
        module_type=_module_type(module),
        module_id=id(module),
        reason=reason,
        param_count=param_count,
        trainable_before=trainable_before,
        trainable_after=trainable_after,
    )


def _emit_event(event: FreezeEvent) -> None:
    print("PHASE2_FREEZE_FROZEN " + json.dumps(asdict(event), sort_keys=True))


def _idir_siren_final_out_features(module: networks.Siren) -> int | None:
    layers = getattr(module, "layers", None)
    if layers is None:
        return None
    try:
        final_layer = layers[-1]
    except (IndexError, TypeError):
        return None
    return getattr(final_layer, "out_features", None)


def _has_trainable_parameters(module: nn.Module) -> bool:
    return any(param.requires_grad for param in module.parameters())


def _patch_eval(
    handle: FreezeEvalHandle,
    cls: type[nn.Module],
    *,
    reason: str,
    should_freeze: Callable[[nn.Module], bool] | None = None,
) -> None:
    original_eval = cls.eval

    def eval_with_freeze(self: nn.Module):  # type: ignore[no-untyped-def]
        result = original_eval(self)
        if (should_freeze is None or should_freeze(self)) and _has_trainable_parameters(self):
            event = freeze_module_parameters(self, reason=reason)
            handle.events.append(event)
            _emit_event(event)
        return result

    cls.eval = eval_with_freeze  # type: ignore[method-assign]
    handle.restore_callbacks.append(lambda: setattr(cls, "eval", original_eval))


def install_phase2_eval_freeze_dropin() -> FreezeEvalHandle:
    """Install process-local eval hooks for Phase-1 modules used in Phase 2."""

    handle = FreezeEvalHandle()
    _patch_eval(
        handle,
        HashEncoderWrapper,
        reason="phase2_hash_encoder_eval",
    )
    _patch_eval(
        handle,
        MLPv1,
        reason="phase2_generation_mlp_eval",
    )
    _patch_eval(
        handle,
        networks.Siren,
        reason="phase2_generation_siren_eval",
        should_freeze=lambda module: _idir_siren_final_out_features(module) == 1,
    )
    return handle


def _cli() -> None:
    handle = install_phase2_eval_freeze_dropin()
    try:
        from sims_mri.training import phase2_registration

        phase2_registration._cli()
    finally:
        handle.restore()


if __name__ == "__main__":
    _cli()
