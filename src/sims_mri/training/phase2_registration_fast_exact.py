"""Opt-in fast/exact Phase-2 registration launcher.

This additive wrapper composes two exact process-local drop-ins before
running :mod:`sims_mri.training.phase2_registration` unchanged:

* frozen Phase-1 eval hooks from ``phase2_registration_freeze_frozen``;
* strict exact SIREN bending-energy drop-ins from
  ``sims_mri.optimization_candidates``.

The default backend policy mirrors the production Phase-2 device preference:
CUDA if available, else MPS if available, else CPU.  CPU/CUDA select the exact
``terms6`` backend; MPS selects the exact ``analytic_siren`` backend.  Override
for validation with::

    SIMS_MRI_PHASE2_BENDING_BACKEND=auto|terms6|analytic_siren

Invoke with the same arguments as the original entrypoint::

    PYTHONPATH=src:. python -m sims_mri.training.phase2_registration_fast_exact ...

This wrapper does not implement new bending math, does not use ``detach()``,
``torch.no_grad()``, or inference mode, and restores all monkeypatches in a
``finally`` block when the delegated production CLI exits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import json
import os
from typing import Any, Literal

import torch

from sims_mri.training.phase2_registration_freeze_frozen import (
    FreezeEvalHandle,
    install_phase2_eval_freeze_dropin,
)

BackendName = Literal["terms6", "analytic_siren"]
RequestedBackend = Literal["auto", "terms6", "analytic_siren"]
DeviceKind = Literal["cuda", "mps", "cpu"]

_BACKEND_ENV = "SIMS_MRI_PHASE2_BENDING_BACKEND"
_VALID_BACKENDS = {"auto", "terms6", "analytic_siren"}
_DEVICE_POLICY = "cuda/cpu->terms6,mps->analytic_siren"
_FREEZE_INSTALLER = (
    "sims_mri.training.phase2_registration_freeze_frozen."
    "install_phase2_eval_freeze_dropin"
)


@dataclass(frozen=True)
class BackendSelection:
    """Resolved exact bending backend policy for one wrapper process."""

    requested_backend: RequestedBackend
    bending_backend: BackendName
    selected_device_kind: DeviceKind
    bending_installer: str
    strict: bool
    selection_source: str
    device_policy: str = _DEVICE_POLICY


@dataclass
class FastExactHandle:
    """Owns the composed process-local freeze and bending monkeypatches."""

    freeze_handle: FreezeEvalHandle
    bending_handle: Any
    selection: BackendSelection
    restored: bool = False

    def restore(self) -> None:
        """Restore bending then freeze hooks once, in reverse install order."""

        if self.restored:
            return
        try:
            self.bending_handle.restore()
        finally:
            self.freeze_handle.restore()
            self.restored = True


def _mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def _resolve_device_kind() -> DeviceKind:
    """Mirror production Phase-2 device preference without allocating models."""

    if torch.cuda.is_available():
        return "cuda"
    if _mps_available():
        return "mps"
    return "cpu"


def _normalize_requested_backend(requested_backend: str | None) -> tuple[RequestedBackend, str]:
    if requested_backend is None:
        raw_value = os.environ.get(_BACKEND_ENV)
        source = "env" if raw_value is not None else "default"
        requested = "auto" if raw_value is None else raw_value.strip().lower()
    else:
        source = "argument"
        requested = requested_backend.strip().lower()

    if requested not in _VALID_BACKENDS:
        valid = ", ".join(sorted(_VALID_BACKENDS))
        raise ValueError(f"{_BACKEND_ENV} must be one of {valid}; got {requested!r}")
    return requested, source  # type: ignore[return-value]


def _select_bending_backend(requested_backend: str | None = None) -> BackendSelection:
    """Select the exact SIREN bending backend for this process."""

    requested, source = _normalize_requested_backend(requested_backend)
    device_kind = _resolve_device_kind()
    if requested == "auto":
        backend: BackendName = "analytic_siren" if device_kind == "mps" else "terms6"
    else:
        backend = requested

    if backend == "terms6":
        installer = "sims_mri.optimization_candidates.phase2_bending_siren_terms6.install_phase2_dropin"
    else:
        installer = "sims_mri.optimization_candidates.phase2_bending_siren_analytic.install_phase2_dropin"

    return BackendSelection(
        requested_backend=requested,
        bending_backend=backend,
        selected_device_kind=device_kind,
        bending_installer=installer,
        strict=True,
        selection_source=source,
    )


def _install_bending_dropin(selection: BackendSelection):  # type: ignore[no-untyped-def]
    """Install the selected strict bending drop-in and return its restore handle."""

    module_name, function_name = selection.bending_installer.rsplit(".", 1)
    module = importlib.import_module(module_name)
    installer = getattr(module, function_name)
    return installer(strict=True)


def _emit_audit(selection: BackendSelection) -> None:
    payload = {
        **asdict(selection),
        "freeze_installer": _FREEZE_INSTALLER,
    }
    print("PHASE2_FAST_EXACT " + json.dumps(payload, sort_keys=True))


def install_phase2_fast_exact_dropins(requested_backend: str | None = None) -> FastExactHandle:
    """Install freeze + exact bending hooks for an opt-in Phase-2 process.

    The returned handle must be restored by the caller.  The function fails
    closed: invalid backend values and strict backend install failures propagate;
    any already-installed freeze hook is restored before the exception escapes.
    """

    selection = _select_bending_backend(requested_backend)
    freeze_handle = install_phase2_eval_freeze_dropin()
    try:
        bending_handle = _install_bending_dropin(selection)
    except Exception:
        freeze_handle.restore()
        raise

    handle = FastExactHandle(
        freeze_handle=freeze_handle,
        bending_handle=bending_handle,
        selection=selection,
    )
    _emit_audit(selection)
    return handle


def _cli() -> None:
    """Run the original Phase-2 CLI with the fast/exact hooks installed."""

    handle = install_phase2_fast_exact_dropins()
    try:
        from sims_mri.training import phase2_registration

        phase2_registration._cli()
    finally:
        handle.restore()


if __name__ == "__main__":
    _cli()
