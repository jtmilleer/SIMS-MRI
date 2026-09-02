"""Exact six-component analytic SIREN Hessian drop-in for bending energy.

This module is intentionally opt-in: importing it does not patch project code.
Use ``install_benchmark_dropin`` or ``install_phase2_dropin`` from a launcher to
activate the context-aware upstream-compatible ``compute_bending_energy``.

The implementation reuses the already-validated SIREN topology extractor from
``sims_mri.optimization_candidates.phase2_bending_siren_analytic`` and propagates only the six
upper-triangular Hessian components consumed by the upstream bending loss:
``xx, yy, zz, xy, xz, yz``. This is an exact representation for the supported
SIREN affine+sine networks, not an approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import weakref

import torch
from torch import nn

from sims_mri.optimization_candidates.phase2_bending_siren_analytic import (
    RestoreHandle,
    StrictAnalyticContextError,
    UnsupportedSirenTopology,
    extract_siren_spec,
)


@dataclass
class _SirenForwardContext:
    input_ref: weakref.ReferenceType[torch.Tensor]
    model_ref: weakref.ReferenceType[nn.Module]
    model_type: type[nn.Module]


_CONTEXTS_BY_INPUT_ID: dict[int, _SirenForwardContext] = {}


def _validate_tensor_contract(
    input_coords: torch.Tensor,
    output: torch.Tensor | None = None,
    batch_size: int | None = None,
) -> None:
    if input_coords.ndim != 2 or input_coords.shape[1] != 3:
        raise ValueError(
            "terms6 analytic SIREN bending expects input_coords shaped [N, 3]; "
            f"got {tuple(input_coords.shape)}"
        )
    if output is not None:
        if output.ndim != 2 or output.shape[1] != 3:
            raise ValueError(
                "terms6 analytic SIREN bending expects output shaped [N, 3]; "
                f"got {tuple(output.shape)}"
            )
        if output.shape[0] != input_coords.shape[0]:
            raise ValueError(
                "input_coords and output must have the same N; "
                f"got {tuple(input_coords.shape)} and {tuple(output.shape)}"
            )
    if batch_size is None:
        raise ValueError("batch_size must be provided for analytic bending-energy scaling")


def _linear_apply_jacobian(weight: torch.Tensor, jacobian: torch.Tensor) -> torch.Tensor:
    # weight: [out, in], jacobian: [N, in, 3] -> [N, out, 3]
    return torch.einsum("oi,nia->noa", weight, jacobian)


def _linear_apply_hessian6(weight: torch.Tensor, hessian6: torch.Tensor) -> torch.Tensor:
    # weight: [out, in], hessian6: [N, in, 6] -> [N, out, 6]
    return torch.einsum("oi,nic->noc", weight, hessian6)


def _analytic_forward_hessian6(spec: Any, input_coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n_points = input_coords.shape[0]
    act = input_coords
    eye = torch.eye(3, device=input_coords.device, dtype=input_coords.dtype)
    jacobian = eye.unsqueeze(0).expand(n_points, 3, 3)
    hessian6 = torch.zeros(n_points, 3, 6, device=input_coords.device, dtype=input_coords.dtype)

    for layer in spec.layers:
        linear = layer.linear
        weight = linear.weight
        act_z = linear(act)
        jac_z = _linear_apply_jacobian(weight, jacobian)
        hess6_z = _linear_apply_hessian6(weight, hessian6)

        if layer.omega is None:
            act = act_z
            jacobian = jac_z
            hessian6 = hess6_z
            continue

        omega = input_coords.new_tensor(layer.omega)
        omega_z = omega * act_z
        sin_z = torch.sin(omega_z)
        c1 = omega * torch.cos(omega_z)
        c2 = -(omega * omega) * sin_z

        jx = jac_z[..., 0]
        jy = jac_z[..., 1]
        jz = jac_z[..., 2]
        outer6 = torch.stack(
            (
                jx * jx,  # xx
                jy * jy,  # yy
                jz * jz,  # zz
                jx * jy,  # xy
                jx * jz,  # xz
                jy * jz,  # yz
            ),
            dim=-1,
        )

        act = sin_z
        jacobian = c1.unsqueeze(-1) * jac_z
        hessian6 = c1.unsqueeze(-1) * hess6_z + c2.unsqueeze(-1) * outer6

    if spec.output_scale != 1.0:
        scale = input_coords.new_tensor(spec.output_scale)
        act = scale * act
        hessian6 = scale * hessian6
    return act, hessian6


def _bending_loss_from_hessian6(hessian6: torch.Tensor, batch_size: int) -> torch.Tensor:
    # hessian6: [N, output_channel, component], component order xx, yy, zz, xy, xz, yz
    dxx = hessian6[:, :, 0]
    dyy = hessian6[:, :, 1]
    dzz = hessian6[:, :, 2]
    dxy = hessian6[:, :, 3]
    dxz = hessian6[:, :, 4]
    dyz = hessian6[:, :, 5]

    loss = torch.mean(torch.square(dxx)) + torch.mean(torch.square(dyy)) + torch.mean(torch.square(dzz))
    loss = loss + 2 * torch.mean(torch.square(dxy)) + 2 * torch.mean(torch.square(dxz)) + torch.mean(torch.square(dyz))
    return loss / batch_size


def compute_bending_energy_for_siren_terms6(
    model: nn.Module,
    input_coords: torch.Tensor,
    batch_size: int,
    *,
    validate_forward: bool = False,
    reference_output: torch.Tensor | None = None,
    reference_mode: str = "auto",
) -> torch.Tensor:
    """Compute exact upstream-equivalent bending energy for a supported SIREN.

    The returned scalar is differentiable with respect to SIREN parameters and
    input coordinates. ``reference_mode`` is reserved for API symmetry with the
    full analytic candidate; this implementation validates by direct forward
    comparison when ``validate_forward`` is true.
    """

    del reference_mode
    _validate_tensor_contract(input_coords, batch_size=batch_size)
    spec = extract_siren_spec(model)
    analytic_output, hessian6 = _analytic_forward_hessian6(spec, input_coords)
    if analytic_output.ndim != 2 or analytic_output.shape != input_coords.shape:
        raise UnsupportedSirenTopology(
            f"terms6 analytic SIREN output must be [N, 3]; got {tuple(analytic_output.shape)}"
        )
    if hessian6.ndim != 3 or hessian6.shape[0] != input_coords.shape[0] or hessian6.shape[1:] != (3, 6):
        raise UnsupportedSirenTopology(
            f"terms6 analytic SIREN Hessian must be [N, 3, 6]; got {tuple(hessian6.shape)}"
        )
    if validate_forward:
        expected = reference_output if reference_output is not None else model(input_coords)
        atol = 1e-6 if input_coords.dtype == torch.float32 else 1e-10
        rtol = 1e-5 if input_coords.dtype == torch.float32 else 1e-10
        if not torch.allclose(analytic_output, expected, atol=atol, rtol=rtol):
            max_abs = torch.max(torch.abs(analytic_output - expected)).detach().cpu().item()
            raise UnsupportedSirenTopology(
                f"terms6 analytic forward mismatch for {spec.source}: max_abs={max_abs:.6g}"
            )
    return _bending_loss_from_hessian6(hessian6, int(batch_size))


def _register_forward_context(input_coords: Any, output: Any, model: nn.Module) -> None:
    if not isinstance(input_coords, torch.Tensor) or not isinstance(output, torch.Tensor):
        return
    if input_coords.ndim != 2 or output.ndim != 2:
        return
    if input_coords.shape[1] != 3 or output.shape[1] != 3 or input_coords.shape[0] != output.shape[0]:
        return
    try:
        extract_siren_spec(model)
    except UnsupportedSirenTopology:
        return
    _CONTEXTS_BY_INPUT_ID[id(input_coords)] = _SirenForwardContext(
        input_ref=weakref.ref(input_coords),
        model_ref=weakref.ref(model),
        model_type=type(model),
    )


def _lookup_context(input_coords: torch.Tensor) -> nn.Module:
    stale: list[int] = []
    for key, context in list(_CONTEXTS_BY_INPUT_ID.items()):
        tensor = context.input_ref()
        model = context.model_ref()
        if tensor is None or model is None:
            stale.append(key)
            continue
        if key == id(input_coords):
            if tensor is not input_coords:
                stale.append(key)
                break
            if input_coords.ndim != 2 or input_coords.shape[1] != 3:
                stale.append(key)
                break
            return model
    for key in stale:
        _CONTEXTS_BY_INPUT_ID.pop(key, None)
    raise StrictAnalyticContextError(
        "no supported SIREN forward context found for this exact input_coords tensor"
    )


def compute_bending_energy(
    input_coords: torch.Tensor,
    output: torch.Tensor,
    batch_size: int | None = None,
) -> torch.Tensor:
    """Upstream-compatible strict terms6 analytic drop-in.

    This function never silently falls back. The matching SIREN model must have
    been registered by an installer-patched forward call using this exact
    ``input_coords`` tensor.
    """

    _validate_tensor_contract(input_coords, output, batch_size)
    model = _lookup_context(input_coords)
    return compute_bending_energy_for_siren_terms6(model, input_coords, int(batch_size))


def install_phase2_dropin(*, strict: bool = True, fallback: Callable[..., torch.Tensor] | None = None) -> RestoreHandle:
    """Patch real IDIR phase-2 SIREN and regularizer functions in this process."""

    del fallback
    if not strict:
        raise ValueError("terms6 analytic SIREN drop-in only supports strict=True in this iteration")
    from IDIR.objectives import regularizers
    from IDIR.networks import networks

    previous_regularizer = regularizers.compute_bending_energy
    previous_forward = networks.Siren.forward

    def patched_forward(self: nn.Module, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        output = previous_forward(self, x, *args, **kwargs)
        _register_forward_context(x, output, self)
        return output

    regularizers.compute_bending_energy = compute_bending_energy
    networks.Siren.forward = patched_forward

    return RestoreHandle(
        [
            lambda: setattr(regularizers, "compute_bending_energy", previous_regularizer),
            lambda: setattr(networks.Siren, "forward", previous_forward),
        ]
    )


def install_benchmark_dropin(
    bench_module: Any,
    *,
    strict: bool = True,
    fallback: Callable[..., torch.Tensor] | None = None,
) -> RestoreHandle:
    """Patch the existing benchmark module to use the strict terms6 drop-in."""

    del fallback
    if not strict:
        raise ValueError("terms6 analytic SIREN benchmark drop-in only supports strict=True in this iteration")
    tiny_cls = getattr(bench_module, "TinySirenDisplacement")
    previous_compute = getattr(bench_module, "compute_bending_energy")
    previous_forward = tiny_cls.forward

    def patched_forward(self: nn.Module, coords: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        output = previous_forward(self, coords, *args, **kwargs)
        _register_forward_context(coords, output, self)
        return output

    bench_module.compute_bending_energy = compute_bending_energy
    tiny_cls.forward = patched_forward

    return RestoreHandle(
        [
            lambda: setattr(bench_module, "compute_bending_energy", previous_compute),
            lambda: setattr(tiny_cls, "forward", previous_forward),
        ]
    )


__all__ = [
    "StrictAnalyticContextError",
    "UnsupportedSirenTopology",
    "RestoreHandle",
    "compute_bending_energy",
    "compute_bending_energy_for_siren_terms6",
    "extract_siren_spec",
    "install_benchmark_dropin",
    "install_phase2_dropin",
]
