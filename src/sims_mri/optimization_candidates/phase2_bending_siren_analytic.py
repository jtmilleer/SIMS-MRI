"""Exact analytic SIREN Hessian drop-in for phase-2 bending energy.

This module is intentionally opt-in: importing it does not patch project code.
Use ``install_benchmark_dropin`` or ``install_phase2_dropin`` from a launcher to
activate the context-aware upstream-compatible ``compute_bending_energy``.

The implementation is deliberately conservative. It supports only the SIREN
layouts that can be mapped exactly in this repository:

* ``benchmarks.bench_bending_energy.TinySirenDisplacement``: a Sequential of
  sine-affine layers followed by a final Linear, scaled by the forward method's
  fixed ``0.05`` factor.
* ``IDIR.networks.networks.Siren``: ``self.layers`` is a Sequential of Linear
  modules, all but the last wrapped by ``sin(self.omega * linear(x))``.

Unsupported models fail closed instead of approximating or falling back.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Any, Callable
import weakref

import torch
from torch import nn


@dataclass(frozen=True)
class _LayerSpec:
    linear: nn.Linear
    omega: float | None


@dataclass(frozen=True)
class _SirenSpec:
    layers: tuple[_LayerSpec, ...]
    output_scale: float
    source: str


@dataclass
class _SirenForwardContext:
    input_ref: weakref.ReferenceType[torch.Tensor]
    model_ref: weakref.ReferenceType[nn.Module]
    model_type: type[nn.Module]


class UnsupportedSirenTopology(RuntimeError):
    """Raised when a model cannot be mapped to the exact analytic SIREN spec."""


class StrictAnalyticContextError(RuntimeError):
    """Raised when the drop-in cannot find a valid registered SIREN context."""


_CONTEXTS_BY_INPUT_ID: dict[int, _SirenForwardContext] = {}


def _validate_tensor_contract(
    input_coords: torch.Tensor,
    output: torch.Tensor | None = None,
    batch_size: int | None = None,
) -> None:
    if input_coords.ndim != 2 or input_coords.shape[1] != 3:
        raise ValueError(
            "analytic SIREN bending expects input_coords shaped [N, 3]; "
            f"got {tuple(input_coords.shape)}"
        )
    if output is not None:
        if output.ndim != 2 or output.shape[1] != 3:
            raise ValueError(
                "analytic SIREN bending expects output shaped [N, 3]; "
                f"got {tuple(output.shape)}"
            )
        if output.shape[0] != input_coords.shape[0]:
            raise ValueError(
                "input_coords and output must have the same N; "
                f"got {tuple(input_coords.shape)} and {tuple(output.shape)}"
            )
    if batch_size is None:
        raise ValueError("batch_size must be provided for analytic bending-energy scaling")


def _is_linear(module: nn.Module) -> bool:
    return isinstance(module, nn.Linear)


def _is_tiny_siren_layer(module: nn.Module) -> bool:
    return hasattr(module, "linear") and isinstance(getattr(module, "linear"), nn.Linear) and hasattr(module, "omega")


def _extract_tiny_siren(model: nn.Module) -> _SirenSpec | None:
    if model.__class__.__name__ != "TinySirenDisplacement":
        return None
    net = getattr(model, "net", None)
    if not isinstance(net, nn.Sequential):
        raise UnsupportedSirenTopology("TinySirenDisplacement.net is not nn.Sequential")

    modules = list(net)
    if len(modules) < 2:
        raise UnsupportedSirenTopology("TinySirenDisplacement.net is too short")
    if not _is_linear(modules[-1]):
        raise UnsupportedSirenTopology("TinySirenDisplacement final module is not nn.Linear")

    layers: list[_LayerSpec] = []
    for idx, module in enumerate(modules[:-1]):
        if not _is_tiny_siren_layer(module):
            raise UnsupportedSirenTopology(
                f"TinySirenDisplacement layer {idx} is not an exact SirenLayer-like module"
            )
        omega = float(getattr(module, "omega"))
        layers.append(_LayerSpec(linear=getattr(module, "linear"), omega=omega))
    layers.append(_LayerSpec(linear=modules[-1], omega=None))

    if layers[0].linear.in_features != 3 or layers[-1].linear.out_features != 3:
        raise UnsupportedSirenTopology(
            "TinySirenDisplacement must map 3D coordinates to 3D displacements"
        )
    return _SirenSpec(tuple(layers), output_scale=0.05, source="TinySirenDisplacement")


def _extract_idir_siren(model: nn.Module) -> _SirenSpec | None:
    if model.__class__.__name__ != "Siren" or model.__class__.__module__ != "IDIR.networks.networks":
        return None
    layers_obj = getattr(model, "layers", None)
    if not isinstance(layers_obj, nn.Sequential):
        raise UnsupportedSirenTopology("IDIR Siren.layers is not nn.Sequential")
    modules = list(layers_obj)
    if len(modules) < 2:
        raise UnsupportedSirenTopology("IDIR Siren.layers is too short")
    if not all(isinstance(module, nn.Linear) for module in modules):
        bad = [type(module).__name__ for module in modules]
        raise UnsupportedSirenTopology(f"IDIR Siren.layers must contain only nn.Linear modules; got {bad}")
    omega = getattr(model, "omega", None)
    if omega is None:
        raise UnsupportedSirenTopology("IDIR Siren has no omega attribute")
    if int(getattr(model, "n_layers", len(modules))) != len(modules):
        raise UnsupportedSirenTopology(
            f"IDIR Siren n_layers={getattr(model, 'n_layers', None)} does not match len(layers)={len(modules)}"
        )

    layers: list[_LayerSpec] = [_LayerSpec(linear=module, omega=float(omega)) for module in modules[:-1]]
    layers.append(_LayerSpec(linear=modules[-1], omega=None))
    if layers[0].linear.in_features != 3 or layers[-1].linear.out_features != 3:
        raise UnsupportedSirenTopology(
            "IDIR Siren analytic bending only supports 3D input and 3D output"
        )
    return _SirenSpec(tuple(layers), output_scale=1.0, source="IDIR.networks.networks.Siren")


def extract_siren_spec(model: nn.Module) -> _SirenSpec:
    """Return an exact analytic spec for a supported SIREN-like module."""

    spec = _extract_tiny_siren(model)
    if spec is not None:
        return spec
    spec = _extract_idir_siren(model)
    if spec is not None:
        return spec
    raise UnsupportedSirenTopology(
        f"unsupported SIREN topology: {model.__class__.__module__}.{model.__class__.__name__}"
    )


def _linear_apply_jacobian(weight: torch.Tensor, jacobian: torch.Tensor) -> torch.Tensor:
    # weight: [out, in], jacobian: [N, in, 3] -> [N, out, 3]
    return torch.einsum("oi,nia->noa", weight, jacobian)


def _linear_apply_hessian(weight: torch.Tensor, hessian: torch.Tensor) -> torch.Tensor:
    # weight: [out, in], hessian: [N, in, 3, 3] -> [N, out, 3, 3]
    return torch.einsum("oi,niab->noab", weight, hessian)


def _analytic_forward_hessian(spec: _SirenSpec, input_coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n_points = input_coords.shape[0]
    act = input_coords
    eye = torch.eye(3, device=input_coords.device, dtype=input_coords.dtype)
    jacobian = eye.unsqueeze(0).expand(n_points, 3, 3)
    hessian = torch.zeros(n_points, 3, 3, 3, device=input_coords.device, dtype=input_coords.dtype)

    for layer in spec.layers:
        linear = layer.linear
        weight = linear.weight
        act_z = linear(act)
        jac_z = _linear_apply_jacobian(weight, jacobian)
        hess_z = _linear_apply_hessian(weight, hessian)

        if layer.omega is None:
            act = act_z
            jacobian = jac_z
            hessian = hess_z
            continue

        omega = input_coords.new_tensor(layer.omega)
        omega_z = omega * act_z
        sin_z = torch.sin(omega_z)
        c1 = omega * torch.cos(omega_z)
        c2 = -(omega * omega) * sin_z
        act = sin_z
        jacobian = c1.unsqueeze(-1) * jac_z
        hessian = c1.unsqueeze(-1).unsqueeze(-1) * hess_z + c2.unsqueeze(-1).unsqueeze(-1) * (
            jac_z.unsqueeze(-1) * jac_z.unsqueeze(-2)
        )

    if spec.output_scale != 1.0:
        scale = input_coords.new_tensor(spec.output_scale)
        act = scale * act
        hessian = scale * hessian
    return act, hessian


def _bending_loss_from_hessian(hessian: torch.Tensor, batch_size: int) -> torch.Tensor:
    # hessian: [N, output_channel, derivative_axis_1, derivative_axis_2]
    dxx = hessian[:, :, 0, 0]
    dyy = hessian[:, :, 1, 1]
    dzz = hessian[:, :, 2, 2]
    dxy = hessian[:, :, 0, 1]
    dxz = hessian[:, :, 0, 2]
    dyz = hessian[:, :, 1, 2]

    loss = torch.mean(torch.square(dxx)) + torch.mean(torch.square(dyy)) + torch.mean(torch.square(dzz))
    loss = loss + 2 * torch.mean(torch.square(dxy)) + 2 * torch.mean(torch.square(dxz)) + torch.mean(torch.square(dyz))
    return loss / batch_size


def compute_bending_energy_for_siren(
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
    input coordinates. ``reference_mode`` is reserved for API clarity; this
    conservative implementation validates only by exact forward comparison when
    ``validate_forward`` is true.
    """

    del reference_mode
    _validate_tensor_contract(input_coords, batch_size=batch_size)
    spec = extract_siren_spec(model)
    analytic_output, hessian = _analytic_forward_hessian(spec, input_coords)
    if analytic_output.ndim != 2 or analytic_output.shape != input_coords.shape:
        raise UnsupportedSirenTopology(
            f"analytic SIREN output must be [N, 3]; got {tuple(analytic_output.shape)}"
        )
    if validate_forward:
        expected = reference_output if reference_output is not None else model(input_coords)
        atol = 1e-6 if input_coords.dtype == torch.float32 else 1e-10
        rtol = 1e-5 if input_coords.dtype == torch.float32 else 1e-10
        if not torch.allclose(analytic_output, expected, atol=atol, rtol=rtol):
            max_abs = torch.max(torch.abs(analytic_output - expected)).detach().cpu().item()
            raise UnsupportedSirenTopology(
                f"analytic forward mismatch for {spec.source}: max_abs={max_abs:.6g}"
            )
    return _bending_loss_from_hessian(hessian, int(batch_size))


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
    """Upstream-compatible strict analytic drop-in.

    This function never silently falls back. The matching SIREN model must have
    been registered by an installer-patched forward call using this exact
    ``input_coords`` tensor.
    """

    _validate_tensor_contract(input_coords, output, batch_size)
    model = _lookup_context(input_coords)
    return compute_bending_energy_for_siren(model, input_coords, int(batch_size))


@dataclass
class RestoreHandle:
    """Small patch restore handle returned by installers."""

    restore_callbacks: list[Callable[[], None]]
    restored: bool = False

    def restore(self) -> None:
        if self.restored:
            return
        for callback in reversed(self.restore_callbacks):
            callback()
        self.restored = True


def install_phase2_dropin(*, strict: bool = True, fallback: Callable[..., torch.Tensor] | None = None) -> RestoreHandle:
    """Patch real IDIR phase-2 SIREN and regularizer functions in this process."""

    del fallback
    if not strict:
        raise ValueError("analytic SIREN drop-in only supports strict=True in this iteration")
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
    """Patch the existing benchmark module to use the strict analytic drop-in."""

    del fallback
    if not strict:
        raise ValueError("analytic SIREN benchmark drop-in only supports strict=True in this iteration")
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
    "compute_bending_energy_for_siren",
    "extract_siren_spec",
    "install_benchmark_dropin",
    "install_phase2_dropin",
]
