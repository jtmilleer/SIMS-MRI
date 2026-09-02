#!/usr/bin/env python3
"""Benchmark a representative synthetic Phase-2 registration training step.

This is a measurement-only harness. It does not patch or edit training code and
it does not implement a new optimization candidate. The step mirrors the hot
Phase-2 path in src/sims_mri/training/phase2_registration.py:

    coords -> registration SIREN -> coordinate transform -> input mapper ->
    frozen generation model -> registration loss -> optional regularizer ->
    backward -> optimizer.step()

The default workload is intentionally compact so it can run on a local CPU or
MPS device. Use --module-flavor production-like and larger model knobs for a
closer but more expensive synthetic proxy.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import torch
from torch import nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from IDIR.networks import networks  # noqa: E402
from IDIR.objectives import regularizers as upstream_regularizers  # noqa: E402
from IDIR.objectives.ncc import NCC  # noqa: E402
from multi_contrast_inr.model import MLPv1  # noqa: E402
from sims_mri.hash_encoder_wrapper_freenerf import HashEncoderWrapper  # noqa: E402


BendingFn = Callable[[torch.Tensor, torch.Tensor, int | None], torch.Tensor]


def _mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("mps" if _mps_available() else "cpu")
    if requested == "mps" and not _mps_available():
        raise SystemExit("Requested --device mps, but torch.backends.mps.is_available() is false")
    return torch.device(requested)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min_ms": None, "median_ms": None, "max_ms": None}
    return {
        "min_ms": min(values),
        "median_ms": statistics.median(values),
        "max_ms": max(values),
    }


class CompactInputMapper(nn.Module):
    """Small trainable coordinate mapper that keeps gradients to coordinates."""

    def __init__(self, hidden_dim: int, out_dim: int, hidden_layers: int = 1):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = 3
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)
        self.output_dim = out_dim

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords)


class CompactGenerationModel(nn.Module):
    """Small trainable intensity model used as the frozen Phase-1 proxy."""

    def __init__(self, input_dim: int, hidden_dim: int, hidden_layers: int):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class BenchModules:
    registration: nn.Module
    input_mapper: nn.Module
    generation: nn.Module
    module_flavor: str
    module_notes: str
    input_dim: int
    reg_param_count: int
    mapper_param_count: int
    generation_param_count: int


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(enabled)


def build_modules(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> BenchModules:
    registration = networks.Siren(
        [3] + [args.reg_hidden_dim] * args.reg_hidden_layers + [3],
        True,
        args.reg_omega,
    )

    if args.module_flavor == "compact":
        input_mapper = CompactInputMapper(
            hidden_dim=args.mapper_hidden_dim,
            out_dim=args.mapper_out_dim,
            hidden_layers=args.mapper_hidden_layers,
        )
        generation = CompactGenerationModel(
            input_dim=input_mapper.output_dim,
            hidden_dim=args.gen_hidden_dim,
            hidden_layers=args.gen_hidden_layers,
        )
        notes = "compact trainable mapper + compact trainable intensity MLP"
        input_dim = input_mapper.output_dim
    elif args.module_flavor == "production-like":
        input_mapper = HashEncoderWrapper(
            n_levels=args.hash_n_levels,
            n_features_per_level=args.hash_n_features_per_level,
            log2_hashmap_size=args.hash_log2_size,
            base_resolution=args.hash_base_resolution,
            per_level_scale=args.hash_per_level_scale,
            concat_coords=args.concat_coords,
            input_range=(-1, 1),
            progressive_unlock=False,
        )
        input_dim = input_mapper.output_dim
        generation = MLPv1(
            input_size=input_dim,
            output_size=1,
            hidden_size=args.gen_hidden_dim,
            num_layers=args.gen_hidden_layers,
            dropout=0,
        )
        notes = "IDIR registration SIREN + HashEncoderWrapper + MLPv1, with configurable reduced sizes"
    else:  # pragma: no cover - argparse choices prevent this
        raise ValueError(f"unknown module flavor {args.module_flavor}")

    registration = registration.to(device=device, dtype=dtype)
    input_mapper = input_mapper.to(device=device, dtype=dtype)
    generation = generation.to(device=device, dtype=dtype)

    # Phase 2 uses eval() for the frozen modules but, as of this measurement
    # pass, does not freeze their parameter grads in training code.
    input_mapper.eval()
    generation.eval()
    registration.train()

    if args.freeze_frozen:
        set_requires_grad(input_mapper, False)
        set_requires_grad(generation, False)
    else:
        set_requires_grad(input_mapper, True)
        set_requires_grad(generation, True)

    return BenchModules(
        registration=registration,
        input_mapper=input_mapper,
        generation=generation,
        module_flavor=args.module_flavor,
        module_notes=notes,
        input_dim=input_dim,
        reg_param_count=count_params(registration),
        mapper_param_count=count_params(input_mapper),
        generation_param_count=count_params(generation),
    )


def synthetic_coords_and_labels(n: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    # Deterministic coordinate cloud in [-1, 1] with a smooth target intensity.
    coords = torch.rand(n, 3, device=device, dtype=dtype) * 2.0 - 1.0
    coords.requires_grad_(True)
    labels = (
        0.45 * torch.sin(3.1 * coords[:, :1])
        + 0.35 * torch.cos(2.3 * coords[:, 1:2])
        + 0.20 * coords[:, 2:3].square()
    ).detach()
    return coords, labels


def resolve_bending_backend(name: str) -> tuple[str, BendingFn | None]:
    if name == "none":
        return "none", None
    if name == "upstream":
        return "IDIR.objectives.regularizers.compute_bending_energy", upstream_regularizers.compute_bending_energy
    if name == "batched_vjp":
        from sims_mri.optimization_candidates.phase2_bending_batched_vjp import compute_bending_energy

        return "sims_mri.optimization_candidates.phase2_bending_batched_vjp.compute_bending_energy", compute_bending_energy
    if name == "analytic_siren":
        from sims_mri.optimization_candidates.phase2_bending_siren_analytic import compute_bending_energy_for_siren

        def bending(coords: torch.Tensor, output_rel: torch.Tensor, batch_size: int | None = None) -> torch.Tensor:
            del output_rel
            if batch_size is None:
                raise ValueError("batch_size is required")
            # _active_registration is set around the measured step. Keeping
            # this wrapper local avoids monkey-patching the production module.
            return compute_bending_energy_for_siren(_active_registration, coords, int(batch_size))

        return "sims_mri.optimization_candidates.phase2_bending_siren_analytic.compute_bending_energy_for_siren", bending
    if name == "terms6":
        from sims_mri.optimization_candidates.phase2_bending_siren_terms6 import compute_bending_energy_for_siren_terms6

        def bending(coords: torch.Tensor, output_rel: torch.Tensor, batch_size: int | None = None) -> torch.Tensor:
            del output_rel
            if batch_size is None:
                raise ValueError("batch_size is required")
            return compute_bending_energy_for_siren_terms6(_active_registration, coords, int(batch_size))

        return "sims_mri.optimization_candidates.phase2_bending_siren_terms6.compute_bending_energy_for_siren_terms6", bending
    raise ValueError(f"unknown backend {name}")


# Process-local pointer used only by analytic backend wrappers above. This is
# deliberately not a production-code patch and is reset for each measured step.
_active_registration: nn.Module


def registration_loss(pred: torch.Tensor, labels: torch.Tensor, loss_name: str, ncc: NCC) -> torch.Tensor:
    if loss_name == "cc":
        return ncc(pred, labels)
    if loss_name == "mse":
        return F.mse_loss(pred.T[None, :], labels.T[None, :])
    if loss_name == "l1":
        return F.l1_loss(pred.T[None, :], labels.T[None, :])
    raise ValueError(f"unsupported synthetic benchmark loss: {loss_name}")


@dataclass
class StepOutputs:
    total_loss: torch.Tensor
    data_loss: torch.Tensor
    regularizer_loss: torch.Tensor | None
    prediction: torch.Tensor


@dataclass
class ProfileAccumulator:
    values: dict[str, list[float]] = field(default_factory=dict)

    def add(self, name: str, elapsed_ms: float) -> None:
        self.values.setdefault(name, []).append(elapsed_ms)

    def summary(self) -> dict[str, dict[str, float | None]]:
        return {name: quantiles(samples) for name, samples in self.values.items()}


def timed_component(
    enabled: bool,
    device: torch.device,
    profile: ProfileAccumulator | None,
    name: str,
):
    if not enabled or profile is None:
        return nullcontext()

    class _Timer:
        def __enter__(self):
            synchronize(device)
            self.start = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc, tb):
            synchronize(device)
            profile.add(name, (time.perf_counter() - self.start) * 1000.0)
            return False

    return _Timer()


def forward_step(
    modules: BenchModules,
    coords: torch.Tensor,
    labels: torch.Tensor,
    *,
    args: argparse.Namespace,
    ncc: NCC,
    bending_fn: BendingFn | None,
    profile: ProfileAccumulator | None = None,
) -> StepOutputs:
    global _active_registration
    _active_registration = modules.registration

    with timed_component(args.profile, coords.device, profile, "registration_forward"):
        displacement = modules.registration(coords)

    with timed_component(args.profile, coords.device, profile, "coordinate_transform"):
        transformed = displacement + coords
        if args.clamp_coords:
            transformed = torch.clamp(transformed, min=-1.0, max=1.0)

    with timed_component(args.profile, coords.device, profile, "input_mapper_forward"):
        mapped = modules.input_mapper(transformed)

    with timed_component(args.profile, coords.device, profile, "generation_forward"):
        prediction = modules.generation(mapped)

    with timed_component(args.profile, coords.device, profile, "loss"):
        data_loss = registration_loss(prediction, labels, args.loss, ncc)
        regularizer_loss = None
        total_loss = data_loss
        if args.regularization != "none" and bending_fn is not None:
            # Mirrors existing phase2 code exactly: output_rel = displacement - coords,
            # even though displacement is already the registration SIREN output.
            output_rel = displacement - coords
            if args.regularization == "bending":
                regularizer_loss = bending_fn(coords, output_rel, args.batch_size_arg or coords.shape[0])
            elif args.regularization == "jacobian":
                regularizer_loss = upstream_regularizers.compute_jacobian_loss(
                    coords, output_rel, batch_size=args.batch_size_arg or coords.shape[0]
                )
            elif args.regularization == "hyper":
                regularizer_loss = upstream_regularizers.compute_hyper_elastic_loss(
                    coords, output_rel, batch_size=args.batch_size_arg or coords.shape[0]
                )
            else:
                raise ValueError(f"unknown regularization {args.regularization}")
            total_loss = total_loss + args.alpha * regularizer_loss
    return StepOutputs(total_loss, data_loss, regularizer_loss, prediction)


def measured_iteration(
    modules: BenchModules,
    optimizer: torch.optim.Optimizer,
    *,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    ncc: NCC,
    bending_fn: BendingFn | None,
    backward: bool,
    profile: ProfileAccumulator | None = None,
) -> StepOutputs:
    optimizer.zero_grad(set_to_none=True)
    if not args.freeze_frozen:
        modules.input_mapper.zero_grad(set_to_none=True)
        modules.generation.zero_grad(set_to_none=True)

    coords, labels = synthetic_coords_and_labels(args.n, device, dtype)
    outputs = forward_step(
        modules,
        coords,
        labels,
        args=args,
        ncc=ncc,
        bending_fn=bending_fn,
        profile=profile,
    )
    if backward:
        with timed_component(args.profile, device, profile, "backward"):
            outputs.total_loss.backward()
        with timed_component(args.profile, device, profile, "optimizer_step"):
            optimizer.step()
    return outputs


def time_samples(
    modules: BenchModules,
    optimizer: torch.optim.Optimizer,
    *,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    ncc: NCC,
    bending_fn: BendingFn | None,
    backward: bool,
) -> tuple[list[float], ProfileAccumulator | None, float, float | None]:
    profile = ProfileAccumulator() if args.profile else None
    last_total_loss = float("nan")
    last_regularizer_loss: float | None = None

    for _ in range(args.warmups):
        outputs = measured_iteration(
            modules,
            optimizer,
            args=args,
            device=device,
            dtype=dtype,
            ncc=ncc,
            bending_fn=bending_fn,
            backward=backward,
            profile=None,
        )
        last_total_loss = outputs.total_loss.detach().cpu().item()
        last_regularizer_loss = (
            None if outputs.regularizer_loss is None else outputs.regularizer_loss.detach().cpu().item()
        )
    synchronize(device)

    timings: list[float] = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        outputs = measured_iteration(
            modules,
            optimizer,
            args=args,
            device=device,
            dtype=dtype,
            ncc=ncc,
            bending_fn=bending_fn,
            backward=backward,
            profile=profile,
        )
        synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        timings.append(elapsed_ms)
        total = outputs.total_loss.detach()
        if not torch.isfinite(total).item():
            raise RuntimeError(f"non-finite total loss: {total.cpu().item()}")
        last_total_loss = total.cpu().item()
        last_regularizer_loss = (
            None if outputs.regularizer_loss is None else outputs.regularizer_loss.detach().cpu().item()
        )
    return timings, profile, last_total_loss, last_regularizer_loss


@dataclass
class TimingResult:
    benchmark: str
    requested_device: str
    device: str
    dtype: str
    n: int
    batch_size_arg: int
    mode: str
    warmups: int
    repeats: int
    module_flavor: str
    module_notes: str
    freeze_frozen: bool
    loss: str
    regularization: str
    alpha: float
    regularizer_backend: str
    regularizer_impl: str
    clamp_coords: bool
    reg_hidden_dim: int
    reg_hidden_layers: int
    mapper_hidden_dim: int
    mapper_hidden_layers: int
    mapper_out_dim: int
    gen_hidden_dim: int
    gen_hidden_layers: int
    hash_n_levels: int
    hash_n_features_per_level: int
    hash_log2_size: int
    concat_coords: bool
    reg_param_count: int
    mapper_param_count: int
    generation_param_count: int
    forward_ms: list[float]
    full_step_ms: list[float]
    forward_median_ms: float | None
    full_step_median_ms: float | None
    last_total_loss: float | None
    last_regularizer_loss: float | None
    profile_ms: dict[str, dict[str, float | None]] | None
    status: str
    error: str | None = None


def benchmark_case(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> TimingResult:
    torch.manual_seed(args.seed)
    modules = build_modules(args, device, dtype)
    optimizer = torch.optim.Adam(modules.registration.parameters(), lr=args.lr)
    ncc = NCC().to(device)
    regularizer_impl, bending_fn = resolve_bending_backend(args.regularizer_backend)

    if args.regularization != "bending" and args.regularizer_backend not in {"none", "upstream"}:
        raise ValueError("non-bending regularization uses upstream regularizer functions; choose --regularizer-backend upstream/none")
    if args.regularization == "none":
        bending_fn = None
        regularizer_impl = "none"
    if args.regularization == "bending" and args.regularizer_backend == "none":
        raise ValueError("--regularization bending requires a real --regularizer-backend")

    forward_ms: list[float] = []
    full_step_ms: list[float] = []
    profile_summary: dict[str, dict[str, float | None]] | None = None
    last_total_loss: float | None = None
    last_regularizer_loss: float | None = None

    try:
        if args.mode in {"forward", "both"}:
            forward_ms, forward_profile, last_total_loss, last_regularizer_loss = time_samples(
                modules,
                optimizer,
                args=args,
                device=device,
                dtype=dtype,
                ncc=ncc,
                bending_fn=bending_fn,
                backward=False,
            )
            if forward_profile is not None:
                profile_summary = {f"forward.{k}": v for k, v in forward_profile.summary().items()}
        if args.mode in {"step", "both"}:
            full_step_ms, step_profile, last_total_loss, last_regularizer_loss = time_samples(
                modules,
                optimizer,
                args=args,
                device=device,
                dtype=dtype,
                ncc=ncc,
                bending_fn=bending_fn,
                backward=True,
            )
            if step_profile is not None:
                profile_summary = profile_summary or {}
                profile_summary.update({f"step.{k}": v for k, v in step_profile.summary().items()})
    except Exception as exc:
        return TimingResult(
            benchmark="phase2_synthetic_step",
            requested_device=args.device,
            device=str(device),
            dtype=str(dtype).replace("torch.", ""),
            n=args.n,
            batch_size_arg=args.batch_size_arg or args.n,
            mode=args.mode,
            warmups=args.warmups,
            repeats=args.repeats,
            module_flavor=modules.module_flavor,
            module_notes=modules.module_notes,
            freeze_frozen=args.freeze_frozen,
            loss=args.loss,
            regularization=args.regularization,
            alpha=args.alpha,
            regularizer_backend=args.regularizer_backend,
            regularizer_impl=regularizer_impl,
            clamp_coords=args.clamp_coords,
            reg_hidden_dim=args.reg_hidden_dim,
            reg_hidden_layers=args.reg_hidden_layers,
            mapper_hidden_dim=args.mapper_hidden_dim,
            mapper_hidden_layers=args.mapper_hidden_layers,
            mapper_out_dim=args.mapper_out_dim,
            gen_hidden_dim=args.gen_hidden_dim,
            gen_hidden_layers=args.gen_hidden_layers,
            hash_n_levels=args.hash_n_levels,
            hash_n_features_per_level=args.hash_n_features_per_level,
            hash_log2_size=args.hash_log2_size,
            concat_coords=args.concat_coords,
            reg_param_count=modules.reg_param_count,
            mapper_param_count=modules.mapper_param_count,
            generation_param_count=modules.generation_param_count,
            forward_ms=forward_ms,
            full_step_ms=full_step_ms,
            forward_median_ms=median_or_none(forward_ms),
            full_step_median_ms=median_or_none(full_step_ms),
            last_total_loss=last_total_loss,
            last_regularizer_loss=last_regularizer_loss,
            profile_ms=profile_summary,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
        )

    return TimingResult(
        benchmark="phase2_synthetic_step",
        requested_device=args.device,
        device=str(device),
        dtype=str(dtype).replace("torch.", ""),
        n=args.n,
        batch_size_arg=args.batch_size_arg or args.n,
        mode=args.mode,
        warmups=args.warmups,
        repeats=args.repeats,
        module_flavor=modules.module_flavor,
        module_notes=modules.module_notes,
        freeze_frozen=args.freeze_frozen,
        loss=args.loss,
        regularization=args.regularization,
        alpha=args.alpha,
        regularizer_backend=args.regularizer_backend,
        regularizer_impl=regularizer_impl,
        clamp_coords=args.clamp_coords,
        reg_hidden_dim=args.reg_hidden_dim,
        reg_hidden_layers=args.reg_hidden_layers,
        mapper_hidden_dim=args.mapper_hidden_dim,
        mapper_hidden_layers=args.mapper_hidden_layers,
        mapper_out_dim=args.mapper_out_dim,
        gen_hidden_dim=args.gen_hidden_dim,
        gen_hidden_layers=args.gen_hidden_layers,
        hash_n_levels=args.hash_n_levels,
        hash_n_features_per_level=args.hash_n_features_per_level,
        hash_log2_size=args.hash_log2_size,
        concat_coords=args.concat_coords,
        reg_param_count=modules.reg_param_count,
        mapper_param_count=modules.mapper_param_count,
        generation_param_count=modules.generation_param_count,
        forward_ms=forward_ms,
        full_step_ms=full_step_ms,
        forward_median_ms=median_or_none(forward_ms),
        full_step_median_ms=median_or_none(full_step_ms),
        last_total_loss=last_total_loss,
        last_regularizer_loss=last_regularizer_loss,
        profile_ms=profile_summary,
        status="ok",
    )


def emit_result(result: TimingResult) -> None:
    print("RESULT_JSON " + json.dumps(asdict(result), sort_keys=True))
    fwd = "n/a" if result.forward_median_ms is None else f"{result.forward_median_ms:.3f} ms"
    step = "n/a" if result.full_step_median_ms is None else f"{result.full_step_median_ms:.3f} ms"
    reg = "n/a" if result.last_regularizer_loss is None else f"{result.last_regularizer_loss:.6g}"
    print(
        "RESULT_TABLE "
        f"device={result.device} flavor={result.module_flavor} freeze_frozen={result.freeze_frozen} "
        f"n={result.n} loss={result.loss} regularization={result.regularization} "
        f"backend={result.regularizer_backend} forward_median={fwd} full_step_median={step} "
        f"last_loss={result.last_total_loss:.6g} last_regularizer={reg} status={result.status}"
    )
    if result.profile_ms:
        print("PROFILE_JSON " + json.dumps(result.profile_ms, sort_keys=True))
    if result.error:
        print(f"RESULT_ERROR {result.error}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--n", type=int, default=1024, help="Synthetic coordinate count for one phase-2 batch")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mode", choices=["forward", "step", "both"], default="both")
    parser.add_argument("--loss", choices=["cc", "mse", "l1"], default="mse")
    parser.add_argument("--regularization", choices=["bending", "jacobian", "hyper", "none"], default="bending")
    parser.add_argument(
        "--regularizer-backend",
        choices=["upstream", "batched_vjp", "analytic_siren", "terms6", "none"],
        default="terms6",
    )
    parser.add_argument("--freeze-frozen", action="store_true", help="Set input_mapper/generation params requires_grad=False")
    parser.add_argument("--module-flavor", choices=["compact", "production-like"], default="compact")
    parser.add_argument("--profile", action="store_true", help="Emit synchronized component timing medians")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--alpha", type=float, default=1000.0)
    parser.add_argument("--batch-size-arg", type=int, default=None, help="batch_size value passed to regularizers; default N")
    parser.add_argument("--clamp-coords", action="store_true")

    # Registration SIREN knobs. Defaults are reduced from production [3,256,256,256,3]
    # so local baseline runs finish quickly; pass --reg-hidden-dim 256 --reg-hidden-layers 3
    # to mirror the production registration model.
    parser.add_argument("--reg-hidden-dim", type=int, default=64)
    parser.add_argument("--reg-hidden-layers", type=int, default=2)
    parser.add_argument("--reg-omega", type=float, default=32.0)

    # Compact mapper/generator knobs.
    parser.add_argument("--mapper-hidden-dim", type=int, default=64)
    parser.add_argument("--mapper-hidden-layers", type=int, default=1)
    parser.add_argument("--mapper-out-dim", type=int, default=32)
    parser.add_argument("--gen-hidden-dim", type=int, default=64)
    parser.add_argument("--gen-hidden-layers", type=int, default=2)

    # Production-like HashEncoderWrapper knobs, defaulting to a smaller local table
    # than the OASIS config. Increase to config values for heavier measurements.
    parser.add_argument("--hash-n-levels", type=int, default=4)
    parser.add_argument("--hash-n-features-per-level", type=int, default=2)
    parser.add_argument("--hash-log2-size", type=int, default=12)
    parser.add_argument("--hash-base-resolution", type=int, default=16)
    parser.add_argument("--hash-per-level-scale", type=float, default=1.39)
    parser.add_argument("--concat-coords", action=argparse.BooleanOptionalAction, default=True)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.n <= 1:
        parser.error("--n must be > 1")
    if args.warmups < 0 or args.repeats <= 0:
        parser.error("--warmups must be >= 0 and --repeats must be > 0")
    if args.reg_hidden_dim <= 0 or args.reg_hidden_layers <= 0:
        parser.error("registration model sizes must be positive")
    if args.mapper_hidden_dim <= 0 or args.mapper_hidden_layers < 0 or args.mapper_out_dim <= 0:
        parser.error("mapper sizes must be positive, with --mapper-hidden-layers >= 0")
    if args.gen_hidden_dim <= 0 or args.gen_hidden_layers < 0:
        parser.error("generation sizes must be positive, with --gen-hidden-layers >= 0")
    if args.batch_size_arg is not None and args.batch_size_arg <= 0:
        parser.error("--batch-size-arg must be positive")
    if args.regularization == "none" and args.regularizer_backend != "none":
        # Normalize rather than fail; this makes no-regularizer comparison less fussy.
        args.regularizer_backend = "none"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    device = resolve_device(args.device)
    if args.batch_size_arg is None:
        args.batch_size_arg = args.n

    print(
        "BENCH_ENV "
        + json.dumps(
            {
                "benchmark": "bench_phase2_step.py",
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "mps_available": _mps_available(),
                "repo_root": str(REPO_ROOT),
            },
            sort_keys=True,
        )
    )
    print("BENCH_ARGS " + json.dumps(vars(args), sort_keys=True))

    result = benchmark_case(args, device, dtype)
    emit_result(result)
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
