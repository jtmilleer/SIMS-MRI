"""Exact batched-VJP drop-in for phase-2 bending energy.

This module intentionally does not patch the upstream regularizer at import time.
Call ``install_dropin()`` from a benchmark/training launcher to replace
``IDIR.objectives.regularizers.compute_bending_energy`` in that process.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def _validate_inputs(input_coords: torch.Tensor, output: torch.Tensor, batch_size: int | None) -> None:
    """Validate the hard-coded 3D coordinate displacement contract."""

    if input_coords.ndim != 2 or output.ndim != 2:
        raise ValueError(
            "compute_bending_energy expects rank-2 tensors shaped [N, 3]; "
            f"got input_coords.shape={tuple(input_coords.shape)} and output.shape={tuple(output.shape)}"
        )
    if input_coords.shape[1] != 3 or output.shape[1] != 3:
        raise ValueError(
            "compute_bending_energy expects 3D coordinates/displacements shaped [N, 3]; "
            f"got input_coords.shape={tuple(input_coords.shape)} and output.shape={tuple(output.shape)}"
        )
    if input_coords.shape[0] != output.shape[0]:
        raise ValueError(
            "input_coords and output must have the same batch dimension; "
            f"got input_coords.shape={tuple(input_coords.shape)} and output.shape={tuple(output.shape)}"
        )
    if batch_size is None:
        raise ValueError("batch_size must be provided for bending-energy scaling")


def _batched_grad(
    outputs: torch.Tensor,
    inputs: torch.Tensor,
    grad_outputs: torch.Tensor,
) -> torch.Tensor:
    """Return a batched VJP with graph construction preserved for backprop."""

    return torch.autograd.grad(
        outputs,
        [inputs],
        grad_outputs=grad_outputs,
        create_graph=True,
        is_grads_batched=True,
    )[0]


def compute_bending_energy(
    input_coords: torch.Tensor,
    output: torch.Tensor,
    batch_size: int | None = None,
) -> torch.Tensor:
    """Compute upstream-equivalent bending energy using two batched VJPs.

    This preserves the current upstream semantics exactly for valid phase-2 calls:
    * ``input_coords`` and ``output`` are shaped ``[N, 3]``.
    * first derivatives are gradients of output-channel sums with respect to
      ``input_coords``;
    * second derivatives use the same diagonal/mixed terms as upstream;
    * the ``dyz`` mixed term keeps the upstream coefficient of ``1``;
    * the result is divided by the caller-provided ``batch_size``.
    """

    _validate_inputs(input_coords, output, batch_size)
    n_points = input_coords.shape[0]

    # First derivatives for all three output channels in one autograd call.
    # Shape: [channel_batch=3, N, output_dim=3]. Each batched grad_output selects
    # one output channel for all points, matching upstream gradient(output[:, i]).
    channel_basis = torch.eye(3, device=output.device, dtype=output.dtype)[:, None, :]
    channel_basis = channel_basis.expand(-1, n_points, -1)
    jac_by_channel = _batched_grad(output, input_coords, channel_basis)
    # jac_by_channel[i, n, j] = d output[n, i] / d input_coords[n, j]

    # Second derivatives for all nine (output channel, first-derivative axis)
    # sources in one autograd call. Shape: [hessian_batch=9, 3, N, 3].
    source_basis = torch.eye(9, device=jac_by_channel.device, dtype=jac_by_channel.dtype)
    source_basis = source_basis.reshape(9, 3, 1, 3).expand(-1, -1, n_points, -1)
    hessian_vjps = _batched_grad(jac_by_channel, input_coords, source_basis)
    hessian = hessian_vjps.reshape(3, 3, n_points, 3)
    # hessian[i, j, n, k] = d^2 output[n, i] / d input_coords[n, j] d input_coords[n, k]

    dxx = hessian[:, 0, :, 0]
    dyy = hessian[:, 1, :, 1]
    dzz = hessian[:, 2, :, 2]
    dxy = hessian[:, 0, :, 1]
    dxz = hessian[:, 0, :, 2]
    dyz = hessian[:, 1, :, 2]

    loss = (
        torch.mean(torch.square(dxx))
        + torch.mean(torch.square(dyy))
        + torch.mean(torch.square(dzz))
    )
    loss = loss + (
        2 * torch.mean(torch.square(dxy))
        + 2 * torch.mean(torch.square(dxz))
        + torch.mean(torch.square(dyz))
    )

    return loss / batch_size


def install_dropin() -> Callable[..., torch.Tensor]:
    """Install this function as the process-local upstream bending-energy drop-in.

    Returns the previous ``IDIR.objectives.regularizers.compute_bending_energy``
    callable so callers/tests can restore it if desired.
    """

    from IDIR.objectives import regularizers

    previous = regularizers.compute_bending_energy
    regularizers.compute_bending_energy = compute_bending_energy
    return previous


__all__ = ["compute_bending_energy", "install_dropin"]
