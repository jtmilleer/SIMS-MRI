import unittest

import torch
from torch import nn

from sims_mri.optimization_candidates.phase2_bending_batched_vjp import (
    compute_bending_energy as compute_batched_vjp_bending,
)
from sims_mri.optimization_candidates.phase2_bending_siren_analytic import (
    compute_bending_energy_for_siren,
)
from sims_mri.optimization_candidates.phase2_bending_siren_terms6 import (
    compute_bending_energy_for_siren_terms6,
)


class SirenLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, *, omega: float):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.omega = omega

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * self.linear(x))


class TinySirenDisplacement(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            SirenLayer(3, 5, omega=2.0),
            SirenLayer(5, 4, omega=1.5),
            nn.Linear(4, 3),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return 0.05 * self.net(coords)


class Phase2BendingCandidateTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1234)
        self.model = TinySirenDisplacement().to(dtype=torch.float64)
        self.coords = torch.randn(9, 3, dtype=torch.float64)

    def _loss_and_coord_grad(self, backend: str) -> tuple[torch.Tensor, torch.Tensor]:
        coords = self.coords.detach().clone().requires_grad_(True)
        output = self.model(coords)
        batch_size = coords.shape[0]

        if backend == "batched_vjp":
            loss = compute_batched_vjp_bending(coords, output, batch_size=batch_size)
        elif backend == "analytic":
            loss = compute_bending_energy_for_siren(self.model, coords, batch_size=batch_size)
        elif backend == "terms6":
            loss = compute_bending_energy_for_siren_terms6(self.model, coords, batch_size=batch_size)
        else:  # pragma: no cover
            raise AssertionError(f"unknown backend {backend}")

        grad = torch.autograd.grad(loss, coords)[0]
        return loss.detach(), grad.detach()

    def test_analytic_backends_match_batched_vjp_scalar_and_coordinate_gradients(self) -> None:
        baseline_loss, baseline_grad = self._loss_and_coord_grad("batched_vjp")

        for backend in ("analytic", "terms6"):
            with self.subTest(backend=backend):
                loss, grad = self._loss_and_coord_grad(backend)
                torch.testing.assert_close(loss, baseline_loss, rtol=1e-9, atol=1e-11)
                torch.testing.assert_close(grad, baseline_grad, rtol=1e-8, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
