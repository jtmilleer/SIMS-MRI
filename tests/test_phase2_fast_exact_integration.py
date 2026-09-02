"""Small CPU contracts against real upstream models; no data or training.

Run after ``bash scripts/setup_upstreams.sh`` and installing project dependencies:
``PYTHONPATH=src:. python -m unittest discover -s tests -p 'test_phase2*.py' -v``.
Only absent upstream repositories are skipped; broken upstream imports fail.
"""

import contextlib
import importlib.util
import io
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from sims_mri.optimization_candidates.phase2_bending_siren_analytic import (
    compute_bending_energy_for_siren,
)
from sims_mri.optimization_candidates.phase2_bending_siren_terms6 import (
    compute_bending_energy_for_siren_terms6,
)


class Phase2FastExactIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        missing = [
            name for name in ("IDIR", "multi_contrast_inr")
            if importlib.util.find_spec(name) is None
        ]
        if missing:
            raise unittest.SkipTest(
                f"Missing upstream repositories: {', '.join(missing)}. "
                "Run bash scripts/setup_upstreams.sh and use PYTHONPATH=src:."
            )

        from IDIR.networks.networks import Siren
        from IDIR.objectives import regularizers
        from multi_contrast_inr.model import MLPv1
        from sims_mri.hash_encoder_wrapper_freenerf import HashEncoderWrapper
        from sims_mri.training import phase2_registration_fast_exact

        cls.Siren = Siren
        cls.MLPv1 = MLPv1
        cls.HashEncoderWrapper = HashEncoderWrapper
        cls.regularizers = regularizers
        cls.wrapper = phase2_registration_fast_exact

    def setUp(self) -> None:
        self.addCleanup(torch.set_default_dtype, torch.get_default_dtype())
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        self.originals = [(obj, name, getattr(obj, name)) for obj, name in self._patch_targets()]
        for obj, name, original in self.originals:
            self.addCleanup(setattr, obj, name, original)

    def _fixture(self, dtype):
        torch.set_default_dtype(dtype)
        torch.manual_seed(12)
        model = self.Siren([3, 8, 8, 3], omega=3)
        coords = torch.randn(17, 3, requires_grad=True)
        return model, coords

    def _assert_gradients_close(self, actual, expected, inputs, *, dtype):
        # Final affine bias has no bending derivative. None and an explicit
        # zero represent the same derivative, but nonzero omissions still fail.
        rtol, atol = (1e-5, 1e-7) if dtype == torch.float32 else (1e-8, 1e-10)
        for index, (got, want, tensor) in enumerate(zip(actual, expected, inputs)):
            with self.subTest(gradient=index):
                got = torch.zeros_like(tensor) if got is None else got
                want = torch.zeros_like(tensor) if want is None else want
                torch.testing.assert_close(got, want, rtol=rtol, atol=atol)

    def _patch_targets(self):
        return (
            (self.regularizers, "compute_bending_energy"),
            (self.Siren, "forward"),
            (self.Siren, "eval"),
            (self.MLPv1, "eval"),
            (self.HashEncoderWrapper, "eval"),
        )

    def test_real_idir_bending_scalar_parameter_and_coordinate_gradients(self) -> None:
        for dtype in (torch.float32, torch.float64):
            model, coords = self._fixture(dtype)
            inputs = tuple(model.parameters()) + (coords,)
            upstream = self.regularizers.compute_bending_energy(
                coords, model(coords), batch_size=len(coords)
            )
            expected = torch.autograd.grad(upstream, inputs, allow_unused=True)
            self.assertGreater(torch.count_nonzero(expected[0]).item(), 0)
            self.assertGreater(torch.count_nonzero(expected[-1]).item(), 0)
            for backend in (
                compute_bending_energy_for_siren_terms6,
                compute_bending_energy_for_siren,
            ):
                with self.subTest(dtype=dtype, backend=backend.__name__):
                    loss = backend(model, coords, batch_size=len(coords))
                    actual = torch.autograd.grad(loss, inputs, allow_unused=True)
                    rtol, atol = (1e-5, 1e-7) if dtype == torch.float32 else (1e-8, 1e-10)
                    torch.testing.assert_close(loss, upstream, rtol=rtol, atol=atol)
                    self._assert_gradients_close(actual, expected, inputs, dtype=dtype)

    def test_installed_wrapper_preserves_image_and_registration_gradients(self) -> None:
        for dtype in (torch.float32, torch.float64):
            for backend in ("terms6", "analytic_siren"):
                with self.subTest(dtype=dtype, backend=backend):
                    model, coords = self._fixture(dtype)
                    image_model = self.MLPv1(
                        input_size=3, hidden_size=8, output_size=1, num_layers=1
                    )
                    inputs = tuple(model.parameters()) + (coords,)
                    displacement = model(coords)
                    baseline_image = image_model(coords + displacement).square().mean()
                    baseline_bending = self.regularizers.compute_bending_energy(
                        coords, displacement, batch_size=len(coords)
                    )
                    expected_image = torch.autograd.grad(
                        baseline_image, inputs, retain_graph=True
                    )
                    expected_total = torch.autograd.grad(
                        baseline_image + baseline_bending, inputs
                    )
                    with contextlib.redirect_stdout(io.StringIO()):
                        handle = self.wrapper.install_phase2_fast_exact_dropins(backend)
                        self.addCleanup(handle.restore)
                        try:
                            # A prior generation step may have left stale grads.
                            for param in image_model.parameters():
                                param.grad = torch.ones_like(param)
                            image_model.eval()
                            model.eval()  # Three-output registration SIRENs stay trainable.
                            self.assertTrue(all(not p.requires_grad for p in image_model.parameters()))
                            self.assertTrue(all(p.grad is None for p in image_model.parameters()))
                            self.assertTrue(all(p.requires_grad for p in model.parameters()))
                            displacement = model(coords)  # Registers the real forward context.
                            image_loss = image_model(coords + displacement).square().mean()
                            bending = self.regularizers.compute_bending_energy(
                                coords, displacement, batch_size=len(coords)
                            )
                            image_grads = torch.autograd.grad(image_loss, inputs, retain_graph=True)
                            # Image-only gradients prevent bending from hiding a detached image path.
                            self.assertGreater(torch.count_nonzero(image_grads[0]).item(), 0)
                            self._assert_gradients_close(image_grads, expected_image, inputs, dtype=dtype)
                            (image_loss + bending).backward()
                            self._assert_gradients_close(
                                tuple(t.grad for t in inputs), expected_total, inputs, dtype=dtype
                            )
                            self.assertTrue(all(p.grad is None for p in image_model.parameters()))
                        finally:
                            handle.restore()
                    handle.restore()  # Restoration remains safe when repeated.
                    for obj, name, original in self.originals:
                        self.assertIs(getattr(obj, name), original)

    def test_cli_restores_all_patches_after_success_and_delegated_exception(self) -> None:
        from sims_mri import training

        for backend in ("terms6", "analytic_siren"):
            for raises in (False, True):
                with self.subTest(backend=backend, raises=raises):
                    error = RuntimeError("delegated CLI failed")

                    def delegated_cli():
                        for obj, name, original in self.originals:
                            self.assertIsNot(getattr(obj, name), original)
                        if raises:
                            raise error

                    # Replace only the expensive training boundary. The wrapper
                    # still installs and restores all real upstream patches.
                    delegated = mock.Mock(side_effect=delegated_cli)
                    with (
                        mock.patch.dict("os.environ", {"SIMS_MRI_PHASE2_BENDING_BACKEND": backend}),
                        mock.patch.object(
                            training, "phase2_registration",
                            SimpleNamespace(_cli=delegated), create=True,
                        ),
                        contextlib.redirect_stdout(io.StringIO()),
                    ):
                        if raises:
                            with self.assertRaises(RuntimeError) as caught:
                                self.wrapper._cli()
                            self.assertIs(caught.exception, error)
                        else:
                            self.wrapper._cli()
                        delegated.assert_called_once_with()
                    for obj, name, original in self.originals:
                        self.assertIs(getattr(obj, name), original)


if __name__ == "__main__":
    unittest.main()
