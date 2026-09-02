"""Resume must preserve the experiment's configuration and phase lineage."""
import copy
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load_runner(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


full = load_runner("full_fast_exact_runner", "run_batch_training_fast_exact.py")
phase2 = load_runner("phase2_fast_exact_runner", "run_batch_training_oasis_phase2_fast_exact.py")


class FastExactResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.old_cwd = Path.cwd()
        os.chdir(self.tmp.name)
        self.addCleanup(os.chdir, self.old_cwd)
        self.config = yaml.safe_load((ROOT / "configs/oasis_ngp_rigid.yaml").read_text())
        self.subject = "synthetic_subject"
        self.config["DATASET"]["SUBJECT_ID"] = self.subject
        self.config_path = "config.yaml"
        Path(self.config_path).write_text(yaml.safe_dump(self.config))
        output = contextlib.redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)
        self.audit = {
            "bending_backend": "terms6", "requested_backend": "auto",
            "strict": True, "selected_device_kind": "cuda",
            "bending_installer": "sims_mri.optimization_candidates.phase2_bending_siren_terms6.install_phase2_dropin",
            "freeze_installer": "sims_mri.training.phase2_registration_freeze_frozen.install_phase2_eval_freeze_dropin",
        }

    def phase1_checkpoint(self, run_id, mtime, config=None):
        path = Path(f"runs/{run_id}/weights/{self.subject}_mlp_mse_e49_model.pt")
        path.parent.mkdir(parents=True)
        path.write_bytes(b"model")
        path.with_name(path.name.replace("_model.pt", "_hash.pt")).write_bytes(b"hash")
        os.utime(path, (mtime, mtime))
        (path.parent.parent / "config_snapshot.json").write_text(json.dumps({
            "unique_id": run_id, "args": {"subject_id": self.subject},
            "config": config or self.config,
        }))
        return str(path)

    def phase2_checkpoint(self, parent, *, audit=True, config=None, status="success", standalone=False):
        path = Path(f"runs/registration/weights/{self.subject}_reg-siren-bend_cc_e49_model_rotation.pt")
        path.parent.mkdir(parents=True)
        path.write_bytes(b"rotation")
        snapshot = {
            "unique_id": "registration", "parent_phase1_id": Path(parent).parent.parent.name,
            "args": {"phase1_checkpoint": parent}, "config": config or self.config,
        }
        (path.parent.parent / "config_snapshot.json").write_text(json.dumps(snapshot))
        command = ["python", "-m", "sims_mri.training.phase2_registration_fast_exact",
                   "--config", self.config_path, "--subject_id", self.subject,
                   "--phase1_checkpoint", parent, "--logging"]
        record = {"phase": "phase2_fast_exact", "status": status,
                  "checkpoint_path": str(path), "command": command,
                  "fast_exact_audit": self.audit if audit else None, "log_file": None}
        if standalone:
            record["phase1_checkpoint"] = parent
            record["phase2_checkpoint"] = record.pop("checkpoint_path")
            subjects = {self.subject: record}
        else:
            subjects = {self.subject: {"phases": [record]}}
        result = Path("runs/batch_phase2_fast_exact_results_test.json" if standalone
                      else "runs/batch_fast_exact_results_test.json")
        result.write_text(json.dumps({"subjects": subjects}))
        return str(path)

    def run_full(self):
        return full.train_subject(self.subject, self.config_path, False, True, True,
                                  "unused_logs", "auto")

    def run_phase2(self):
        return phase2.train_subject_phase2_fast_exact(
            self.subject, self.config_path, False, True, True, "unused_logs", "auto", "runs")

    def test_resume_recomputes_registration_for_newer_phase1(self):
        # Given a completed registration from A and a newer matching Phase 1 B.
        old = self.phase1_checkpoint("phase1_A", 100)
        self.phase2_checkpoint(old)
        newer = self.phase1_checkpoint("phase1_B", 200)
        # When resuming, Phase 2 must run using B instead of reusing A's registration.
        results = self.run_full()
        self.assertEqual(results[0].checkpoint_path, newer)
        self.assertEqual(results[1].status, "dry-run")
        self.assertIn(newer, results[1].command)
        self.assertEqual(self.run_phase2().status, "dry-run")

    def test_matching_completed_record_resumes_with_its_audit(self):
        parent = self.phase1_checkpoint("phase1_A", 100)
        registration = self.phase2_checkpoint(parent)
        results = self.run_full()
        self.assertEqual([r.status for r in results], ["skipped", "skipped", "dry-run"])
        self.assertEqual(results[1].fast_exact_audit, self.audit)
        self.assertIn(registration, results[2].command)
        self.assertEqual(self.run_phase2().fast_exact_audit, self.audit)

    def test_legacy_registration_without_audit_is_recomputed(self):
        parent = self.phase1_checkpoint("phase1_A", 100)
        self.phase2_checkpoint(parent, audit=False)
        self.assertEqual(self.run_full()[1].status, "dry-run")
        self.assertEqual(self.run_phase2().status, "dry-run")

    def test_changed_registration_config_is_not_reused(self):
        parent = self.phase1_checkpoint("phase1_A", 100)
        changed = copy.deepcopy(self.config)
        changed["TRAINING"]["USE_CC"] = False
        self.phase2_checkpoint(parent, config=changed)
        self.assertEqual(self.run_full()[1].status, "dry-run")
        self.assertEqual(self.run_phase2().status, "dry-run")

    def test_newer_phase1_with_wrong_config_is_ignored(self):
        matching = self.phase1_checkpoint("phase1_A", 100)
        changed = copy.deepcopy(self.config)
        changed["TRAINING"]["SEED"] = 99
        self.phase1_checkpoint("phase1_B", 200, config=changed)
        self.assertEqual(self.run_full()[0].checkpoint_path, matching)
        self.assertEqual(self.run_phase2().phase1_checkpoint, matching)

    def test_missing_hash_does_not_resume_incomplete_phase1(self):
        parent = self.phase1_checkpoint("phase1_A", 100)
        Path(parent.replace("_model.pt", "_hash.pt")).unlink()
        self.assertEqual(self.run_full()[0].status, "dry-run")
        self.assertEqual(self.run_phase2().status, "failed")

    def test_phase2_only_record_can_resume_in_either_runner(self):
        parent = self.phase1_checkpoint("phase1_A", 100)
        self.phase2_checkpoint(parent, standalone=True)
        self.assertEqual(self.run_full()[1].fast_exact_audit, self.audit)
        self.assertEqual(self.run_phase2().fast_exact_audit, self.audit)

    def test_repeated_resume_preserves_evidence_when_result_file_is_replaced(self):
        parent = self.phase1_checkpoint("phase1_A", 100)
        self.phase2_checkpoint(parent)
        results = self.run_full()
        full.write_results_json(
            "runs/batch_fast_exact_results_test.json", {self.subject: results},
            self.config_path, 0.0, 1, "auto")
        self.assertEqual(self.run_full()[1].status, "skipped")
        resumed = self.run_phase2()
        self.assertEqual(resumed.status, "skipped")
        phase2.write_results_json(
            "runs/batch_fast_exact_results_test.json", {self.subject: resumed},
            self.config_path, "runs", 0.0, 1)
        self.assertEqual(self.run_full()[1].status, "skipped")

    def test_old_skipped_record_without_provenance_is_not_resumed(self):
        parent = self.phase1_checkpoint("phase1_A", 100)
        self.phase2_checkpoint(parent, status="skipped", audit=False)
        self.assertEqual(self.run_full()[1].status, "dry-run")

    def test_failed_registration_record_is_not_resumed(self):
        parent = self.phase1_checkpoint("phase1_A", 100)
        self.phase2_checkpoint(parent, status="failed")
        self.assertEqual(self.run_full()[1].status, "dry-run")
        self.assertEqual(self.run_phase2().status, "dry-run")

    def test_missing_registration_checkpoint_is_recomputed(self):
        parent = self.phase1_checkpoint("phase1_A", 100)
        Path(self.phase2_checkpoint(parent)).unlink()
        self.assertEqual(self.run_full()[1].status, "dry-run")

    def test_explicit_backend_request_rejects_different_saved_backend(self):
        parent = self.phase1_checkpoint("phase1_A", 100)
        self.phase2_checkpoint(parent)
        results = full.train_subject(self.subject, self.config_path, False, True, True,
                                     "unused_logs", "analytic_siren")
        self.assertEqual(results[1].status, "dry-run")

    def test_malformed_records_do_not_hide_a_valid_completed_run(self):
        parent = self.phase1_checkpoint("phase1_A", 100)
        self.phase2_checkpoint(parent)
        Path("runs/broken.json").write_text("{")
        Path("runs/other.json").write_text(json.dumps({"subjects": [None]}))
        self.assertEqual(self.run_full()[1].status, "skipped")

    def test_generation_cli_override_not_in_config_is_not_reused(self):
        parent = self.phase1_checkpoint("phase1_A", 100)
        snapshot_file = Path(parent).parent.parent / "config_snapshot.json"
        snapshot = json.loads(snapshot_file.read_text())
        snapshot["args"]["hash_n_levels"] = 8
        snapshot_file.write_text(json.dumps(snapshot))
        self.assertEqual(self.run_full()[0].status, "dry-run")
        self.assertEqual(self.run_phase2().status, "failed")

    def test_fresh_run_rejects_non_exact_audit_before_phase3(self):
        def phase_boundary(cmd, phase, subject, log_dir, dry_run, **kwargs):
            if phase == "phase1":
                result = full.PhaseResult(phase=phase, subject_id=subject, status="success")
                return result, "--phase1_checkpoint runs/new/weights/test_model.pt"
            result = full.PhaseResult(phase=phase, subject_id=subject, status="success")
            return result, 'PHASE2_FAST_EXACT {"strict": false}'

        # Stub only subprocess execution; the real orchestrator must stop.
        with patch.object(full, "run_phase", side_effect=phase_boundary) as boundary:
            results = full.train_subject(self.subject, self.config_path, False, False, False,
                                         "unused_logs", "auto")
        self.assertEqual(results[-1].phase, "phase2_fast_exact")
        self.assertEqual(results[-1].status, "failed")
        self.assertEqual(boundary.call_count, 2)

    def test_phase2_only_fresh_run_rejects_non_exact_audit(self):
        import subprocess
        cmd = ["python", "-m", "sims_mri.training.phase2_registration_fast_exact"]
        completed = subprocess.CompletedProcess(cmd, 0, 'PHASE2_FAST_EXACT false', '')
        with patch.object(phase2.subprocess, "run", return_value=completed):
            result = phase2.run_phase2_fast_exact(cmd, self.subject, 'parent.pt', 'logs', 'auto')
        self.assertEqual(result.status, "failed")

    def test_custom_root_parent_without_optional_id_can_resume(self):
        parent = self.phase1_checkpoint("phase1_A", 100)
        Path("runs").rename("archive")
        parent = parent.replace("runs/", "archive/", 1)
        registration = self.phase2_checkpoint(parent, standalone=True)
        snapshot_file = Path(registration).parent.parent / "config_snapshot.json"
        snapshot = json.loads(snapshot_file.read_text())
        # Upstream extract_parent_id_from_path returns None outside runs/<id>/.
        snapshot["parent_phase1_id"] = None
        snapshot_file.write_text(json.dumps(snapshot))
        Path("runs/batch_phase2_fast_exact_results_test.json").rename(
            "archive/batch_phase2_fast_exact_results_test.json")
        result = phase2.train_subject_phase2_fast_exact(
            self.subject, self.config_path, False, True, True, "unused_logs", "auto", "archive")
        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.phase1_checkpoint, parent)
        self.assertEqual(result.fast_exact_audit, self.audit)

    def test_phase2_cli_saves_default_results_in_checkpoint_search_root(self):
        Path("archive").mkdir()
        Path("configs").mkdir()
        Path("src/sims_mri").mkdir(parents=True)
        result = phase2.Phase2FastExactResult(subject_id=self.subject, status="success")
        argv = ["runner", "--config", self.config_path, "--subjects", self.subject,
                "--checkpoint_search_root", "archive"]
        # Only substitute GPU training; exercise the real CLI/result writer.
        with (patch.object(sys, "argv", argv),
              patch.object(phase2, "train_subject_phase2_fast_exact", return_value=result)):
            phase2.main()
        self.assertEqual(len(list(Path("archive").glob("batch_phase2_fast_exact_results_*.json"))), 1)


if __name__ == "__main__":
    unittest.main()
