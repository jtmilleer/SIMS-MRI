"""Read existing run evidence before reusing fast-exact checkpoints.

No tensors are deserialized. A filename or timestamp alone is never evidence
that a registration belongs to the requested generation model/configuration.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any


@dataclass(frozen=True)
class RegistrationResume:
    checkpoint_path: str
    audit: dict[str, Any]
    result_file: str
    log_file: str | None
    command: list[str]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _exists(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _same_path(value: Any, expected: str) -> bool:
    return isinstance(value, str) and Path(value).resolve() == Path(expected).resolve()


def _requested_config(config_path: str, subject_id: str) -> dict[str, Any]:
    import yaml

    with open(config_path) as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not isinstance(config.get("DATASET"), dict):
        raise ValueError("Checkpoint validation requires a YAML config with a DATASET mapping")
    config["DATASET"]["SUBJECT_ID"] = subject_id
    return config


def _snapshot(checkpoint: str) -> dict[str, Any]:
    run_dir = Path(checkpoint).parent.parent
    snapshot = _read_json(run_dir / "config_snapshot.json")
    return snapshot if snapshot.get("unique_id") == run_dir.name else {}


def find_phase1_checkpoint(subject_id: str, config_path: str, search_root: str = "runs") -> str | None:
    """Return the newest compatible Phase 1 model with its hash encoder."""
    config = _requested_config(config_path, subject_id)
    candidates = []
    for path in Path(search_root).rglob(f"{subject_id}_*_model.pt"):
        snapshot = _snapshot(str(path))
        args = snapshot.get("args")
        if not isinstance(args, dict) or snapshot.get("config") != config:
            continue
        if snapshot.get("parent_phase1_id") or args.get("phase1_checkpoint"):
            continue
        # CLI hash overrides are stored in args, not reflected in config_dict.
        generation = config.get("GENERATION_MODEL", {})
        defaults = {
            "hash_n_levels": ("HASH_N_LEVELS", 16),
            "hash_n_features_per_level": ("HASH_N_FEATURES_PER_LEVEL", 2),
            "hash_log2_size": ("HASH_LOG2_SIZE", 19),
            "hash_base_resolution": ("HASH_BASE_RESOLUTION", 16),
            "hash_per_level_scale": ("HASH_PER_LEVEL_SCALE", 1.39),
            "concat_coords": ("CONCAT_COORDS", False),
            "progressive_hash_unlock": ("PROGRESSIVE_HASH_UNLOCK", False),
            "hash_unlock_end_fraction": ("HASH_UNLOCK_END_FRACTION", 0.9),
        }
        if any(key in args and args[key] != generation.get(field, default)
               for key, (field, default) in defaults.items()):
            continue
        # Match experiment_utils.hash_ckpt_for without importing training deps.
        hash_path = Path(str(path).replace("_model", "_hash"))
        if _exists(path) and _exists(hash_path):
            candidates.append(path)
    return str(max(candidates, key=lambda p: p.stat().st_mtime)) if candidates else None


def valid_fast_exact_audit(audit: Any, requested_backend: str) -> bool:
    """Check the exact backend and freeze identity, including explicit requests."""
    if not isinstance(audit, dict) or audit.get("strict") is not True:
        return False
    backend = audit.get("bending_backend")
    if backend not in ("terms6", "analytic_siren"):
        return False
    if requested_backend != "auto" and backend != requested_backend:
        return False
    installer = "phase2_bending_siren_terms6" if backend == "terms6" else "phase2_bending_siren_analytic"
    return (
        audit.get("bending_installer") == f"sims_mri.optimization_candidates.{installer}.install_phase2_dropin"
        and audit.get("freeze_installer") == "sims_mri.training.phase2_registration_freeze_frozen.install_phase2_eval_freeze_dropin"
    )


def _matches_command(command: Any, subject_id: str, parent: str) -> bool:
    # Accept the command shape emitted by these runners, never unrecorded CLI
    # overrides whose effective values might differ from the config snapshot.
    if not isinstance(command, list) or len(command) < 3:
        return False
    if command[1:3] != ["-m", "sims_mri.training.phase2_registration_fast_exact"]:
        return False
    values = {}
    tokens = iter(command[3:])
    for flag in tokens:
        if flag == "--logging":
            continue
        if flag not in ("--config", "--subject_id", "--phase1_checkpoint") or flag in values:
            return False
        value = next(tokens, None)
        if not isinstance(value, str):
            return False
        values[flag] = value
    return (isinstance(values.get("--config"), str)
            and values.get("--subject_id") == subject_id
            and _same_path(values.get("--phase1_checkpoint"), parent))


def find_registration_resume(
    subject_id: str, config_path: str, phase1_checkpoint: str,
    requested_backend: str, search_root: str = "runs",
) -> RegistrationResume | None:
    """Select a successful recorded registration for this exact parent/config.

    Both existing full-pipeline and Phase-2-only batch result formats are read.
    Keep their result JSON in the checkpoint search root. Missing/legacy/failed
    records are not resumable; callers must run registration again.
    """
    config = _requested_config(config_path, subject_id)
    candidates = []
    for result_file in Path(search_root).glob("*.json"):
        subjects = _read_json(result_file).get("subjects", {})
        subject = subjects.get(subject_id, {}) if isinstance(subjects, dict) else {}
        if not isinstance(subject, dict):
            continue
        records = subject.get("phases", [subject])
        if not isinstance(records, list):
            continue
        for record in records:
            if (not isinstance(record, dict)
                    or record.get("phase") != "phase2_fast_exact"
                    or record.get("status") not in ("success", "skipped")):
                continue
            audit = record.get("fast_exact_audit")
            checkpoint = record.get("checkpoint_path", record.get("phase2_checkpoint"))
            if not isinstance(checkpoint, str) or not _exists(Path(checkpoint)):
                continue
            if not valid_fast_exact_audit(audit, requested_backend):
                continue
            if not _matches_command(record.get("command"), subject_id, phase1_checkpoint):
                continue
            snapshot = _snapshot(checkpoint)
            args = snapshot.get("args")
            if not isinstance(args, dict) or snapshot.get("config") != config:
                continue
            if not _same_path(args.get("phase1_checkpoint"), phase1_checkpoint):
                continue
            # Mirror experiment_utils.extract_parent_id_from_path: the label
            # is optional outside runs/<id>/. Exact paths above bind lineage.
            parent_match = re.search(r"runs[/\\]([^/\\]+)[/\\]", args["phase1_checkpoint"])
            parent_id = parent_match.group(1) if parent_match else None
            if snapshot.get("parent_phase1_id") != parent_id:
                continue
            candidates.append(RegistrationResume(
                checkpoint, audit, str(result_file), record.get("log_file"), record["command"]
            ))
    return max(candidates, key=lambda r: Path(r.checkpoint_path).stat().st_mtime) if candidates else None
