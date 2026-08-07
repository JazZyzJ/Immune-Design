"""Resolve ONE V2 Canary cell's config from the tracked template and the measured artifacts.

`doc/RF_Fusion_v2_Cluster_Runbook.md` §3 requires four resolved configs, one per
`(protein, r_step)` cell, materialized outside the repo and never by editing the tracked template.
Each carries eighteen PLAN §5.2 content identities, a threshold that must be copied VERBATIM out of
a calibration artifact, and a band digest that is deliberately NOT the band file's sha256.  Filling
that by hand is how a cell ends up signed under another cell's calibration, so it is done here and
checked here.

**The script emits the launch command with the config, and that is the point.**  A resolved config
declares which roles are frozen and which are observed at runtime; the driver's `--input-file` and
`--shard-input` lists must agree with that declaration or the run fails after the allocation is
paid.  Two hands maintaining the same fact is how they drift, so the producer that computes the
digests also writes the argument vectors that supply the files those digests came from
(`<out>.args.sh`, `source`-able from a SLURM script).

Everything is a CLI argument -- no cluster path appears in this module.

Three checks refuse rather than warn, because each is a way to produce a config that runs and is
wrong:

* the hotspot artifact's Head domain must equal the template's `(allele, score_scale, k_min,
  k_max)`.  `bind_admission_policy` compares that 4-tuple, and `N_H^whole` is a maximum over the
  grid it defines: a threshold measured under one domain does not bound designs scored under
  another.
* the band artifact must actually carry a cell at the requested `(r_step, stratum_key)`.
  `lookup_band` is exact and never interpolates, so a missing cell yields a typed null for every
  transition -- a fact about the band, reported after the GPU time, that is knowable here.
* no `REPLACE_*` placeholder may survive.  The loader refuses them, but it refuses at launch; this
  refuses at materialization with the field named.

Registered in `doc/SCRIPTS.md` under the RF-Refine Fusion V2 section.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

__all__ = [
    "build_parser", "main",
    "file_digest", "head_config_hash", "no_constraint_manifest_digest",
    "resolve_content_bindings", "fill_config",
]


class MaterializeError(RuntimeError):
    """A resolved config this producer refuses to write."""


def file_digest(path: Any) -> str:
    """SHA-256 over a file's bytes -- the same function the driver signs declared inputs with."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def head_config_hash(config_dir: Any) -> str:
    """The Head config directory's realized identity, read from the ONE implementation.

    `OnlineHeadScorer` reports this number as `head_config_hash`, and the V2 runtime compares the
    reference's evaluator identity against every endpoint's.  A second rolling hash here that
    agreed today would be a second implementation to drift tomorrow, so the producer imports v0's.
    """
    from scripts.run_if_phase_c1 import _compute_head_config_hash

    return _compute_head_config_hash(Path(config_dir))


def no_constraint_manifest_digest() -> str:
    """The project's canonical digest for "this cohort declares NO constraint manifest".

    An unconstrained protein has no manifest file, but PLAN §5.2 gives `constraint_manifest` no
    exemption: a run whose conditioning is simply silent about constraints cannot be told apart
    from one whose manifest went missing.  `rho_maturity_scan` already answered this -- it records
    a typed ABSENCE digest, and `5ZHV_B`'s own `B(r)` artifact carries it -- so the resolved config
    declares that same number rather than inventing a second convention or hashing a file written
    only to be hashed.  Being frozen, it also makes `--input-file constraint_manifest=...` refuse
    any manifest on an unconstrained cell.
    """
    from scripts.rho_maturity_scan import _canonical_digest

    return _canonical_digest({"schema": "scan-constraint-binding/1", "constraint_manifest": None})


def _band_content_digest(band_json: Any, *, r_step: int, stratum_key: str) -> str:
    """The band TABLE's `calibration_content_digest`, and proof the requested cell exists.

    Runbook §3 note 2: this role's `expected_sha256` is the table's canonical content digest, NOT
    the file's sha256.  `make_band_table` rebinds it and `q_phi` compares against it; the file
    digest would refuse every real run.
    """
    payload = json.loads(Path(band_json).read_text(encoding="utf-8"))
    provenance = payload.get("provenance") or {}
    digest = provenance.get("calibration_content_digest")
    if not digest:
        raise MaterializeError(f"{band_json} carries no provenance.calibration_content_digest")
    cells = [band for band in payload.get("bands") or ()
             if int(band.get("step", -1)) == int(r_step)
             and str(band.get("stratum_key")) == str(stratum_key)]
    if not cells:
        available = sorted({(int(b.get("step", -1)), str(b.get("stratum_key")))
                            for b in payload.get("bands") or ()})
        raise MaterializeError(
            f"{band_json} has no band at (step={r_step}, stratum_key={stratum_key!r}); it carries "
            f"{available}.  lookup_band is exact and never interpolates, so this cell would yield "
            "a typed null for every transition -- knowable now rather than after the GPU time"
        )
    return str(digest)


def _delta_new_block(hotspot_json: Any, *, head: dict) -> dict:
    """The `delta_new` block, copied VERBATIM, after checking it was measured on THIS Head domain."""
    payload = json.loads(Path(hotspot_json).read_text(encoding="utf-8"))
    block = payload.get("delta_new")
    if not isinstance(block, dict):
        raise MaterializeError(f"{hotspot_json} carries no delta_new block")
    artifact = block.get("artifact") or {}
    declared = (str(head["allele"]), str(head["score_scale"]),
                int(head["window_k_min"]), int(head["window_k_max"]))
    measured = (str(artifact.get("allele")), str(artifact.get("score_scale")),
                int(artifact.get("window_k_min", -1)), int(artifact.get("window_k_max", -1)))
    if declared != measured:
        names = ("allele", "score_scale", "window_k_min", "window_k_max")
        diff = ", ".join(f"{n}: config {d!r} != artifact {m!r}"
                         for n, d, m in zip(names, declared, measured) if d != m)
        raise MaterializeError(
            f"the hotspot threshold in {hotspot_json} was measured under a different Head domain "
            f"({diff}).  N_H^whole is a maximum over the grid these fields define, so a threshold "
            "measured under one domain does not bound designs scored under another"
        )
    return json.loads(json.dumps(block))  # a plain, YAML-safe copy


def resolve_content_bindings(args) -> tuple[dict[str, str | None], dict[str, str]]:
    """`role -> frozen digest or None`, plus `role -> the runtime file that supplies it`.

    A role appears in exactly one of the two.  The split is not cosmetic: a frozen role is bound
    before the run and a supplied file that contradicts it is refused, while a runtime role is
    signed by whatever bytes actually arrived.  Three roles here are frozen for a reason no file
    can express -- the Head config is a DIRECTORY rolling-hash, the band is a table digest rather
    than a file digest, and an unconstrained cell's constraint manifest is a typed absence.
    """
    constrained = bool(args.constraint_manifest)
    frozen: dict[str, str] = {
        "code_revision": str(args.code_revision),
        # THIS protein's canonical sequence file, not the manifest that resolves it.  A Canary cell
        # is one protein, and `bind_cumulative_reference` compares the digest it recomputes from
        # the reference BYTES against `policy.complete_reference_content_digest`, which is exactly
        # this field -- so a manifest digest here can never equal it and the shard dies with
        # `ReferenceRebindAttempt` after the checkpoints are resident.  The cohort-level manifest
        # is a resolution table and binds separately, as `reference_sequences`.
        "complete_reference_sequence": file_digest(args.reference_sequence),
        "projection_policy_spec": file_digest(args.projection_policy_spec),
        "schedule_band_calibration": _band_content_digest(
            args.band_json, r_step=int(args.r_step), stratum_key=str(args.stratum_key)),
        "head_config": head_config_hash(args.head_config_dir),
        "head_checkpoint": file_digest(args.head_checkpoint),
    }
    runtime: dict[str, str] = {
        "cohort_table": str(args.cohort_table),
        # Plural: the cohort's reference TABLE, which is what `complete_reference_manifest` reads.
        "reference_sequences": str(args.reference_manifest),
        "backbone": str(args.backbone),
        # The mask is DERIVED from these coordinates.  `_conditioning` replaces this digest with
        # the realized per-protein mask, so the file records where the mask came from.
        "coordinate_mask": str(args.backbone),
        "rf_sampler_config": str(args.rf_sampler_config),
        "dplm_checkpoint": str(args.dplm_checkpoint),
        # Likewise: the alphabet lives in the checkpoint, and `_conditioning` replaces this with
        # the realized tokenizer digest.
        "tokenizer": str(args.dplm_checkpoint),
        "structure_config": str(args.structure_config),
        "v0_structure_gate_config": str(args.v0_structure_gate_config),
        # The weights that fold every endpoint -- a single content-addressed file, which is a
        # stronger identity for the backend than the name of the code that calls it.
        "structure_backend": str(args.structure_backend),
    }
    if constrained:
        runtime["constraint_manifest"] = str(args.constraint_manifest)
        runtime["fixed_token_policy"] = str(args.constraint_manifest)
    else:
        absence = no_constraint_manifest_digest()
        frozen["constraint_manifest"] = absence
        frozen["fixed_token_policy"] = absence
    return frozen, runtime


def fill_config(template: dict, *, args, frozen: dict, runtime: dict) -> dict:
    """Return the resolved config; refuse if any `REPLACE_*` placeholder would survive."""
    config = json.loads(json.dumps(template))  # deep copy without YAML aliases

    config["identity"]["code_revision"] = str(args.code_revision)
    # The campaign names the EXPERIMENT.  Two runs that share a template but answer different
    # questions -- the Canary's "does it execute", the mechanism cohort's "does it transmit" --
    # must not sign their artifacts under one campaign, or the second reads as more of the first.
    if getattr(args, "campaign_id", None):
        config["identity"]["campaign_id"] = str(args.campaign_id)

    points = config["schedule"]["points"]
    if len(points) != 1:
        raise MaterializeError(
            f"the template declares {len(points)} schedule points; a Canary cell is ONE "
            "(protein, r_step) and a multi-point template cannot be resolved into one"
        )
    points[0]["r_step"] = int(args.r_step)
    points[0]["band_key"] = f"step{int(args.r_step)}"

    config["safety"]["delta_new_cumulative"] = _delta_new_block(
        args.hotspot_json, head=config["head"])

    for row in config["content"]:
        role = row["role"]
        if role in frozen:
            row["binding"], row["expected_sha256"] = "frozen", frozen[role]
        elif role in runtime:
            row["binding"], row["expected_sha256"] = "runtime", None
        else:
            raise MaterializeError(
                f"content role {role!r} was bound neither frozen nor runtime; PLAN §5.2 fails "
                "closed on missing content identity"
            )

    leftovers = sorted(_placeholders(config))
    if leftovers:
        raise MaterializeError(
            f"unresolved placeholder(s) remain: {leftovers}.  The loader would refuse them at "
            "launch; refusing here names the field instead"
        )
    return config


def _placeholders(node: Any, path: str = "") -> list[str]:
    if isinstance(node, dict):
        return [item for key, value in node.items()
                for item in _placeholders(value, f"{path}.{key}" if path else str(key))]
    if isinstance(node, list):
        return [item for index, value in enumerate(node)
                for item in _placeholders(value, f"{path}[{index}]")]
    return [path] if isinstance(node, str) and node.startswith("REPLACE") else []


def _args_script(args, *, frozen: dict, runtime: dict, config_path: Path) -> str:
    """The driver argument vectors this config requires, as a `source`-able shell fragment.

    `--input-file ROLE=PATH` signs the run and is checked against any frozen digest; `--shard-input
    NAME=PATH` is what the oracle stack and the conditioning identity actually read.  A role can
    need both, under two different names (`dplm_checkpoint` / `base_if_checkpoint`), which is
    exactly the kind of thing a hand-written command gets wrong once and then copies four times.

    `schedule_band_calibration`, `head_config`, `fixed_token_policy` and an unconstrained
    `constraint_manifest` are deliberately absent from `--input-file`: their frozen digests are a
    table digest, a directory rolling-hash, a policy label and a typed absence, so the driver's
    file-sha256 check would refuse the very files that produced them.
    """
    input_files = [
        # Frozen roles whose digest IS the file's sha256 -- these are the ones the driver can
        # genuinely verify, so declaring them turns the run signature into a real check.
        f"complete_reference_sequence={args.reference_sequence}",
        f"projection_policy_spec={args.projection_policy_spec}",
        f"head_checkpoint={args.head_checkpoint}",
    ] + [f"{role}={path}" for role, path in sorted(runtime.items())
         # coordinate_mask/tokenizer alias another role's file; recording them twice in the
         # manifest would claim two content identities where one file was read.
         if role not in {"coordinate_mask", "tokenizer", "fixed_token_policy"}]

    shard_inputs = dict(runtime)
    shard_inputs.update({
        # Names the oracle stack owns, distinct from the role vocabulary on purpose.
        "base_if_checkpoint": str(args.dplm_checkpoint),
        "test_set_parquet": str(args.cohort_table),
        "pdb_root": str(args.pdb_root),
        "refold_cache_dir": str(args.refold_cache_dir),
        "head_config": str(args.head_config_dir),
        "head_checkpoint": str(args.head_checkpoint),
        "head_variant_id": str(args.head_variant_id),
        "esmfold2_site_packages": str(args.esmfold2_site_packages),
        "schedule_band_calibration": str(args.band_json),
        "complete_reference_manifest": str(args.reference_manifest),
        "protein_stratum_manifest": str(args.stratum_manifest),
    })
    # `fixed_token_policy` stays in SHARD_INPUTS even though no oracle reads that key: it is one of
    # the eighteen roles `_conditioning` must find a digest for, and on an anchored cell it is bound
    # RUNTIME, so dropping it made the shard refuse with "no content identity for role(s)
    # ['fixed_token_policy']".  On an unconstrained cell it is frozen and absent from `runtime`, so
    # it never appears here at all.

    lines = [
        "# GENERATED by scripts/materialize_v2_canary_config.py -- do not edit by hand.",
        f"# cell: {args.protein_id} r={int(args.r_step)}  stratum={args.stratum_key}",
        f"V2_CONFIG={shlex.quote(str(config_path))}",
        f"V2_COHORT={shlex.quote(str(args.protein_id))}",
        "INPUT_FILES=(",
        *[f"  {shlex.quote(item)}" for item in input_files],
        ")",
        "SHARD_INPUTS=(",
        *[f"  {shlex.quote(f'{name}={path}')}" for name, path in sorted(shard_inputs.items())],
        ")",
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="materialize_v2_canary_config",
        description="resolve one V2 Canary cell's config + launch arguments from measured artifacts",
    )
    parser.add_argument("--template", required=True, help="tracked V2 Canary config template")
    parser.add_argument("--out", required=True, help="resolved config destination (.yaml)")
    parser.add_argument("--protein-id", required=True)
    parser.add_argument("--r-step", required=True, type=int)
    parser.add_argument("--stratum-key", required=True,
                        help="the COHORT stratum the band was measured over; not the band_key")
    parser.add_argument("--code-revision", required=True, help="7-64 lowercase hex; 'unknown' is refused")
    parser.add_argument("--band-json", required=True, help="this stratum's B(r) artifact")
    parser.add_argument("--hotspot-json", required=True, help="this protein's hotspot calibration")
    parser.add_argument("--reference-manifest", required=True, help="{protein_id: {path, sha256}}")
    parser.add_argument("--stratum-manifest", required=True, help="{protein_id: stratum_key}")
    parser.add_argument("--reference-sequence", required=True, help="this protein's .seq")
    parser.add_argument("--projection-policy-spec", required=True)
    parser.add_argument("--head-config-dir", required=True)
    parser.add_argument("--head-checkpoint", required=True)
    parser.add_argument("--head-variant-id", required=True)
    parser.add_argument("--structure-backend", required=True,
                        help="the structure model's weight file (its bytes ARE the backend)")
    parser.add_argument("--structure-config", required=True)
    parser.add_argument("--v0-structure-gate-config", required=True)
    parser.add_argument("--rf-sampler-config", required=True)
    parser.add_argument("--dplm-checkpoint", required=True)
    parser.add_argument("--cohort-table", required=True, help="the test-set parquet")
    parser.add_argument("--backbone", required=True, help="this protein's structure file")
    parser.add_argument("--pdb-root", required=True)
    parser.add_argument("--refold-cache-dir", required=True)
    parser.add_argument("--esmfold2-site-packages", required=True)
    parser.add_argument("--constraint-manifest", default=None,
                        help="omit for an unconstrained cell; its absence is then declared, "
                             "not left silent")
    parser.add_argument("--campaign-id", default=None,
                        help="override identity.campaign_id (default: the template's).  Use a "
                             "distinct campaign for a distinct question, e.g. the mechanism cohort")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out)

    template = yaml.safe_load(Path(args.template).read_text(encoding="utf-8"))
    frozen, runtime = resolve_content_bindings(args)
    config = fill_config(template, args=args, frozen=frozen, runtime=runtime)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(config, sort_keys=True, default_flow_style=False),
                   encoding="utf-8")

    # Written, then LOADED: a config this producer cannot itself resolve is not a config, and the
    # digest below is the one the run will sign, computed by the loader rather than predicted here.
    from scripts.rf_fusion_v2_preflight import load_v2_config_file

    resolved = load_v2_config_file(out)

    args_path = out.with_suffix(".args.sh")
    args_path.write_text(_args_script(args, frozen=frozen, runtime=runtime, config_path=out),
                         encoding="utf-8")

    print(json.dumps({
        "config": str(out),
        "args": str(args_path),
        "protein_id": args.protein_id,
        "r_step": int(args.r_step),
        "stratum_key": args.stratum_key,
        "config_digest": resolved.config_digest(),
        "delta_new_cumulative": resolved.safety.delta_new_cumulative.value,
        "delta_new_unit": resolved.safety.delta_new_cumulative.unit,
        "frozen_roles": {role: frozen[role] for role in sorted(frozen)},
        "runtime_roles": sorted(runtime),
    }, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
