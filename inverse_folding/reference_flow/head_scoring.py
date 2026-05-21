"""Reference-Flow head-scoring adapter for Phase D1 (and forward to D2/D3).

The Phase C1 sampler is denoiser-agnostic and must not import epitope-head
modules. ``OnlineHeadScorer`` is the single bridge between the sampler /
controller and the production ``InferencePredictor`` API. It exposes a stable
batch interface so D1 (one completion per refresh) and D2 (many block
candidates per refresh) hit the same code path.

The scorer also owns the static WT window-score cache: D1 active blocks need
window-level static-vs-dynamic excess risk, but B2 h-map artifacts do not
store per-window logits. The cache is keyed by

    (protein_id, sequence_md5, allele, head_checkpoint_digest,
     head_config_hash, score_scale, window_k_min, window_k_max)

and is persisted as a parquet table with a sidecar ``meta.json`` mirroring
the B2 h-map layout. Mismatched provenance fails fast on load.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from epitope_head.inference.predictor import InferencePredictor


_REQUIRED_META_KEYS = frozenset(
    {
        "allele",
        "head_checkpoint_digest",
        "head_config_hash",
        "score_scale",
        "window_k_min",
        "window_k_max",
    }
)


class HeadScoringCacheError(RuntimeError):
    """Raised when the static window-score cache is inconsistent."""


@dataclass(frozen=True)
class WindowRiskRecord:
    """One scoring window (start/end half-open, peptide length k) with raw logit."""

    start_0b: int
    end_0b: int
    k: int
    z: float


@dataclass(frozen=True)
class HeadScore:
    """One sequence scored under a fixed head/allele/scale."""

    protein_id: str
    sequence_md5: str
    sequence_length: int
    allele: str
    score_scale: str
    windows: tuple[WindowRiskRecord, ...]
    residue_hotspot: tuple[float, ...] | None = None
    global_risk: float | None = None


@dataclass(frozen=True)
class BatchHeadScores:
    """Ordered batch of ``HeadScore`` matching the caller's input order."""

    scores: tuple[HeadScore, ...]


@dataclass
class StaticWindowCache:
    """In-memory representation of the on-disk static window-score cache."""

    rows: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def rows_for(self, protein_id: str, sequence_md5: str) -> list[dict]:
        return [
            r for r in self.rows
            if r["protein_id"] == protein_id and r["sequence_md5"] == sequence_md5
        ]

    @classmethod
    def load(cls, cache_path: Path, meta_path: Path) -> "StaticWindowCache":
        if not cache_path.exists():
            raise HeadScoringCacheError(f"cache parquet not found: {cache_path}")
        if not meta_path.exists():
            raise HeadScoringCacheError(f"cache meta sidecar not found: {meta_path}")
        with open(meta_path) as f:
            meta = json.load(f)
        missing = _REQUIRED_META_KEYS - set(meta.keys())
        if missing:
            raise HeadScoringCacheError(
                f"static cache meta missing required keys: {sorted(missing)}"
            )
        df = pd.read_parquet(cache_path)
        rows = df.to_dict(orient="records")
        return cls(rows=rows, meta=meta)

    def save(self, cache_path: Path, meta_path: Path) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows).to_parquet(cache_path, index=False)
        with open(meta_path, "w") as f:
            json.dump(self.meta, f, indent=2, sort_keys=True)


class OnlineHeadScorer:
    """Stable batch interface plus static WT window cache.

    D1 calls ``score_batch_same_protein`` with one record per refresh. D2 will
    pass many block candidates through the same call without changing the
    sampler.
    """

    def __init__(
        self,
        *,
        predictor: InferencePredictor,
        allele: str,
        allele_idx: int,
        head_checkpoint_digest: str,
        head_config_hash: str,
        score_scale: str,
        window_k_min: int,
        window_k_max: int,
        static_cache_path: Path | str | None = None,
        static_cache_meta_path: Path | str | None = None,
        static_cache_policy: str = "lazy_write",
        window_batch_size: int | None = None,
    ) -> None:
        if score_scale != "raw_logit":
            raise ValueError(
                f"OnlineHeadScorer only supports score_scale='raw_logit' for D1 (got {score_scale!r})"
            )
        if static_cache_policy not in {"lazy_write", "read_only"}:
            raise ValueError(
                "static_cache_policy must be 'lazy_write' or 'read_only'"
            )
        self.predictor = predictor
        self.allele = str(allele)
        self.allele_idx = int(allele_idx)
        self.head_checkpoint_digest = str(head_checkpoint_digest)
        self.head_config_hash = str(head_config_hash)
        self.score_scale = score_scale
        self.window_k_min = int(window_k_min)
        self.window_k_max = int(window_k_max)
        self.static_cache_path = Path(static_cache_path) if static_cache_path else None
        self.static_cache_meta_path = (
            Path(static_cache_meta_path) if static_cache_meta_path else None
        )
        self.static_cache_policy = static_cache_policy
        self.window_batch_size = window_batch_size

        # In-memory static cache keyed by (protein_id, sequence_md5).
        self._static_cache: dict[tuple[str, str], HeadScore] = {}
        if self.static_cache_path is not None and self.static_cache_path.exists():
            if self.static_cache_meta_path is None or not self.static_cache_meta_path.exists():
                raise HeadScoringCacheError(
                    "static_cache_path exists but meta sidecar is missing"
                )
            cache = StaticWindowCache.load(self.static_cache_path, self.static_cache_meta_path)
            self._verify_meta(cache.meta)
            self._populate_from_rows(cache.rows)

    # ---------- batch scoring ----------

    def score_batch_same_protein(
        self,
        *,
        protein_id: str,
        records: list[tuple[str, str]],
    ) -> BatchHeadScores:
        """Score ``records`` (list of ``(label, sequence)``) under the bound allele.

        ``label`` is opaque to the scorer; D2 will set it to a candidate id. The
        scorer returns ``HeadScore`` per record with deterministic ``sequence_md5``.
        """
        if not records:
            return BatchHeadScores(scores=())

        flat_records = [(label, seq) for label, seq in records]
        sequences_only = [(label, seq) for label, seq in flat_records]
        predictor_records = [(label, seq) for label, seq in sequences_only]
        predictions = self.predictor.predict_proteins(
            predictor_records,
            allele_idx=self.allele_idx,
            window_batch_size=self.window_batch_size,
        )

        scores: list[HeadScore] = []
        for (label, seq), pred_row in zip(flat_records, predictions):
            prediction = pred_row["prediction"]
            scores.append(self._wrap_prediction(protein_id, seq, prediction))
        return BatchHeadScores(scores=tuple(scores))

    # ---------- static cache ----------

    def get_or_compute_static(self, protein_id: str, sequence: str) -> HeadScore:
        seq_md5 = _md5(sequence)
        key = (protein_id, seq_md5)
        cached = self._static_cache.get(key)
        if cached is not None:
            return cached
        batch = self.score_batch_same_protein(
            protein_id=protein_id, records=[("static", sequence)]
        )
        head_score = batch.scores[0]
        self._static_cache[key] = head_score
        return head_score

    def lookup_static(self, protein_id: str, sequence: str) -> HeadScore:
        """Pure cache lookup; raises if the requested entry is absent.

        Unlike ``get_or_compute_static`` this never invokes the head. D1 callers
        use this when the test set sequence is expected to already be present
        in the loaded cache; an unexpected sequence_md5 must surface as a hard
        error rather than silently computing a new entry.
        """
        seq_md5 = _md5(sequence)
        key = (protein_id, seq_md5)
        cached = self._static_cache.get(key)
        if cached is None:
            raise HeadScoringCacheError(
                f"static cache miss for protein_id={protein_id!r} sequence_md5={seq_md5!r}"
            )
        return cached

    def flush_static_cache(
        self,
        *,
        source_dataset: str,
        source_dataset_rowcount: int,
    ) -> None:
        if self.static_cache_path is None or self.static_cache_meta_path is None:
            raise HeadScoringCacheError(
                "static_cache_path and static_cache_meta_path must be set to flush"
            )
        if self.static_cache_policy == "read_only":
            raise HeadScoringCacheError("static cache is read_only; refusing to flush")

        rows: list[dict] = []
        for (protein_id, seq_md5), head_score in self._static_cache.items():
            for w in head_score.windows:
                rows.append({
                    "protein_id": protein_id,
                    "allele": self.allele,
                    "sequence_md5": seq_md5,
                    "sequence_length": head_score.sequence_length,
                    "window_start_0b": int(w.start_0b),
                    "window_end_0b": int(w.end_0b),
                    "k": int(w.k),
                    "z_static": float(w.z),
                    "head_checkpoint_digest": self.head_checkpoint_digest,
                    "head_config_hash": self.head_config_hash,
                    "score_scale": self.score_scale,
                    "window_k_min": self.window_k_min,
                    "window_k_max": self.window_k_max,
                })

        meta = {
            "allele": self.allele,
            "head_checkpoint_path": self.predictor.checkpoint_metadata.get(
                "checkpoint_path", ""
            ),
            "head_checkpoint_digest": self.head_checkpoint_digest,
            "head_variant_id": self.predictor.checkpoint_metadata.get("variant_id", ""),
            "head_config_hash": self.head_config_hash,
            "score_scale": self.score_scale,
            "window_k_min": self.window_k_min,
            "window_k_max": self.window_k_max,
            "n_proteins_total": len({pid for pid, _ in self._static_cache.keys()}),
            "n_windows_total": len(rows),
            "source_dataset": source_dataset,
            "source_dataset_rowcount": int(source_dataset_rowcount),
            "git_commit": _read_git_commit(),
            "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        }
        cache = StaticWindowCache(rows=rows, meta=meta)
        cache.save(self.static_cache_path, self.static_cache_meta_path)

    # ---------- internals ----------

    def _wrap_prediction(
        self, protein_id: str, sequence: str, prediction: dict
    ) -> HeadScore:
        windows = tuple(
            WindowRiskRecord(
                start_0b=int(w["start_0b"]),
                end_0b=int(w["end_0b"]),
                k=int(w["k"]),
                z=float(w["z"]),
            )
            for w in prediction["window_logits"]
        )
        residue_hotspot_tensor = prediction.get("residue_hotspot")
        residue_hotspot: tuple[float, ...] | None = (
            tuple(float(x) for x in residue_hotspot_tensor.tolist())
            if residue_hotspot_tensor is not None
            else None
        )
        global_risk = prediction.get("global_risk")
        return HeadScore(
            protein_id=protein_id,
            sequence_md5=_md5(sequence),
            sequence_length=len(sequence),
            allele=self.allele,
            score_scale=self.score_scale,
            windows=windows,
            residue_hotspot=residue_hotspot,
            global_risk=float(global_risk) if global_risk is not None else None,
        )

    def _verify_meta(self, meta: dict) -> None:
        expected = {
            "allele": self.allele,
            "head_checkpoint_digest": self.head_checkpoint_digest,
            "head_config_hash": self.head_config_hash,
            "score_scale": self.score_scale,
            "window_k_min": self.window_k_min,
            "window_k_max": self.window_k_max,
        }
        for key, want in expected.items():
            have = meta.get(key)
            if have != want:
                if key.startswith("window_k"):
                    raise HeadScoringCacheError(
                        f"static cache meta mismatch on {key}: "
                        f"loaded {have!r} vs runtime {want!r}; window_k range drift"
                    )
                raise HeadScoringCacheError(
                    f"static cache meta mismatch on {key}: "
                    f"loaded {have!r} vs runtime {want!r}"
                )

    def _populate_from_rows(self, rows: Iterable[dict]) -> None:
        grouped: dict[tuple[str, str], list[dict]] = {}
        for row in rows:
            key = (str(row["protein_id"]), str(row["sequence_md5"]))
            grouped.setdefault(key, []).append(row)
        for (protein_id, sequence_md5), row_list in grouped.items():
            row_list_sorted = sorted(
                row_list, key=lambda r: (int(r["window_start_0b"]), int(r["k"]))
            )
            windows = tuple(
                WindowRiskRecord(
                    start_0b=int(r["window_start_0b"]),
                    end_0b=int(r["window_end_0b"]),
                    k=int(r["k"]),
                    z=float(r["z_static"]),
                )
                for r in row_list_sorted
            )
            sequence_length = int(row_list_sorted[0]["sequence_length"])
            self._static_cache[(protein_id, sequence_md5)] = HeadScore(
                protein_id=protein_id,
                sequence_md5=sequence_md5,
                sequence_length=sequence_length,
                allele=self.allele,
                score_scale=self.score_scale,
                windows=windows,
            )


def _md5(seq: str) -> str:
    return hashlib.md5(seq.encode("utf-8")).hexdigest()


def _read_git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"
