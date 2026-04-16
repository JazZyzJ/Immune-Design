"""Structural-context feature extraction for MHC-II epitope analysis.

Given a Tier 1 candidate entry (from `tier1_candidates.json`) and its PDB file,
produce a per-residue DataFrame with:
    - `is_epitope` (0/1 from union of IEDB EL spans, peptide-string verified)
    - `rsa`        (DSSP relative SASA)
    - `ss8` / `ss3` (DSSP 8-state and collapsed 3-state)
    - `bfactor_ca` / `bfactor_ca_z` (raw and within-protein z-scored Cα B-factor)
    - `contact_number` (Cα neighbors within 8 Å, sequence-adjacent excluded)

Strictness contract (non-negotiable; caller must not catch these and move on):
    1. PDB ATOM → FASTA sequence alignment. Author residue numbers are expected
       to follow `chain_range` absolute numbering (author_resnum = chain_start +
       seq_idx). Any aa mismatch, duplicate mapping, or out-of-range author_resnum
       raises `AlignmentError` with every offending position listed.
    2. Epitope span (`start_0b`, `end_0b`) → sequence slice is verified against
       the stored `peptide` string. Mismatch raises `SpanSanityError`.
    3. Residues without an ATOM record (crystallographically disordered) keep
       `is_epitope` intact but have NaN structural features. They are kept in
       the table — downstream stats drop NaN rows per metric rather than the
       loader hiding them.

Module-level helpers also provide effect-size statistics (Cliff's δ,
Mann-Whitney U) and a DerSimonian-Laird random-effects meta-analysis used for
cross-protein pooling of the categorical SS-coil enrichment.
"""
from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────

# Tien et al. 2013 theoretical Gly-X-Gly maxASA (Å²), used only if a RSA value
# from DSSP is missing for a standard residue.
MAX_ASA_TIEN = {
    "A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0,
    "E": 223.0, "Q": 225.0, "G": 104.0, "H": 224.0, "I": 197.0,
    "L": 201.0, "K": 236.0, "M": 224.0, "F": 240.0, "P": 159.0,
    "S": 155.0, "T": 172.0, "W": 285.0, "Y": 263.0, "V": 174.0,
}

# DSSP 8-state → 3-state collapse:
#   H (α-helix), G (3-10 helix), I (π-helix)      → H
#   E (β-strand), B (β-bridge)                    → E
#   T (turn), S (bend), C/-/space (coil), P       → C
SS8_TO_SS3 = {
    "H": "H", "G": "H", "I": "H",
    "E": "E", "B": "E",
    "T": "C", "S": "C", "C": "C", "-": "C", " ": "C", "P": "C",
}

AA3_TO_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",  # selenomethionine — treat as methionine for sequence purposes
}


class AlignmentError(ValueError):
    """Raised when strict ATOM→FASTA alignment fails."""


class SpanSanityError(ValueError):
    """Raised when an IEDB span's peptide string does not match the sequence slice."""


class DSSPError(RuntimeError):
    """Raised when DSSP execution/parsing fails for a structure file."""


@dataclass(frozen=True)
class AtomResidue:
    """Minimal view of a chain residue carrying a Cα atom."""
    author_resnum: int
    ins_code: str
    aa1: str
    ca_coord: np.ndarray  # shape (3,)
    ca_bfactor: float


# ─────────────────────────────────────────────────────────────────────────
# PDB parsing (chain-restricted)
# ─────────────────────────────────────────────────────────────────────────

def parse_pdb_chain(
    pdb_path: Path,
    chain_id: str,
    model_idx: int = 0,
) -> List[AtomResidue]:
    """Return an ordered list of Cα-carrying residues for the requested chain."""
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", str(pdb_path))
    models = list(structure.get_models())
    if model_idx >= len(models):
        raise ValueError(
            f"{pdb_path}: requested model_idx={model_idx} but structure has "
            f"{len(models)} model(s)"
        )
    model = models[model_idx]

    chain = next((ch for ch in model.get_chains() if ch.id == chain_id), None)
    if chain is None:
        raise ValueError(f"{pdb_path}: chain '{chain_id}' not found")

    residues: List[AtomResidue] = []
    for res in chain.get_residues():
        het, resnum, icode = res.id
        # Accept standard residues only. Accept MSE (selenomethionine) since
        # it is chemically Met-like; skip everything else with a DEBUG log.
        aa1 = AA3_TO_AA1.get(res.resname)
        if aa1 is None:
            logger.debug(
                "Skipping non-standard residue %s at %s%d (het=%r) in %s:%s",
                res.resname, icode.strip() or "", resnum, het, pdb_path.name, chain_id,
            )
            continue
        if "CA" not in res:
            logger.debug(
                "Skipping residue %s %s%d (no CA) in %s:%s",
                res.resname, icode.strip() or "", resnum, pdb_path.name, chain_id,
            )
            continue
        ca = res["CA"]
        residues.append(AtomResidue(
            author_resnum=int(resnum),
            ins_code=icode if isinstance(icode, str) else " ",
            aa1=aa1,
            ca_coord=np.array(ca.get_coord(), dtype=np.float64),
            ca_bfactor=float(ca.get_bfactor()),
        ))
    return residues


# ─────────────────────────────────────────────────────────────────────────
# Strict alignment
# ─────────────────────────────────────────────────────────────────────────

def _strict_align_atom_to_fasta(
    atom_residues: Sequence[AtomResidue],
    fasta_seq: str,
    chain_range_start: int,
) -> Tuple[List[Optional[int]], List[Optional[AtomResidue]]]:
    seq_len = len(fasta_seq)
    seq_to_atom: List[Optional[AtomResidue]] = [None] * seq_len
    seq_to_author: List[Optional[int]] = [None] * seq_len

    out_of_range: List[str] = []
    mismatches: List[str] = []
    duplicates: List[str] = []

    for r in atom_residues:
        if r.ins_code.strip() != "":
            # Insertion codes in the chain range are ambiguous for
            # author_resnum→seq_idx arithmetic. Fail loudly.
            mismatches.append(
                f"insertion code '{r.ins_code.strip()}' at author_resnum={r.author_resnum} "
                "violates strict mapping (author_resnum unique-per-seq_idx assumption)"
            )
            continue
        seq_idx = r.author_resnum - chain_range_start
        if seq_idx < 0 or seq_idx >= seq_len:
            out_of_range.append(
                f"author_resnum={r.author_resnum} → seq_idx={seq_idx} outside [0, {seq_len})"
            )
            continue
        if fasta_seq[seq_idx] != r.aa1:
            mismatches.append(
                f"seq_idx={seq_idx} (author_resnum={r.author_resnum}): "
                f"FASTA='{fasta_seq[seq_idx]}' vs ATOM='{r.aa1}'"
            )
            continue
        if seq_to_atom[seq_idx] is not None:
            prev = seq_to_atom[seq_idx]
            duplicates.append(
                f"seq_idx={seq_idx}: already mapped to author_resnum={prev.author_resnum}, "
                f"conflict with author_resnum={r.author_resnum}"
            )
            continue
        seq_to_atom[seq_idx] = r
        seq_to_author[seq_idx] = r.author_resnum

    problems: List[str] = []
    if out_of_range:
        problems.append(
            "Out-of-range ATOM residues (author_resnum outside chain_range):\n  "
            + "\n  ".join(out_of_range[:20])
        )
    if mismatches:
        problems.append(
            "ATOM vs FASTA residue mismatches:\n  " + "\n  ".join(mismatches[:20])
        )
    if duplicates:
        problems.append(
            "Duplicate seq_idx mappings:\n  " + "\n  ".join(duplicates[:20])
        )
    if problems:
        raise AlignmentError(
            "Strict ATOM→FASTA alignment failed.\n" + "\n".join(problems)
        )

    return seq_to_author, seq_to_atom


def _map_contiguous_match(
    atom_residues: Sequence[AtomResidue],
    fasta_seq: str,
) -> Optional[Tuple[List[Optional[int]], List[Optional[AtomResidue]], dict]]:
    atom_seq = "".join(r.aa1 for r in atom_residues)
    if not atom_seq:
        return None

    seq_to_atom: List[Optional[AtomResidue]] = [None] * len(fasta_seq)
    seq_to_author: List[Optional[int]] = [None] * len(fasta_seq)

    atom_in_fasta: List[int] = []
    start = fasta_seq.find(atom_seq)
    while start != -1:
        atom_in_fasta.append(start)
        start = fasta_seq.find(atom_seq, start + 1)

    if atom_in_fasta:
        if len(atom_in_fasta) > 1:
            raise AlignmentError(
                "Fallback ATOM→FASTA alignment is ambiguous: atom-derived sequence "
                f"matches FASTA at multiple offsets {atom_in_fasta[:5]}"
            )
        start_idx = atom_in_fasta[0]
        for i, r in enumerate(atom_residues):
            seq_idx = start_idx + i
            seq_to_atom[seq_idx] = r
            seq_to_author[seq_idx] = r.author_resnum
        return seq_to_author, seq_to_atom, {
            "mode": "atom_in_fasta",
            "start_idx": start_idx,
            "aligned_len": len(atom_residues),
        }

    fasta_in_atom: List[int] = []
    start = atom_seq.find(fasta_seq)
    while start != -1:
        fasta_in_atom.append(start)
        start = atom_seq.find(fasta_seq, start + 1)

    if not fasta_in_atom:
        return None
    if len(fasta_in_atom) > 1:
        raise AlignmentError(
            "Fallback ATOM→FASTA alignment is ambiguous: FASTA matches atom-derived "
            f"sequence at multiple offsets {fasta_in_atom[:5]}"
        )
    start_idx = fasta_in_atom[0]
    for seq_idx, r in enumerate(atom_residues[start_idx:start_idx + len(fasta_seq)]):
        seq_to_atom[seq_idx] = r
        seq_to_author[seq_idx] = r.author_resnum
    return seq_to_author, seq_to_atom, {
        "mode": "fasta_in_atom",
        "start_idx": start_idx,
        "aligned_len": len(fasta_seq),
    }


def _map_via_pairwise_alignment(
    atom_residues: Sequence[AtomResidue],
    fasta_seq: str,
) -> Tuple[List[Optional[int]], List[Optional[AtomResidue]], dict]:
    from Bio import Align

    atom_seq = "".join(r.aa1 for r in atom_residues)
    if not atom_seq:
        raise AlignmentError("Cannot run fallback alignment on an empty ATOM sequence")

    aligner = Align.PairwiseAligner(mode="global")
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -2.0
    aligner.extend_gap_score = -0.5

    alignments = aligner.align(fasta_seq, atom_seq)
    if len(alignments) == 0:
        raise AlignmentError("Fallback pairwise alignment found no candidate mapping")

    top = alignments[0]
    top_blocks = [(tuple(t), tuple(q)) for t, q in zip(top.aligned[0], top.aligned[1])]
    if len(alignments) > 1:
        second = alignments[1]
        second_blocks = [(tuple(t), tuple(q)) for t, q in zip(second.aligned[0], second.aligned[1])]
        if second.score == top.score and second_blocks != top_blocks:
            raise AlignmentError(
                "Fallback ATOM→FASTA alignment is ambiguous: multiple highest-scoring "
                "sequence alignments exist"
            )

    seq_to_atom: List[Optional[AtomResidue]] = [None] * len(fasta_seq)
    seq_to_author: List[Optional[int]] = [None] * len(fasta_seq)
    aligned_pairs = 0
    matches = 0
    mismatches = 0
    for (target_start, target_end), (query_start, query_end) in top_blocks:
        span = target_end - target_start
        if span != (query_end - query_start):
            raise AlignmentError("Fallback alignment produced unequal block lengths")
        for delta in range(span):
            seq_idx = target_start + delta
            atom_idx = query_start + delta
            r = atom_residues[atom_idx]
            seq_to_atom[seq_idx] = r
            seq_to_author[seq_idx] = r.author_resnum
            aligned_pairs += 1
            if fasta_seq[seq_idx] == r.aa1:
                matches += 1
            else:
                mismatches += 1

    atom_coverage = aligned_pairs / len(atom_seq)
    fasta_coverage = aligned_pairs / len(fasta_seq) if fasta_seq else 0.0
    identity = matches / aligned_pairs if aligned_pairs else 0.0
    stats = {
        "aligned_pairs": aligned_pairs,
        "matches": matches,
        "mismatches": mismatches,
        "identity": identity,
        "atom_coverage": atom_coverage,
        "fasta_coverage": fasta_coverage,
        "score": float(top.score),
    }
    if atom_coverage < 0.95 or identity < 0.97 or fasta_coverage < 0.85:
        raise AlignmentError(
            "Fallback ATOM→FASTA alignment quality too low: "
            f"identity={identity:.3f}, atom_coverage={atom_coverage:.3f}, "
            f"fasta_coverage={fasta_coverage:.3f}, mismatches={mismatches}"
        )
    return seq_to_author, seq_to_atom, stats


def align_atom_to_fasta(
    atom_residues: Sequence[AtomResidue],
    fasta_seq: str,
    chain_range_start: int,
) -> Tuple[List[Optional[int]], List[Optional[AtomResidue]]]:
    """Map each FASTA seq_idx → (author_resnum, AtomResidue) or None.

    Preferred policy is the original strict contract:
    `author_resnum = chain_range_start + seq_idx`.

    Fallback policy (used only if strict mapping fails) is deliberately narrow:
    1. unique exact contiguous ATOM-sequence match within FASTA
    2. otherwise, a unique high-identity global sequence alignment

    The fallback is intended for structures whose residue numbering and/or
    termini differ from the candidate FASTA while preserving a reliable residue
    correspondence. Low-quality or ambiguous alignments still fail loudly.
    """
    try:
        return _strict_align_atom_to_fasta(atom_residues, fasta_seq, chain_range_start)
    except AlignmentError as strict_err:
        contiguous = _map_contiguous_match(atom_residues, fasta_seq)
        if contiguous is not None:
            seq_to_author, seq_to_atom, info = contiguous
            logger.warning(
                "Strict ATOM→FASTA mapping failed; accepted contiguous-sequence fallback "
                "(mode=%s, start=%d, aligned_len=%d, atom_len=%d, chain_range_start=%d): %s",
                info["mode"],
                info["start_idx"],
                info["aligned_len"],
                len(atom_residues),
                chain_range_start,
                strict_err,
            )
            return seq_to_author, seq_to_atom

        seq_to_author, seq_to_atom, stats = _map_via_pairwise_alignment(atom_residues, fasta_seq)
        logger.warning(
            "Strict ATOM→FASTA mapping failed; accepted pairwise-sequence fallback "
            "(identity=%.3f, atom_coverage=%.3f, fasta_coverage=%.3f, mismatches=%d): %s",
            stats["identity"],
            stats["atom_coverage"],
            stats["fasta_coverage"],
            stats["mismatches"],
            strict_err,
        )
        return seq_to_author, seq_to_atom


# ─────────────────────────────────────────────────────────────────────────
# is_epitope with peptide sanity check
# ─────────────────────────────────────────────────────────────────────────

def build_is_epitope(
    spans: Iterable[dict],
    fasta_seq: str,
    chain_range_start: int,
) -> np.ndarray:
    """Return is_epitope[i] ∈ {0,1} over union of spans. Verifies peptide text.

    Coordinate contract (verified against all 261 Tier 1 spans in
    `outputs/if/test_set/tier1_candidates.json`, 2026-04-16):

    - `chain_range_start` is the **1-based** first author residue number of the
      chain as stored in `tier1_candidates.json::chain_range`. For 1LI1_C
      ("chain_range": "1485-1712") this is 1485 and the chain FASTA residue 0
      corresponds to author residue 1485.
    - `start_0b` / `end_0b` form a **half-open** [start_0b, end_0b) interval in
      **0-based absolute** coordinates, i.e. `end_0b − start_0b == len(peptide)`
      on all spans in the JSON.
    - The conversion from absolute to sequence-local is therefore
      `seq_idx = abs_pos − (chain_range_start − 1)`.

    Any span whose mapped slice does not equal its stored `peptide` string, or
    falls outside `[0, len(fasta_seq)]`, raises `SpanSanityError` with every
    offending span listed. No silent drops.
    """
    seq_len = len(fasta_seq)
    abs_offset = chain_range_start - 1
    is_ep = np.zeros(seq_len, dtype=np.int8)
    problems: List[str] = []
    for span in spans:
        start = int(span["start_0b"]) - abs_offset
        end = int(span["end_0b"]) - abs_offset
        peptide = span.get("peptide", "") or ""
        if start < 0 or end > seq_len or start >= end:
            problems.append(
                f"span ({span['start_0b']},{span['end_0b']}) → "
                f"seq range [{start},{end}) outside [0,{seq_len}]"
            )
            continue
        slice_aa = fasta_seq[start:end]
        if peptide and slice_aa != peptide:
            problems.append(
                f"span ({span['start_0b']},{span['end_0b']}): "
                f"expected peptide='{peptide}' vs sequence slice='{slice_aa}'"
            )
            continue
        is_ep[start:end] = 1
    if problems:
        raise SpanSanityError(
            "Epitope span sanity-check failed:\n  " + "\n  ".join(problems)
        )
    return is_ep


# ─────────────────────────────────────────────────────────────────────────
# DSSP: RSA + SS-8
# ─────────────────────────────────────────────────────────────────────────

def _which(prog: str) -> Optional[str]:
    for d in os.environ.get("PATH", "").split(os.pathsep):
        candidate = os.path.join(d, prog)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _find_dssp_binary() -> str:
    for name in ("mkdssp", "dssp"):
        path = _which(name)
        if path:
            return path
    raise RuntimeError(
        "DSSP binary not found in PATH (tried 'mkdssp' and 'dssp'). "
        "Install the DSSP package or point --dssp-bin at the executable."
    )


def compute_rsa_ss_via_dssp(
    pdb_path: Path,
    chain_id: str,
    dssp_bin: Optional[str] = None,
    model_idx: int = 0,
) -> Dict[Tuple[int, str], Tuple[float, str]]:
    """Run DSSP and return {(author_resnum, ins_code): (rsa, ss8)} for one chain.

    `rsa` is the relative solvent accessibility returned by BioPython's DSSP
    wrapper (already normalized to [0, ~1]). Missing values come back as NaN.
    `ss8` uses raw DSSP codes; ' '/'-' are preserved for SS8_TO_SS3 to collapse.
    """
    from Bio.PDB import PDBIO, PDBParser
    from Bio.PDB.DSSP import DSSP

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", str(pdb_path))
    models = list(structure.get_models())
    if model_idx >= len(models):
        raise ValueError(
            f"{pdb_path}: requested model_idx={model_idx}, have {len(models)} models"
        )
    model = models[model_idx]

    dssp_bin = dssp_bin or _find_dssp_binary()
    try:
        # Re-serialize to a minimal temporary PDB before invoking DSSP. Some RCSB
        # downloads contain header records that mkdssp rejects even though the
        # coordinate section itself parses fine.
        with tempfile.TemporaryDirectory(prefix="dssp_") as tmpdir:
            tmp_pdb = Path(tmpdir) / f"{pdb_path.stem}.pdb"
            io = PDBIO()
            io.set_structure(structure)
            io.save(str(tmp_pdb))
            dssp = DSSP(model, str(tmp_pdb), dssp=dssp_bin)
    except Exception as e:
        raise DSSPError(f"{pdb_path}: DSSP failed via '{dssp_bin}': {e}") from e

    out: Dict[Tuple[int, str], Tuple[float, str]] = {}
    for key in dssp.keys():
        ch_id, res_id = key
        if ch_id != chain_id:
            continue
        _het, resnum, icode = res_id
        record = dssp[key]
        # Biopython DSSP record: (index, aa, ss, rel_acc, phi, psi, ...)
        ss8 = record[2] if record[2] not in ("", None) else "-"
        rel_acc = record[3]
        try:
            rsa = float(rel_acc)
        except (TypeError, ValueError):
            rsa = float("nan")
        out[(int(resnum), icode if isinstance(icode, str) else " ")] = (rsa, ss8)
    return out


# ─────────────────────────────────────────────────────────────────────────
# Contact number
# ─────────────────────────────────────────────────────────────────────────

def compute_contact_numbers(
    residues: Sequence[AtomResidue],
    cutoff_ang: float = 8.0,
    seq_excl: int = 1,
) -> np.ndarray:
    """Per-residue Cα contact count within `cutoff_ang` Å.

    Self and author_resnum neighbors within ±seq_excl are excluded. This uses
    the author residue numbers, which handles gaps in the ATOM-ordered list
    correctly (e.g., a disordered loop).
    """
    n = len(residues)
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    coords = np.stack([r.ca_coord for r in residues], axis=0)
    resnums = np.array([r.author_resnum for r in residues], dtype=np.int64)
    d2 = np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1)
    within = d2 <= (cutoff_ang ** 2)
    np.fill_diagonal(within, False)
    if seq_excl > 0:
        close_in_seq = np.abs(resnums[:, None] - resnums[None, :]) <= seq_excl
        within &= ~close_in_seq
    return within.sum(axis=1).astype(np.int32)


# ─────────────────────────────────────────────────────────────────────────
# Top-level builder
# ─────────────────────────────────────────────────────────────────────────

def build_residue_table(
    protein_entry: dict,
    pdb_path: Path,
    dssp_bin: Optional[str] = None,
    contact_cutoff_ang: float = 8.0,
    contact_seq_excl: int = 1,
) -> pd.DataFrame:
    """Full per-residue feature table for one Tier 1 candidate.

    Columns: protein_id, seq_idx, author_resnum, aa, is_epitope, rsa,
             ss8, ss3, bfactor_ca, bfactor_ca_z, contact_number.
    """
    chain_id = protein_entry["chain"]
    fasta_seq = protein_entry["sequence"]
    chain_range = protein_entry["chain_range"]  # e.g. "1485-1712"
    chain_start_abs = int(chain_range.split("-")[0])
    protein_id = protein_entry["protein_id"]

    atom_residues = parse_pdb_chain(pdb_path, chain_id)
    if not atom_residues:
        raise ValueError(
            f"{pdb_path}: no Cα-bearing residues found in chain {chain_id}"
        )

    seq_to_author, seq_to_atom = align_atom_to_fasta(
        atom_residues, fasta_seq, chain_start_abs
    )

    is_epitope = build_is_epitope(
        protein_entry.get("experimental_epitopes", []),
        fasta_seq,
        chain_start_abs,
    )

    rsa_ss_map = compute_rsa_ss_via_dssp(pdb_path, chain_id, dssp_bin=dssp_bin)

    cn_values = compute_contact_numbers(
        atom_residues, cutoff_ang=contact_cutoff_ang, seq_excl=contact_seq_excl,
    )
    atom_id_to_cn = {id(r): int(cn_values[i]) for i, r in enumerate(atom_residues)}

    rows = []
    for i, aa in enumerate(fasta_seq):
        r = seq_to_atom[i]
        row = {
            "protein_id": protein_id,
            "seq_idx": i,
            "author_resnum": seq_to_author[i] if seq_to_author[i] is not None else pd.NA,
            "aa": aa,
            "is_epitope": int(is_epitope[i]),
        }
        if r is None:
            row.update({
                "rsa": np.nan,
                "ss8": pd.NA,
                "ss3": pd.NA,
                "bfactor_ca": np.nan,
                "contact_number": pd.NA,
            })
        else:
            rsa, ss8 = rsa_ss_map.get((r.author_resnum, r.ins_code), (np.nan, "-"))
            ss3 = SS8_TO_SS3.get(ss8, "C")
            row.update({
                "rsa": float(rsa),
                "ss8": ss8,
                "ss3": ss3,
                "bfactor_ca": float(r.ca_bfactor),
                "contact_number": atom_id_to_cn.get(id(r), pd.NA),
            })
        rows.append(row)

    df = pd.DataFrame(rows)

    # Within-protein B-factor z-score. Only over residues with valid bfactor.
    bvals = df["bfactor_ca"].astype(float)
    valid = bvals.notna()
    df["bfactor_ca_z"] = np.nan
    if valid.sum() > 1:
        mu = bvals[valid].mean()
        sd = bvals[valid].std(ddof=1)
        if sd and sd > 0:
            df.loc[valid, "bfactor_ca_z"] = (bvals[valid] - mu) / sd
        else:
            df.loc[valid, "bfactor_ca_z"] = 0.0

    return df


# ─────────────────────────────────────────────────────────────────────────
# Effect sizes and random-effects meta
# ─────────────────────────────────────────────────────────────────────────

def cliffs_delta(pos: np.ndarray, neg: np.ndarray) -> float:
    """Cliff's δ ∈ [-1, 1]. Vectorized rank-based implementation."""
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    pos = pos[~np.isnan(pos)]
    neg = neg[~np.isnan(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    all_vals = np.concatenate([pos, neg])
    ranks = pd.Series(all_vals).rank(method="average").to_numpy()
    r_pos = ranks[:pos.size].sum()
    U = r_pos - pos.size * (pos.size + 1) / 2.0  # Mann-Whitney U for pos
    delta = 2.0 * U / (pos.size * neg.size) - 1.0
    return float(delta)


def mannwhitney_p(pos: np.ndarray, neg: np.ndarray) -> float:
    """Two-sided Mann-Whitney U p-value via scipy."""
    from scipy.stats import mannwhitneyu

    pos = np.asarray(pos, dtype=float); pos = pos[~np.isnan(pos)]
    neg = np.asarray(neg, dtype=float); neg = neg[~np.isnan(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    try:
        return float(mannwhitneyu(pos, neg, alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


def log_or_coil(df: pd.DataFrame) -> Tuple[float, float, Tuple[int, int, int, int]]:
    """Per-protein log-odds-ratio for SS3=='C' given epitope.

    Uses Haldane-Anscombe 0.5 continuity correction if any cell is zero.
    Returns (log_or, se, (a,b,c,d)) where:
        a = epitope∩coil, b = epitope∩non-coil,
        c = non-epitope∩coil, d = non-epitope∩non-coil.
    """
    sub = df.dropna(subset=["ss3"]).copy()
    is_coil = (sub["ss3"] == "C").astype(int).to_numpy()
    is_ep = sub["is_epitope"].astype(int).to_numpy()
    a = int(((is_ep == 1) & (is_coil == 1)).sum())
    b = int(((is_ep == 1) & (is_coil == 0)).sum())
    c = int(((is_ep == 0) & (is_coil == 1)).sum())
    d = int(((is_ep == 0) & (is_coil == 0)).sum())
    counts = (a, b, c, d)
    af, bf, cf, df_ = float(a), float(b), float(c), float(d)
    if 0 in counts:
        af += 0.5; bf += 0.5; cf += 0.5; df_ += 0.5
    num = af * df_
    den = bf * cf
    if num <= 0 or den <= 0:
        return float("nan"), float("nan"), counts
    log_or = float(np.log(num / den))
    se = float(np.sqrt(1.0 / af + 1.0 / bf + 1.0 / cf + 1.0 / df_))
    return log_or, se, counts


def random_effects_meta(effects: np.ndarray, ses: np.ndarray) -> dict:
    """DerSimonian-Laird random-effects meta-analysis.

    Returns dict with pooled estimate, SE, 95 % CI, between-study variance
    τ², Cochran's Q, I² heterogeneity, and k (# studies retained).
    """
    effects = np.asarray(effects, dtype=float)
    ses = np.asarray(ses, dtype=float)
    keep = np.isfinite(effects) & np.isfinite(ses) & (ses > 0)
    effects = effects[keep]
    ses = ses[keep]
    k = int(effects.size)
    if k == 0:
        return {"pooled": float("nan"), "se": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"),
                "tau2": float("nan"), "Q": float("nan"),
                "I2": float("nan"), "k": 0}

    w_fe = 1.0 / (ses ** 2)
    theta_fe = float((w_fe * effects).sum() / w_fe.sum())
    Q = float((w_fe * (effects - theta_fe) ** 2).sum())
    dfree = max(k - 1, 0)
    C = float(w_fe.sum() - (w_fe ** 2).sum() / w_fe.sum()) if k > 1 else 0.0
    tau2 = max(0.0, (Q - dfree) / C) if C > 0 else 0.0
    w_re = 1.0 / (ses ** 2 + tau2)
    pooled = float((w_re * effects).sum() / w_re.sum())
    se = float(np.sqrt(1.0 / w_re.sum()))
    I2 = float(max(0.0, (Q - dfree) / Q * 100.0)) if Q > 0 else 0.0
    return {
        "pooled": pooled,
        "se": se,
        "ci_low": float(pooled - 1.96 * se),
        "ci_high": float(pooled + 1.96 * se),
        "tau2": float(tau2),
        "Q": Q,
        "I2": I2,
        "k": k,
    }
