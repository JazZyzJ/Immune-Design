#!/bin/bash
# =============================================================================
# _migrate_tag_unify.sh — one-shot DATA migration for the allele-tag unification.
#
#   RUN ONLY WHEN NO JOBS ARE USING h_maps/pdbs.
#
# Renames on-disk h-map files and pdbs_if_ready allele directories from the
# legacy short/stripped tags to the unified file-safe tag `HLA-DRB1_07_01`
# (and 04_01 / 15_01), then drops a back-compat SYMLINK old -> new at every
# renamed path so any in-flight job or stale reference still resolves.
#
# Mapping:
#   h_maps_DRB1_<X>.parquet|.meta.json          -> h_maps_HLA-DRB1_<X>.*
#   h_maps_cath_DRB1_<X>.parquet|.meta.json     -> h_maps_cath_HLA-DRB1_<X>.*
#   h_maps_uricase_DRB1_<X>.parquet|.meta.json  -> h_maps_uricase_HLA-DRB1_<X>.*
#   pdbs_if_ready/0701                           -> pdbs_if_ready/HLA-DRB1_07_01
#   pdbs_if_ready/0401                           -> pdbs_if_ready/HLA-DRB1_04_01
#   pdbs_if_ready/1501                           -> pdbs_if_ready/HLA-DRB1_15_01
#
# Idempotent + guarded: skips a pair when the source is missing, never clobbers
# an existing real destination, and treats an already-migrated (symlinked) path
# as done. Safe to re-run. Cluster base is parameterized via ${BASE}.
#
# DRY-RUN by default: prints the actions it WOULD take. Pass APPLY=1 to execute.
#   BASE=/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design APPLY=1 bash scripts/_migrate_tag_unify.sh
# =============================================================================
set -euo pipefail

BASE="${BASE:-/scratch/gpfs/KAIYIJIANG/zijie/work/immune-design}"
APPLY="${APPLY:-0}"

if [[ "${APPLY}" == "1" ]]; then
    MODE_LABEL="APPLY"
else
    MODE_LABEL="DRY-RUN (set APPLY=1 to execute)"
fi

echo "=============================================================="
echo " allele-tag unification data migration"
echo "   BASE : ${BASE}"
echo "   MODE : ${MODE_LABEL}"
echo "=============================================================="

run() {
    # Echo the command; execute only when APPLY=1.
    echo "    + $*"
    if [[ "${APPLY}" == "1" ]]; then
        "$@"
    fi
}

# migrate <old_path> <new_path>
# Renames old -> new (file or directory) and leaves a back-compat symlink
# old -> new. The symlink target is the NEW basename so the link is relative
# and stays valid regardless of mount point.
migrate() {
    local old="$1" new="$2"
    local new_base
    new_base="$(basename "${new}")"

    if [[ -L "${old}" ]]; then
        # Already a symlink → migration was already applied here.
        echo "  [skip] already migrated (symlink): ${old}"
        return 0
    fi

    if [[ -e "${old}" && ! -e "${new}" ]]; then
        # Normal case: real source present, destination free.
        echo "  [move] ${old}"
        echo "      -> ${new}"
        run mv "${old}" "${new}"
        run ln -s "${new_base}" "${old}"
        return 0
    fi

    if [[ ! -e "${old}" && -e "${new}" ]]; then
        # Already renamed but no back-compat link yet → add the link only.
        echo "  [link] back-compat symlink for existing ${new}"
        run ln -s "${new_base}" "${old}"
        return 0
    fi

    if [[ -e "${old}" && -e "${new}" ]]; then
        # Both real exist → do NOT clobber; require manual resolution.
        echo "  [WARN] both source and destination exist, skipping to avoid clobber:"
        echo "         old=${old}"
        echo "         new=${new}"
        return 0
    fi

    # Neither exists.
    echo "  [skip] source missing: ${old}"
    return 0
}

# ── h-map files (test set, CATH, uricase) ────────────────────────────────────
HMAP_TEST_DIR="${BASE}/if_test_set/if_ready/h_maps_v2"
HMAP_CATH_DIR="${BASE}/cath_4.3/h_maps"
HMAP_URICASE_DIR="${BASE}/uricases/h_maps"

echo ""
echo "-- h-map files --"
for X in 07_01 04_01 15_01; do
    for EXT in parquet meta.json; do
        migrate "${HMAP_TEST_DIR}/h_maps_DRB1_${X}.${EXT}" \
                "${HMAP_TEST_DIR}/h_maps_HLA-DRB1_${X}.${EXT}"
    done
done
for X in 07_01 04_01; do
    for EXT in parquet meta.json; do
        migrate "${HMAP_CATH_DIR}/h_maps_cath_DRB1_${X}.${EXT}" \
                "${HMAP_CATH_DIR}/h_maps_cath_HLA-DRB1_${X}.${EXT}"
        migrate "${HMAP_URICASE_DIR}/h_maps_uricase_DRB1_${X}.${EXT}" \
                "${HMAP_URICASE_DIR}/h_maps_uricase_HLA-DRB1_${X}.${EXT}"
    done
done

# ── pdbs_if_ready allele directories ────────────────────────────────────────
PDB_DIR="${BASE}/if_test_set/pdbs_if_ready"

echo ""
echo "-- pdbs_if_ready allele directories --"
migrate "${PDB_DIR}/0701" "${PDB_DIR}/HLA-DRB1_07_01"
migrate "${PDB_DIR}/0401" "${PDB_DIR}/HLA-DRB1_04_01"
migrate "${PDB_DIR}/1501" "${PDB_DIR}/HLA-DRB1_15_01"

echo ""
echo "Done (${MODE_LABEL})."
