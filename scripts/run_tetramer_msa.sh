#!/bin/bash
# Login-node MSA driver for the tetramer refold gate (PLAN_TETRAMER_GATE.md §3.1).
#
# Della compute nodes have no internet, but the login node reaches the ColabFold MSA server.
# So MSA search runs HERE (login node); prediction runs later on a compute node via
# scripts/submit_tetramer_predict.slurm. For each pre-MSA input JSON, `protenix msa` queries
# the ColabFold server (~seconds) and writes `<stem>-update-msa.json` next to it (embedding the
# pairedMsaPath/unpairedMsaPath). This driver loops over an input dir, is idempotent (skips
# JSONs whose -update-msa.json already exists), and writes a ready-list of the update JSONs for
# the prediction array.
#
# Usage: bash scripts/run_tetramer_msa.sh <IN_DIR> <MSA_OUT_DIR> [READY_LIST]
#   IN_DIR       dir of pre-MSA *.json (from build_tetramer_input.py)
#   MSA_OUT_DIR  where per-sequence MSA artifacts + logs go
#   READY_LIST   output list of -update-msa.json paths (default: IN_DIR/msa_ready.txt)
set -uo pipefail

IN_DIR="${1:?IN_DIR}"; MSA_OUT="${2:?MSA_OUT_DIR}"; LIST="${3:-$IN_DIR/msa_ready.txt}"
PROTENIX="${PROTENIX_RUN:-/scratch/gpfs/KAIYIJIANG/tools/protenix/bin/protenix-run}"
mkdir -p "$MSA_OUT"; : > "$LIST"

n_ok=0 n_skip=0 n_fail=0
for j in "$IN_DIR"/*.json; do
  case "$j" in *-update-msa.json) continue;; esac
  [ -e "$j" ] || continue
  stem="$(basename "$j" .json)"
  upd="$IN_DIR/${stem}-update-msa.json"
  if [ -f "$upd" ]; then echo "[msa] skip $stem (update json present)"; echo "$upd" >> "$LIST"; n_skip=$((n_skip+1)); continue; fi
  echo "[msa] $stem ..."
  if "$PROTENIX" msa --input "$j" --out_dir "$MSA_OUT/$stem" --msa_server_mode protenix > "$MSA_OUT/${stem}.log" 2>&1 && [ -f "$upd" ]; then
    echo "$upd" >> "$LIST"; n_ok=$((n_ok+1)); echo "[msa]   ok -> $(basename "$upd")"
  else
    n_fail=$((n_fail+1)); echo "[msa]   FAIL (see $MSA_OUT/${stem}.log)"
  fi
done
echo "[msa] done: ok=$n_ok skip=$n_skip fail=$n_fail ; ready-list $(wc -l < "$LIST") entries -> $LIST"
