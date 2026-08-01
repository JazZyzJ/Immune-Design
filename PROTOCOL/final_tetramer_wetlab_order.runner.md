# RUNNER: Final tetramer-aware wet-lab order

Use this runner after refinement, holo prediction, ligand-pocket evaluation,
and two-interface evaluation are complete. The governing scientific hierarchy
is defined by the protocol named in the case file. All run-specific thresholds,
axis components, weights, quotas, and diversity rules come from
`cases/<tag>.yaml`.

## Step 0 - Validate the case

```bash
python scripts/validate_case.py cases/{{tag}}.yaml
```

Proceed only on `ALL-PASS`.

## Step 1 - Freeze and inventory inputs

Confirm the candidate table, predicted-WT row, crystal/reference artifacts,
and all metric sidecars named under `inputs` exist and are non-empty. Record:

- total variants;
- technically valid variants;
- hard-gate survivors;
- unique source lineages;
- counts by experimental arm.

Do not silently substitute a newer table or a different WT reference.

## Step 2 - Apply hard admission layers

Apply the technical, monomer, tetramer-topology, ligand/pocket, confidence, and
other hard rules exactly as written in `criteria.hard_gates`.

A failed hard gate cannot be compensated by a favorable score. Immune score is
not a structural compensator.

## Step 3 - Derive declared ranking axes

For every surviving variant:

1. calculate each declared component from the source column or formula in the
   case;
2. normalize only over the eligible cohort using the declared direction;
3. preserve Interface I and Interface II separately;
4. calculate cross-interface worst values only where declared;
5. emit every component percentile and aggregate score into the output table.

Rosetta-derived values may be used only in the role declared by the case. A
soft auxiliary must not be promoted to a hard gate or an additional Pareto
axis during execution.

## Step 4 - Select source representatives and arm quotas

1. Select one candidate per declared source lineage using the same ranking
   contract used for the final order.
2. Recompute Pareto fronts after source deduplication when the case requires
   it.
3. Select within each fixed experimental arm quota.
4. Enforce the declared pairwise sequence-identity rule across the final set.
5. Stop if a quota cannot be filled without relaxing a gate or diversity rule.

Arm labels determine allocation only. They must not enter a within-arm quality
score unless the case explicitly records that intervention.

## Step 5 - Validate outputs

Verify:

- selected count and every arm quota;
- all hard-gate columns are true;
- unique candidate IDs, source lineages, and sequences;
- FASTA headers and sequences match the all-metrics CSV in order;
- all ranked component values are finite;
- the selection formula recomputes exactly from emitted columns;
- every selected arm rank is consecutive from one to its quota;
- comparison artifacts match any declared superseded or audit set.

Write the selected CSV, FASTA, full scored pool, machine-readable policy,
per-arm summary, comparison table, and README under
`Results/Wetlab/<order>/`.

## Step 6 - Close the case

1. Fill `outputs` with real paths and counts.
2. Set status to `selected` while the order is ready but not yet purchased.
3. Change status to `ordered` only after purchase.
4. Append one line to `cases/index.jsonl`.
5. Do not write `LOG.md` for the screening decision alone.
