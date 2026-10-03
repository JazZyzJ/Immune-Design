# TigerData backup and transfer between clusters

Use this workflow on Della, Tiger, Stellar, or another cluster with TigerData
project access. Run jobs on that cluster's scratch filesystem, archive completed
outputs in TigerData, then restore inputs to the destination cluster's scratch.

```text
Source cluster scratch → TigerData archive → Destination cluster scratch
```

## Access and naming

| Location | Purpose |
|---|---|
| Source/destination scratch | Cluster-local inputs, outputs, packaging and extraction |
| Cluster login node | Access `/tigerdata` and copy files using rsync |
| Globus collection | Large or long-running transfers with integrity verification |

SSH into the appropriate login host: `della.princeton.edu`, `tiger.princeton.edu`,
or your assigned host on another cluster. Use Princeton VPN when required.
The same scratch pathname on two clusters does **not** mean shared data.
Tiger compute nodes cannot access `/tigerdata`; do not put these transfer commands
in a Tiger compute-node Slurm job. Other clusters' mounts must also be checked.
[Access guidance](https://tigerdata.princeton.edu/get-started/training-and-resources),
[filesystem guidance](https://researchcomputing.princeton.edu/support/knowledge-base/data-storage).

The existing project and personal archive root are:

```bash
TD_PROJECT='/tigerdata/Jiang_Lab/MLPE'
TD_ROOT="$TD_PROJECT/zijie/Immune-Design"
ls -ld "$TD_PROJECT"
checkquota
```

For another user/project, change those two paths. Project membership is required.
If the mount is unavailable, use the project's Globus collection or contact its
Data Manager; creating a local directory does not establish TigerData access.

New bundles use `archives/<origin-cluster>/<bundle-id>/`. Keep `bundle-id` unique
and immutable, for example `uricase_20261003_v1`. Record the **source** cluster
even when downloading from a different cluster. The older validated pilot under
`run/return_packages/` remains at its existing path.

## 1. Prepare a completed dataset on the source cluster

The examples use Bash. Replace the sample identifiers and absolute scratch paths.
For large datasets, run packaging and local hashing in a CPU Slurm allocation;
write its logs to `logs/`. Use scratch for staging, with enough room for the archive.
Include configs, input identifiers, source/version information and result metadata
with the dataset. Include required targets of any symlinks outside the source folder.

```bash
set -euo pipefail
ORIGIN_CLUSTER='della'  # Use tiger, stellar, etc. when that is the source.
BUNDLE_ID='uricase_20261003_v1'
SOURCE_DIR='/absolute/source-cluster/scratch/completed-run'
STAGE_DIR='/absolute/source-cluster/scratch/transfer'
BUNDLE_DIR="$STAGE_DIR/$BUNDLE_ID"

test -d "$SOURCE_DIR"
mkdir -p "$STAGE_DIR"
mkdir "$BUNDLE_DIR"  # Requires a fresh bundle; prevents replacing an existing one.
tar -C "$SOURCE_DIR" -cf "$BUNDLE_DIR/data.tar" .
printf 'origin_cluster=%s\nsource_directory=%s\nsource_host=%s\ncreated_utc=%s\n' \
  "$ORIGIN_CLUSTER" "$SOURCE_DIR" "$(hostname -f)" "$(date -u +%FT%TZ)" \
  > "$BUNDLE_DIR/SOURCE.txt"
(cd "$BUNDLE_DIR" && sha256sum data.tar SOURCE.txt > SHA256SUMS)
```

Uncompressed tar preserves directory structure and reduces small-file overhead.
Existing `.tar.gz`/`.tar.zst` packages can be used instead of `data.tar`; adjust the
checksum and extraction filenames accordingly. For very large datasets, use one
bundle per completed experiment or shard rather than one enormous archive.

## 2. Upload and verify from the source login node

Set the variables from sections 1 and Access again if this is a new shell.

```bash
set -euo pipefail
test -w "$TD_PROJECT"
ARCHIVE_DIR="$TD_ROOT/archives/$ORIGIN_CLUSTER/$BUNDLE_ID"
mkdir -p "$ARCHIVE_DIR"
rsync -rt --partial --info=progress2 "$BUNDLE_DIR/" "$ARCHIVE_DIR/"
(cd "$ARCHIVE_DIR" && sha256sum -c SHA256SUMS)
```

Both checksum entries must report `OK`. Rerun the same rsync command after an
interruption; incomplete files are retried and matching completed files are skipped.
The trailing `/` copies the bundle's contents into the intended directory.
`-rt` avoids changing TigerData owner/group/permissions. Keep the original source
until destination verification and a restore test succeed. Remove only explicitly
selected source data afterward; backup alone does not free scratch space.

## 3. Download on any destination cluster, verify, and extract

Log into the **destination** cluster and choose its own scratch paths. Keep the
archive's original cluster identifier; do not replace it with the destination name.

```bash
set -euo pipefail
TD_ROOT='/tigerdata/Jiang_Lab/MLPE/zijie/Immune-Design'
ORIGIN_CLUSTER='della'
BUNDLE_ID='uricase_20261003_v1'
DEST_PARENT='/absolute/destination-cluster/scratch/restored'
ARCHIVE_DIR="$TD_ROOT/archives/$ORIGIN_CLUSTER/$BUNDLE_ID"
LOCAL_BUNDLE="$DEST_PARENT/$ORIGIN_CLUSTER/$BUNDLE_ID"
RESTORED_DATA="$LOCAL_BUNDLE/restored"

test -r "$ARCHIVE_DIR/SHA256SUMS"
mkdir -p "$LOCAL_BUNDLE"
rsync -rt --partial --info=progress2 "$ARCHIVE_DIR/" "$LOCAL_BUNDLE/"
(cd "$LOCAL_BUNDLE" && sha256sum -c SHA256SUMS)
mkdir "$RESTORED_DATA"  # Extract into a fresh directory.
tar --no-same-owner -xf "$LOCAL_BUNDLE/data.tar" -C "$RESTORED_DATA"
```

For large bundles, verify the local copy and extract it in a destination CPU
allocation after rsync completes. Submit computation using `RESTORED_DATA`, updating
cluster paths in configs or CLI arguments. Transporting files does not rewrite paths
embedded in manifests or provide the destination's software environment.

## Large transfers: Globus

For transfers likely to take over 15 minutes, use [Globus](https://researchcomputing.princeton.edu/services/data-transfer-networking/data-transfer-globus):

1. Sign in with Princeton credentials and select the source cluster's scratch
   collection and the **actual** bundle path visible in that collection.
2. Select the TigerData project's provisioned collection and the matching archive
   directory. Enable checksum verification and preserve file timestamps.
3. Wait for successful completion. Download by reversing the endpoints, using the
   destination cluster's scratch collection. Verify local `SHA256SUMS` before extraction.

Collection names and visible paths can differ between clusters and system generations;
confirm them in the [official endpoint guide](https://researchcomputing.princeton.edu/services/data-transfer-networking/data-transfer-globus).
TigerData NFS access does not imply a Globus collection is provisioned; the project's
Sponsor/Manager can [request one](https://tigerdata.princeton.edu/tigerdata-request-forms).
For CLI transfers, explicitly use `--verify-checksum`; see the
[Globus reference](https://docs.globus.org/cli/reference/transfer/).

## Operational rules and verification status

- Archive completed, stable bundles. TigerData file updates create versions that
  [consume quota](https://tigerdata.princeton.edu/understanding-your-projects-storage-usage).
- Keep upload and restore copies until checks pass; these examples never delete data.
- Use login nodes for modest copies and Globus for large transfers. Large packaging,
  local checksum work and extraction belong on compute nodes with scratch access.
- Della → TigerData upload and destination read-back were validated on 2026-10-01
  using a 77 MB result package; source and destination SHA-256 matched. Evidence:
  [storage audit](Results/Analysis/storage_audit_20261001/README.md).
  Tiger access follows official documentation and has not been live-tested here.
