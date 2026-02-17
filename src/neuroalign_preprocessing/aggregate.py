"""Aggregate parcellated TSV files into per-metric Parquet datasets.

Output layout
-------------
output_dir/
    cat12/
        gm.parquet
        wm.parquet
        ct.parquet
    qsiparc/
        {WORKFLOW}/
            {model}_{param}.parquet
            ...

Each file contains all sessions for that metric in long format, with
``subject_code`` and ``session_id`` as identifier columns.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from .parsers import parse_bids_entities, read_parcellation
from .sessions import load_sessions
from .naming_utils import generate_parquet_filename

logger = logging.getLogger(__name__)

# CAT12 parcellated file path template (relative to cat12_root)
_CAT12_TEMPLATE = (
    "sub-{sub}/ses-{ses}/anat/atlas-{atlas}/"
    "sub-{sub}_ses-{ses}_atlas-{atlas}_space-MNI152NLin2009cAsym_res-01"
    "_tissue-{tissue}{mask_str}_parc.tsv"
)

# QSIParc glob pattern (relative to qsiparc_root)
_QSIPARC_GLOB = (
    "qsirecon-{workflow}/sub-{sub}/ses-{ses}/dwi/atlas-{atlas}/"
    "sub-{sub}_ses-{ses}_atlas-{atlas}_space-*_res-*_model-*_param-*"
    "_mask-{mask}_parc.tsv"
)


class _StreamingParquetWriter:
    """Thin wrapper around pq.ParquetWriter with schema-alignment on first write.

    Subsequent batches are reindexed to the schema established by the first
    write: missing columns are filled with null, extra columns are dropped.
    """

    def __init__(self, path: Path, compression: Optional[str]) -> None:
        self.path = path
        self.compression = compression
        self._writer: Optional[pq.ParquetWriter] = None
        self._schema: Optional[pa.Schema] = None
        self.n_rows = 0

    def write(self, table: pa.Table) -> None:
        if self._writer is None:
            self._schema = table.schema
            self._writer = pq.ParquetWriter(
                str(self.path), self._schema, compression=self.compression
            )
        else:
            # Align incoming table to established schema
            existing_names = set(table.schema.names)
            columns = []
            for field in self._schema:
                if field.name in existing_names:
                    col = table.column(field.name)
                    if col.type != field.type:
                        col = col.cast(field.type, safe=False)
                    columns.append(col)
                else:
                    columns.append(pa.nulls(len(table), type=field.type))
            table = pa.table(
                {field.name: columns[i] for i, field in enumerate(self._schema)}
            )
        self._writer.write_table(table)
        self.n_rows += len(table)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()

    def __enter__(self) -> "_StreamingParquetWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()


def aggregate_cat12(
    cat12_root: Path,
    sessions_csv: Path,
    output_dir: Path,
    atlas: str,
    tissues: tuple[str, ...] = ("GM", "WM", "CT"),
    mask: Optional[str] = "gm",
    maskthr: Optional[int] = None,
    compression: Optional[str] = "snappy",
) -> dict[str, int]:
    """Aggregate CAT12 parcellated TSVs into per-tissue Parquet files.

    Writes one file per tissue to ``output_dir/cat12/{tissue.lower()}.parquet``.
    Uses PyArrow streaming writes so memory usage is O(one session).

    Parameters
    ----------
    cat12_root:
        Root directory of the CAT12 parcellated outputs (the ``cat12/``
        directory that contains ``sub-*/`` subdirectories).
    sessions_csv:
        Path to linked_sessions.csv.
    output_dir:
        Root output directory. A ``cat12/`` subdirectory is created here.
    atlas:
        Atlas name (e.g. ``"Schaefer2018N400n7Tian2020S3"``).
    tissues:
        Tissue labels to load.  Must match the ``tissue-*`` BIDS entity in
        the parcellated filenames (case-sensitive).
    mask:
        Mask label (e.g. ``"gm"``). Pass ``None`` to omit the mask entity.
    compression:
        Parquet compression codec (``"snappy"``, ``"gzip"``, ``"zstd"``, or
        ``None`` for uncompressed).

    Returns
    -------
    dict[str, int]
        Mapping of tissue label -> number of rows written.
    """
    sessions = load_sessions(sessions_csv)
    mask_str = f"_mask-{mask}" if mask else ""
    cat12_out = output_dir / "cat12"
    cat12_out.mkdir(parents=True, exist_ok=True)

    results: dict[str, int] = {}

    for tissue in tissues:
        filename = generate_parquet_filename(
            entities={'atlas': atlas, 'tissue': tissue},
            modality='cat12',
            mask=mask,
            maskthr=maskthr
        )
        out_path = cat12_out / filename
        n_missing = 0

        with _StreamingParquetWriter(out_path, compression) as writer:
            for _, row in tqdm(
                sessions.iterrows(),
                total=len(sessions),
                desc=f"CAT12 {tissue}",
                leave=False,
            ):
                sub, ses = row["subject_code"], row["session_id"]
                tsv_path = cat12_root / _CAT12_TEMPLATE.format(
                    sub=sub, ses=ses, atlas=atlas, tissue=tissue, mask_str=mask_str
                )
                df = read_parcellation(
                    tsv_path,
                    extra_columns={"subject_code": sub, "session_id": ses, "tissue": tissue},
                )
                if df is None:
                    n_missing += 1
                    continue
                writer.write(pa.Table.from_pandas(df, preserve_index=False))

        if writer.n_rows == 0:
            # Nothing written — remove empty file if created
            out_path.unlink(missing_ok=True)
            logger.warning("CAT12 %s: no data found, skipping", tissue)
        else:
            logger.info(
                "CAT12 %s: %d rows written to %s (%d sessions missing)",
                tissue,
                writer.n_rows,
                out_path,
                n_missing,
            )

        results[tissue] = writer.n_rows

    return results


def aggregate_qsiparc(
    qsiparc_root: Path,
    sessions_csv: Path,
    output_dir: Path,
    atlas: str,
    mask: str = "gm",
    maskthr: Optional[int] = None,
    compression: Optional[str] = "snappy",
) -> dict[str, int]:
    """Aggregate QSIParc parcellated TSVs into per-workflow/param Parquet files.

    Auto-discovers all ``qsirecon-*`` workflows under ``qsiparc_root``.
    For each workflow, discovers the set of (model, param) combinations
    present in the first session that has data, then streams all sessions
    into ``output_dir/qsiparc/{WORKFLOW}/{model}_{param}.parquet``.

    Parameters
    ----------
    qsiparc_root:
        Root directory containing ``qsirecon-{workflow}/`` subdirectories.
    sessions_csv:
        Path to linked_sessions.csv.
    output_dir:
        Root output directory. A ``qsiparc/`` subdirectory is created here.
    atlas:
        Atlas name.
    mask:
        Mask label.
    compression:
        Parquet compression codec.

    Returns
    -------
    dict[str, int]
        Mapping of ``"{workflow}/{model}_{param}"`` -> number of rows written.
    """
    sessions = load_sessions(sessions_csv)

    workflows = sorted(
        d.name.split("-", 1)[1]
        for d in qsiparc_root.iterdir()
        if d.is_dir() and d.name.startswith("qsirecon-")
    )
    if not workflows:
        logger.warning("No qsirecon-* workflows found in %s", qsiparc_root)
        return {}

    logger.info("Discovered workflows: %s", workflows)

    qsiparc_out = output_dir / "qsiparc"
    results: dict[str, int] = {}

    for workflow in workflows:
        pattern_template = _QSIPARC_GLOB.format(
            workflow=workflow, sub="{sub}", ses="{ses}", atlas=atlas, mask=mask
        )

        # --- Pass 1: discover (model, param) keys from any available session ---
        combo_keys: list[tuple[str, str]] = []
        for _, row in sessions.iterrows():
            sub, ses = row["subject_code"], row["session_id"]
            pattern = pattern_template.format(sub=sub, ses=ses)
            matches = list(qsiparc_root.glob(pattern))
            if not matches:
                continue
            for m in matches:
                entities = parse_bids_entities(m.name, keys=["model", "param"])
                if "model" in entities and "param" in entities:
                    key = (entities["model"], entities["param"])
                    if key not in combo_keys:
                        combo_keys.append(key)
            break  # one session is enough to discover combos

        if not combo_keys:
            logger.warning("Workflow %s: no (model, param) combinations found", workflow)
            continue

        logger.info("Workflow %s: combos %s", workflow, combo_keys)

        # --- Pass 2: stream all sessions into one file per (model, param) ---
        wf_out = qsiparc_out / workflow
        wf_out.mkdir(parents=True, exist_ok=True)

        writers: dict[tuple[str, str], _StreamingParquetWriter] = {
            combo: _StreamingParquetWriter(
                wf_out / generate_parquet_filename(
                    entities={'atlas': atlas, 'model': combo[0], 'param': combo[1]},
                    modality='qsiparc',
                    mask=mask,
                    maskthr=maskthr
                ), compression
            )
            for combo in combo_keys
        }

        n_missing = 0
        try:
            for _, row in tqdm(
                sessions.iterrows(),
                total=len(sessions),
                desc=f"QSIParc {workflow}",
                leave=False,
            ):
                sub, ses = row["subject_code"], row["session_id"]
                pattern = pattern_template.format(sub=sub, ses=ses)
                matches = list(qsiparc_root.glob(pattern))
                if not matches:
                    n_missing += 1
                    continue
                for tsv_path in matches:
                    entities = parse_bids_entities(tsv_path.name, keys=["model", "param"])
                    model = entities.get("model")
                    param = entities.get("param")
                    if not model or not param:
                        continue
                    combo = (model, param)
                    if combo not in writers:
                        # New combo discovered mid-run — add a writer for it
                        writers[combo] = _StreamingParquetWriter(
                            wf_out / generate_parquet_filename(
                                entities={'atlas': atlas, 'model': model, 'param': param},
                                modality='qsiparc',
                                mask=mask,
                                maskthr=maskthr
                            ), compression
                        )
                    df = read_parcellation(
                        tsv_path,
                        extra_columns={
                            "subject_code": sub,
                            "session_id": ses,
                            "workflow": workflow,
                            "model": model,
                            "param": param,
                        },
                    )
                    if df is None:
                        continue
                    writers[combo].write(pa.Table.from_pandas(df, preserve_index=False))
        finally:
            for combo, writer in writers.items():
                writer.close()
                key = f"{workflow}/{combo[0]}_{combo[1]}"
                if writer.n_rows == 0:
                    (wf_out / f"{combo[0]}_{combo[1]}.parquet").unlink(missing_ok=True)
                    logger.warning("%s: no data written, skipping", key)
                else:
                    logger.info("%s: %d rows written", key, writer.n_rows)
                results[key] = writer.n_rows

        logger.info(
            "Workflow %s: done (%d sessions missing)", workflow, n_missing
        )

    return results
