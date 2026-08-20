from __future__ import annotations

import csv
from io import BytesIO, StringIO
import json
import re
import zipfile
from typing import Iterable, List

from ocp_app.core.structure_ops import atoms_to_cif_string

from .models import EngineeredStructure


def candidate_summary_records(candidates: Iterable[EngineeredStructure]) -> List[dict]:
    return [c.summary_record() for c in candidates]


def _safe_name(value: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return out or "candidate"


def export_engineered_candidates_zip(candidates: Iterable[EngineeredStructure]) -> bytes:
    items = list(candidates)
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        records = candidate_summary_records(items)
        csv_buf = StringIO()
        fieldnames = sorted({k for row in records for k in row.keys()}) if records else ["candidate_id"]
        writer = csv.DictWriter(csv_buf, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)
        zf.writestr("candidate_summary.csv", csv_buf.getvalue())

        for cand in items:
            folder = _safe_name(cand.candidate_id)
            zf.writestr(f"{folder}/structure.cif", atoms_to_cif_string(cand.atoms))
            meta = {
                "candidate_id": cand.candidate_id,
                "label": cand.label,
                "recipe": cand.recipe.as_dict(),
                "validation": cand.validation,
                "atom_provenance": cand.atom_provenance,
            }
            zf.writestr(
                f"{folder}/metadata.json",
                json.dumps(meta, indent=2, sort_keys=True, default=str),
            )
    return buf.getvalue()
