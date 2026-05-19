import csv
import json
import sqlite3
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHEMBL_DIR = PROJECT_ROOT / "DataPrepare/chembl_related"
CHEMBL_DB = CHEMBL_DIR / "chembl_36/chembl_36_sqlite/chembl_36.db"
MAPPING_TSV = CHEMBL_DIR / "processed_data/chembl_tdc_overlap/tdc_unique_molecule_chembl_matches.tsv"
CACHE_DIR = CHEMBL_DIR / "processed_data/chembl_tool_cache"
MATCH_METHOD_PRIORITY = ("chembl_parent", "chembl_standardized", "rdkit")
MAX_ACTIVITIES_TO_SHOW = 25

_MAPPING_BY_SMILES: dict[str, dict[str, str]] | None = None
_CONN: sqlite3.Connection | None = None
_CACHE_BY_MODE: dict[bool, dict[str, str]] = {}


def _tool(name: str, description: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles": {
                        "type": "string",
                        "description": "SMILES string of the molecule.",
                    }
                },
                "required": ["smiles"],
                "additionalProperties": False,
            },
        },
    }


def _load_mapping() -> dict[str, dict[str, str]]:
    global _MAPPING_BY_SMILES
    if _MAPPING_BY_SMILES is None:
        with MAPPING_TSV.open(encoding="utf-8", newline="") as handle:
            _MAPPING_BY_SMILES = {row["smiles"]: row for row in csv.DictReader(handle, delimiter="\t")}
    return _MAPPING_BY_SMILES


def _connect() -> sqlite3.Connection:
    global _CONN
    if _CONN is None:
        _CONN = sqlite3.connect(f"file:{CHEMBL_DB}?mode=ro", uri=True)
        _CONN.row_factory = sqlite3.Row
    return _CONN


def _cache_path(include_activities: bool) -> Path:
    filename = "chembl_info_with_activities.jsonl" if include_activities else "chembl_info_properties_only.jsonl"
    return CACHE_DIR / filename


def _load_cache(include_activities: bool) -> dict[str, str]:
    if include_activities in _CACHE_BY_MODE:
        return _CACHE_BY_MODE[include_activities]

    cache: dict[str, str] = {}
    path = _cache_path(include_activities)
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                smiles = row.get("smiles")
                result = row.get("result")
                if isinstance(smiles, str) and isinstance(result, str):
                    cache[smiles] = result
    _CACHE_BY_MODE[include_activities] = cache
    return cache


def _cached_result(smiles: str, include_activities: bool) -> str | None:
    return _load_cache(include_activities).get(smiles)


def _truthy(value: str | None) -> bool:
    return str(value or "").lower() == "true"


def _first_strict_match(smiles: str) -> tuple[str, str] | None:
    row = _load_mapping().get(smiles)
    if not row:
        return None
    for method in MATCH_METHOD_PRIORITY:
        if not _truthy(row.get(f"{method}_full_match")):
            continue
        chembl_ids = [item for item in row.get(f"{method}_chembl_ids", "").split(";") if item]
        if chembl_ids:
            return method, chembl_ids[0]
    return None


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return "NA"
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def _truncate(text: Any, limit: int = 180) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _rdkit_properties(smiles: str) -> dict[str, Any]:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES; RDKit could not parse it.")
    return {
        "source": "RDKit",
        "canonical_smiles": Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "mw": Descriptors.MolWt(mol),
        "alogp": Crippen.MolLogP(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "hbd": Lipinski.NumHDonors(mol),
        "psa": rdMolDescriptors.CalcTPSA(mol),
        "rtb": Lipinski.NumRotatableBonds(mol),
        "aromatic_rings": Lipinski.NumAromaticRings(mol),
        "heavy_atoms": Descriptors.HeavyAtomCount(mol),
        "qed": QED.qed(mol),
        "formal_charge": Chem.GetFormalCharge(mol),
    }


def _chembl_properties(chembl_id: str) -> sqlite3.Row | None:
    return _connect().execute(
        """
        SELECT
          md.molregno,
          md.chembl_id,
          COALESCE(md.pref_name, '') AS name,
          md.molecule_type,
          md.max_phase,
          cp.full_molformula AS formula,
          cp.mw_freebase AS mw,
          cp.alogp,
          cp.hba,
          cp.hbd,
          cp.psa,
          cp.rtb,
          cp.aromatic_rings,
          cp.heavy_atoms,
          cp.qed_weighted AS qed,
          cs.canonical_smiles
        FROM molecule_dictionary md
        LEFT JOIN compound_properties cp ON cp.molregno = md.molregno
        LEFT JOIN compound_structures cs ON cs.molregno = md.molregno
        WHERE md.chembl_id = ?
        """,
        (chembl_id,),
    ).fetchone()


def _activity_count(molregno: int) -> int:
    return int(
        _connect()
        .execute("SELECT COUNT(*) FROM activities WHERE molregno = ?", (molregno,))
        .fetchone()[0]
    )


def _activities(molregno: int) -> list[sqlite3.Row]:
    return list(
        _connect().execute(
            """
            SELECT
              act.standard_type,
              act.standard_relation AS relation,
              act.standard_value AS value,
              act.standard_units AS units,
              act.pchembl_value,
              act.activity_comment,
              act.data_validity_comment,
              act.standard_text_value,
              COALESCE(td.pref_name, '') AS target_name,
              a.assay_type,
              a.assay_test_type,
              a.assay_organism,
              a.description AS assay_description
            FROM activities act
            JOIN assays a ON a.assay_id = act.assay_id
            LEFT JOIN target_dictionary td ON td.tid = a.tid
            WHERE act.molregno = ?
            ORDER BY act.pchembl_value IS NULL, act.pchembl_value DESC, act.standard_type
            """,
            (molregno,),
        )
    )


def _property_lines(props: dict[str, Any] | sqlite3.Row) -> list[str]:
    get = props.get if isinstance(props, dict) else lambda key, default=None: props[key] if key in props.keys() else default
    lines = []
    lines.append(
        "MW={mw}; LogP={alogp}; HBA={hba}; HBD={hbd}; TPSA={psa}; rotatable_bonds={rtb}".format(
            mw=_fmt(get("mw")),
            alogp=_fmt(get("alogp")),
            hba=_fmt(get("hba")),
            hbd=_fmt(get("hbd")),
            psa=_fmt(get("psa")),
            rtb=_fmt(get("rtb")),
        )
    )
    lines.append(
        "aromatic_rings={aromatic_rings}; heavy_atoms={heavy_atoms}; QED={qed}; formula={formula}".format(
            aromatic_rings=_fmt(get("aromatic_rings")),
            heavy_atoms=_fmt(get("heavy_atoms")),
            qed=_fmt(get("qed")),
            formula=_fmt(get("formula")),
        )
    )
    if get("formal_charge") is not None:
        lines.append(f"formal_charge={_fmt(get('formal_charge'))}")
    if get("canonical_smiles"):
        lines.append(f"canonical_smiles={_truncate(get('canonical_smiles'), 140)}")
    return lines


def _format_activity(row: sqlite3.Row, index: int) -> str:
    value = _fmt(row["value"])
    relation = _fmt(row["relation"])
    units = _fmt(row["units"])
    pchembl = _fmt(row["pchembl_value"])
    fields = [
        f"{index}. {row['standard_type'] or 'activity'} {relation} {value} {units}".strip(),
        f"pChEMBL={pchembl}",
    ]
    if row["target_name"]:
        fields.append(f"target={_truncate(row['target_name'], 80)}")
    if row["assay_test_type"]:
        fields.append(f"test={row['assay_test_type']}")
    if row["assay_organism"]:
        fields.append(f"organism={_truncate(row['assay_organism'], 60)}")
    if row["activity_comment"]:
        fields.append(f"comment={_truncate(row['activity_comment'], 80)}")
    if row["data_validity_comment"]:
        fields.append(f"validity={row['data_validity_comment']}")
    if row["assay_description"]:
        fields.append(f"assay_description={_truncate(row['assay_description'], 220)}")
    return "; ".join(fields)


def chembl_info(smiles: str, *, include_activities: bool = True, use_cache: bool = True) -> str:
    """
    Return concise ChEMBL information for a TDC molecule.

    Strict ChEMBL matches use the precomputed TDC-to-ChEMBL full-InChIKey mapping.
    Activity output is controlled by the hidden include_activities argument; the
    OpenAI tool schema only exposes SMILES. Cached outputs are used when available.
    If activities are enabled and a matched molecule has 25 or more ChEMBL activity
    rows, activities are omitted and only properties are returned. If no strict match
    exists, RDKit properties are computed from the input SMILES as a fallback.
    """
    try:
        if use_cache:
            cached = _cached_result(smiles, include_activities)
            if cached is not None:
                return cached

        match = _first_strict_match(smiles)
        if match is None:
            props = _rdkit_properties(smiles)
            lines = ["No strict ChEMBL full-InChIKey match. Returning RDKit properties only.", "properties:"]
            lines.extend(f"- {line}" for line in _property_lines(props))
            return "\n".join(lines)

        method, chembl_id = match
        props = _chembl_properties(chembl_id)
        if props is None:
            fallback = _rdkit_properties(smiles)
            lines = [
                f"Strict mapping found by {method} to {chembl_id}, but ChEMBL properties were not found.",
                "Returning RDKit properties only.",
                "properties:",
            ]
            lines.extend(f"- {line}" for line in _property_lines(fallback))
            return "\n".join(lines)

        lines = [f"Strict ChEMBL full-InChIKey match via {method}.", "properties:"]
        lines.extend(f"- {line}" for line in _property_lines(props))

        if not include_activities:
            return "\n".join(lines)

        count = _activity_count(int(props["molregno"]))
        if count >= MAX_ACTIVITIES_TO_SHOW:
            lines.append(
                f"activities: omitted because this molecule has {count} ChEMBL activity rows "
                f"(threshold is <{MAX_ACTIVITIES_TO_SHOW})."
            )
            return "\n".join(lines)

        rows = _activities(int(props["molregno"]))
        if not rows:
            lines.append("activities: none found in ChEMBL.")
            return "\n".join(lines)

        lines.append(f"activities ({count} rows):")
        lines.extend(_format_activity(row, idx) for idx, row in enumerate(rows, start=1))
        return "\n".join(lines)
    except Exception as exc:
        return f"Error in chembl_info: {type(exc).__name__}: {exc}"


CHEMBL_INFO_OPENAI_TOOLS = [
    _tool(
        "chembl_info",
        (
            "Given a molecule SMILES, return concise information that can help predict the task label: "
            "molecular properties and, when available, relevant assay activity values with assay descriptions."
        ),
    )
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("smiles")
    parser.add_argument("--no-activities", action="store_true")
    args = parser.parse_args()
    print(chembl_info(args.smiles, include_activities=not args.no_activities))
    print(json.dumps(CHEMBL_INFO_OPENAI_TOOLS, indent=2))
