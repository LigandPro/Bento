"""Map ligand annotations to test datasets."""

from __future__ import annotations

import argparse
import ast
import glob
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

tqdm.pandas()

PROTEIN_3LETTER_CODES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}

LIGAND_GROUPS = {
    "aa": ["alpha_amino_acids", "peptide_like", "long_peptide"],
    "ster": ["steroids"],
    "nt": [
        "pyrimidine-nucleotide",
        "purine-nucleotide",
        "pyrimidine-nucleozide",
        "purine-nucleozide",
    ],
    "sac": [
        "aldose pyranose",
        "ketose pyranose",
        "pentose pyranose",
        "ketose furanose",
        "aldose furanose",
        "pentose furanose",
        "desoxy-pentose furanose",
    ],
    "macro": ["cycles with >7 members"],
    "eo": ["at least 3 carbons + metal", "element_organcs"],
    "lip": [
        "fatty acids/esters (>8 carbons chain)",
        "triglyceride (ester)",
        "triglyceride (ether)",
        "phospholipid",
        "lipide-like",
    ],
    "cof": [
        "hem-like",
        "biotin-like",
        "B6-like",
        "flavin-like",
        "FMN-like",
        "nicotin-like",
        "quinone-like",
        "glutathione-like",
    ],
    "spiro": ["spiro"],
    "fused": ["condensed_system"],
}


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _get_annotation_ids(df: pd.DataFrame, category: str) -> set[str]:
    columns = [col for col in LIGAND_GROUPS[category] if col in df.columns]
    if not columns:
        return set()
    selected = df[df[columns].fillna(0).gt(0).any(axis=1)]
    return set(selected.index.astype(str))


def _get_unique_types(ligand_types: tuple[tuple[str, ...], ...]) -> tuple[str, ...] | str:
    counter = Counter(ligand_types)
    if len(counter) == 1:
        return next(iter(counter))
    most_common = counter.most_common()
    if most_common[0][1] > sum(value for _, value in most_common[1:]):
        return most_common[0][0]
    merged = tuple(sorted(set(sum(ligand_types, ()))))
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Map ligand annotations to tests table.")
    parser.add_argument("--tests-file", default="datasets/tests.tsv", help="Input tests TSV file.")
    parser.add_argument(
        "--annotations-dir",
        default="annotations",
        help="Annotations root directory.",
    )
    parser.add_argument(
        "--output-tests-file",
        default="datasets/tests_annotated.tsv",
        help="Output path for the annotated tests table.",
    )
    parser.add_argument(
        "--output-tests-exploded-file",
        default="datasets/tests_exploded_annotated.tsv",
        help="Output path for the exploded annotated tests table.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run mapping without writing output files.",
    )
    args = parser.parse_args()

    annotations_dir = Path(args.annotations_dir)
    ligand_classes_dir = annotations_dir / "ligand_classes"

    print(f"Reading tests table: {args.tests_file}")
    tests = pd.read_table(args.tests_file)
    tests["ligand"] = tests["ligand"].map(ast.literal_eval).map(tuple)

    print(f"Loading physicochemical annotation maps from: {annotations_dir}")
    for annotation_path in sorted(glob.glob(str(annotations_dir / "*.json"))):
        annotation_file = Path(annotation_path)
        data = _load_json(annotation_file)
        column_name = annotation_file.stem
        tests[column_name] = tests["uid"].map(data)

    saccharide_like = _load_json(ligand_classes_dir / "saccharide_like.json")
    cofactors = _load_json(ligand_classes_dir / "cofactors.json")
    modres_aa = _load_json(ligand_classes_dir / "modres_aa.json")
    ligands_data = pd.read_table(ligand_classes_dir / "ligands_data.tsv")
    test_annotation = pd.read_csv(ligand_classes_dir / "test_ligand_classes.csv").set_index("uid")

    annotation_sets = {
        category: _get_annotation_ids(test_annotation, category) for category in LIGAND_GROUPS
    }

    cofactor_ids = set(cofactors) | set(
        ligands_data[
            ligands_data["name"].str.contains(
                "cofactor|ubiquinone|PORPHYRIN|PHEOPHYTIN|CHLOROPHYLL|nicotinamide.*nucleotide|flavin.*nucleotide",
                case=False,
                na=False,
            )
        ]["ligand_id"]
    )
    aa_ids = (
        set(modres_aa)
        | {"ACE", "NH2"}
        | set(ligands_data[ligands_data["type"] == "PEPTIDE-LIKE"]["ligand_id"])
        | PROTEIN_3LETTER_CODES
    )
    nt_ids = set(ligands_data[ligands_data["pdbx_type"] == "nucleic acid"]["ligand_id"]) | set(
        ligands_data[
            ligands_data["name"].str.contains(
                "uridin.*phosphate|cytidin.*phosphate|adenin.*phosphate|thymidin.*phosphate|guanosin.*phosphate",
                case=False,
                na=False,
            )
        ]["ligand_id"]
    )
    saccharide_name_filter = ligands_data["name"].str.contains(
        "L-.*ose$|D-.*ose$", case=False, na=False
    )
    sac_ids = set(
        ligands_data[saccharide_name_filter & (ligands_data["mw"] < 300)]["ligand_id"]
    ) | set(saccharide_like)

    def annotate_ligands(ligand_id: str, uid: str) -> tuple[str, ...]:
        labels: list[str] = []
        if ligand_id in cofactor_ids:
            labels.append("cof")
        if ligand_id in aa_ids or uid in annotation_sets["aa"]:
            labels.append("aa")
        if ligand_id in nt_ids or uid in annotation_sets["nt"]:
            labels.append("nt")
        if ligand_id in sac_ids:
            labels.append("sac")
        if uid in annotation_sets["ster"]:
            labels.append("ster")
        if uid in annotation_sets["lip"]:
            labels.append("lip")
        if uid in annotation_sets["macro"]:
            labels.append("macro")
        if uid in annotation_sets["eo"]:
            labels.append("eo")
        if not labels:
            labels.append("other")
        return tuple(labels)

    print("Building ligand type annotations...")
    tests["ligand_types"] = tests.progress_apply(
        lambda row: tuple(annotate_ligands(ligand_id, row["uid"]) for ligand_id in row["ligand"]),
        axis=1,
    )
    tests["ligand_types_unique"] = tests["ligand_types"].progress_apply(_get_unique_types)

    print("Creating exploded ligand type table...")
    tests_exploded = tests.explode("ligand_types_unique", ignore_index=True)
    tests_exploded["ligand_types_unique_etc"] = tests_exploded["ligand_types_unique"].map(
        lambda value: "etc" if value in {"lip", "ster", "eo"} else value
    )

    if args.no_save:
        print("Mapping completed without writing outputs (--no-save).")
        return

    print(f"Writing annotated table: {args.output_tests_file}")
    tests.to_csv(args.output_tests_file, sep="\t", index=False)
    print(f"Writing exploded annotated table: {args.output_tests_exploded_file}")
    tests_exploded.to_csv(args.output_tests_exploded_file, sep="\t", index=False)
    print("Done.")


if __name__ == "__main__":
    main()
