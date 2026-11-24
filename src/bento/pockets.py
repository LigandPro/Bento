"""Binding-site extraction and GLoSA runtime helpers for Bento."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import product, takewhile
from pathlib import Path
from string import ascii_letters, digits
from typing import Any

import numpy as np

AtomKey = tuple[str, int, str, str, str]
Coord3D = tuple[float, float, float]


@dataclass(frozen=True)
class PocketPreparationResult:
    """Outcome of preparing a binding-site PDB file."""

    bs_path: Path
    residue_count: int
    cutoff_used: float
    used_fallback: bool
    reused_existing: bool


def chemical_features_path(pocket_path: Path) -> Path:
    """Return the companion GLoSA chemical-feature PDB path."""
    return pocket_path.with_name(f"{pocket_path.stem}-cf.pdb")


def has_atoms(path: Path) -> bool:
    """Return True if a PDB file exists and contains at least one ATOM/HETATM line."""
    if not path.is_file():
        return False
    with path.open("rb") as handle:
        return any(line.startswith((b"ATOM  ", b"HETATM")) for line in handle)


def clean_pdb_file(path: Path) -> None:
    """Remove ANISOU/TER lines and ensure a single TER before END."""
    lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(("ANISOU", "TER")):
                continue
            lines.append(line)

    cleaned: list[str] = []
    saw_end = False
    for line in lines:
        if line.strip() == "END":
            cleaned.append("TER\n")
            saw_end = True
        cleaned.append(line)
    if not saw_end:
        cleaned.append("TER\n")
        cleaned.append("END\n")

    with path.open("w", encoding="utf-8") as handle:
        handle.writelines(cleaned)


def _pocket_atom_rows(path: Path) -> Iterator[tuple[AtomKey, Coord3D, Any]]:
    """Yield first-model atom coordinates, residue keys, and export metadata."""
    suffix = path.suffix.lower()
    if suffix in {".pdb", ".ent"}:
        first_segments: dict[str, str] = {}
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith(("ENDMDL", "END   ")) or line.strip() == "END":
                    break
                if not line.startswith(("ATOM  ", "HETATM")):
                    continue
                chain = line[21]
                segment = line[72:76].strip()
                if segment != first_segments.setdefault(chain, segment):
                    chain = f"{chain}:{segment}"
                key: AtomKey = (
                    chain,
                    int(line[22:26]),
                    line[26],
                    line[17:20].strip(),
                    line[:6].strip(),
                )
                xyz: Coord3D = (
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                )
                yield key, xyz, line
        return

    if suffix not in {".cif", ".mmcif"}:
        raise ValueError(f"Expected a PDB or mmCIF structure: {path}")

    from Bio.PDB.MMCIF2Dict import MMCIF2Dict

    data = MMCIF2Dict(str(path))
    count = len(data.get("_atom_site.Cartn_x", []))

    def column(name: str, default: str = "") -> list[str]:
        return data.get(f"_atom_site.{name}", [default] * count)

    models = column("pdbx_PDB_model_num", "1")
    if not models:
        return

    rows = zip(
        models,
        column("group_PDB"),
        column("label_comp_id"),
        column("auth_asym_id"),
        column("label_asym_id"),
        column("auth_seq_id"),
        column("label_seq_id"),
        column("pdbx_PDB_ins_code"),
        column("Cartn_x"),
        column("Cartn_y"),
        column("Cartn_z"),
        column("label_atom_id"),
        column("label_alt_id"),
        column("type_symbol"),
        column("occupancy"),
        column("B_iso_or_equiv"),
        strict=False,
    )
    missing = {"", ".", "?"}
    for model, record, name, chain, label_chain, seq, label_seq, ins, x, y, z, *atom in rows:
        if model != models[0]:
            break
        if record not in {"ATOM", "HETATM"}:
            continue
        resolved_chain = label_chain if chain in missing else chain
        resolved_seq = label_seq if seq in missing else seq
        resolved_ins = " " if ins in missing else ins
        yield (
            (resolved_chain, int(resolved_seq), resolved_ins, name, record),
            (float(x), float(y), float(z)),
            atom,
        )


def _save_pocket_rows(pocket: dict[AtomKey, list[tuple[Coord3D, Any]]], output_path: Path) -> None:
    """Build a Biopython structure from selected residues and write PDB/mmCIF."""
    from Bio.PDB import MMCIFIO, PDBIO
    from Bio.PDB.StructureBuilder import StructureBuilder

    use_cif = output_path.suffix.lower() in {".cif", ".mmcif"}
    if use_cif and not pocket:
        output_path.write_text("data_pocket\n#\n", encoding="utf-8")
        return

    chain_ids: dict[str, str | None] = dict.fromkeys(key[0] for key in pocket)
    free_ids = iter(char for char in ascii_letters + digits if char not in chain_ids)
    for chain in chain_ids:
        chain_ids[chain] = chain if use_cif or len(chain) == 1 else next(free_ids, None)
        if chain_ids[chain] is None:
            raise ValueError("No unused single-character PDB chain IDs remain. Save as mmCIF.")

    builder = StructureBuilder()
    builder.init_structure("pocket")
    builder.init_model(0)
    builder.init_seg(" ")
    current_chain: str | None = None

    for (chain, number, ins, resname, record), rows in pocket.items():
        if chain != current_chain:
            chain_id = chain_ids[chain]
            assert chain_id is not None
            builder.init_chain(chain_id)
            current_chain = chain
        field = " " if record == "ATOM" else "W" if resname in {"HOH", "WAT"} else "H"
        builder.init_residue(resname, field, number, ins)

        atoms: list[tuple[Coord3D, str, str, str, str, str, str]] = []
        for xyz, raw in rows:
            if isinstance(raw, str):
                fullname = raw[12:16]
                name = fullname.strip()
                alt = raw[16]
                element = raw[76:78].strip()
                occupancy = raw[54:60].strip()
                bfactor = raw[60:66].strip()
            else:
                name, alt, element, occupancy, bfactor = raw
                fullname = name
            atoms.append((xyz, name, fullname, alt, element, occupancy, bfactor))

        reserved = {atom[1] for atom in atoms}
        generated_names = map("".join, product(digits + ascii_letters, repeat=4))
        free_names = (candidate for candidate in generated_names if candidate not in reserved)
        locations: dict[str, set[str]] = {}
        for xyz, name, fullname, alt, element, occupancy, bfactor in atoms:
            alt_clean = " " if alt in {"", ".", "?"} else alt
            seen = locations.setdefault(name, set())
            if alt_clean in seen or (alt_clean == " " and seen):
                name = fullname = next(free_names)
            else:
                seen.add(alt_clean)
            occ_val = 1.0 if occupancy in {"", ".", "?"} else float(occupancy)
            bfac_val = 0.0 if bfactor in {"", ".", "?"} else float(bfactor)
            builder.init_atom(
                name,
                np.asarray(xyz),
                bfac_val,
                occ_val,
                alt_clean,
                fullname,
                element=element.upper() or None,
            )

    writer = MMCIFIO() if use_cif else PDBIO()
    writer.set_structure(builder.get_structure())
    writer.save(str(output_path))


def _sdf_coordinates(path: Path) -> Any:
    """Read first-record SDF coordinates, falling back to V2000/V3000 text parsing."""
    try:
        from rdkit import Chem
    except ImportError:
        Chem = None  # type: ignore[assignment]

    if Chem is not None:
        with path.open("rb") as handle:
            try:
                molecule = next(
                    Chem.ForwardSDMolSupplier(handle, sanitize=False, removeHs=False),
                    None,
                )
            except (RuntimeError, ValueError):
                molecule = None
        if molecule is not None and molecule.GetNumConformers() and molecule.GetNumAtoms():
            return molecule.GetConformer().GetPositions()

    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            record = takewhile(lambda line: line.strip() != "$$$$", handle)
            for _ in range(3):
                next(record)
            counts = next(record)
            if "V3000" in counts:
                for line in record:
                    if line.strip() == "M  V30 BEGIN ATOM":
                        break
                coordinates: list[Coord3D] = []
                continuation = False
                for line in record:
                    fields = line.split()
                    if fields == ["M", "V30", "END", "ATOM"]:
                        if coordinates:
                            return coordinates
                        break
                    if not continuation:
                        coordinates.append((float(fields[4]), float(fields[5]), float(fields[6])))
                    continuation = line.rstrip().endswith("-")
            else:
                count = int(counts[:3])
                if count > 0:
                    coordinates = []
                    for _ in range(count):
                        line = next(record)
                        coordinates.append(
                            (
                                float(line[0:10]),
                                float(line[10:20]),
                                float(line[20:30]),
                            )
                        )
                    return coordinates
    except (IndexError, StopIteration, ValueError) as exc:
        raise ValueError(f"Cannot read first-record SDF coordinates: {path}") from exc
    raise ValueError(f"Cannot read first-record SDF coordinates: {path}")


def extract_pocket(
    protein_path: str | Path,
    ligand_path: str | Path,
    *,
    renumber: bool = True,
    threshold: float = 4.5,
    save: bool = False,
    output_path: str | Path | None = None,
    protein_only: bool = True,
) -> list[list[Any]]:
    """Extract contacting protein residues within `threshold` angstroms of a ligand.

    Supports PDB/mmCIF proteins and PDB/mmCIF/SDF/MOL/MOL2 ligands in the same
    coordinate frame. Distances include all coordinate records in the first model.
    """
    from Bio.PDB.Polypeptide import is_aa
    from scipy.spatial import cKDTree

    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    if save and output_path is None:
        raise ValueError("output_path is required when save=True")

    protein = Path(protein_path)
    ligand = Path(ligand_path)
    source = list(_pocket_atom_rows(protein))
    if not source:
        raise ValueError(f"No atoms in the first model: {protein}")

    extension = ligand.suffix.lower()
    if extension == ".sdf":
        ligand_xyz = _sdf_coordinates(ligand)
    elif extension in {".mol", ".mol2"}:
        from rdkit import Chem

        reader = Chem.MolFromMolFile if extension == ".mol" else Chem.MolFromMol2File
        molecule = reader(str(ligand), sanitize=False, removeHs=False)
        if molecule is None or not molecule.GetNumConformers() or not molecule.GetNumAtoms():
            raise ValueError(f"No ligand coordinates in {ligand}")
        ligand_xyz = molecule.GetConformer().GetPositions()
    else:
        ligand_xyz = [xyz for _, xyz, _ in _pocket_atom_rows(ligand)]
        if not ligand_xyz:
            raise ValueError(f"No atoms in the first model: {ligand}")

    candidates = [
        (key, xyz)
        for key, xyz, _ in source
        if not protein_only or (key[4] == "ATOM" and is_aa(key[3]))
    ]
    selected: set[AtomKey] = set()
    if candidates:
        keys, coordinates = zip(*candidates, strict=True)
        distances, _ = cKDTree(ligand_xyz).query(coordinates)
        selected = {
            key for key, distance in zip(keys, distances, strict=True) if distance <= threshold
        }

    starts: dict[str, int] = {}
    pocket: dict[AtomKey, list[tuple[Coord3D, Any]]] = {}
    for key, xyz, raw in source:
        chain, number, *_ = key
        starts.setdefault(chain, number)
        if key in selected:
            atoms = pocket.setdefault(key, [])
            if save:
                atoms.append((xyz, raw))

    if save:
        assert output_path is not None
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        _save_pocket_rows(pocket, out)

    return [
        [chain, number - starts[chain] + 1 if renumber else number] for chain, number, *_ in pocket
    ]


def prepare_binding_site(
    protein_path: str | Path,
    ligand_path: str | Path,
    bs_path: str | Path,
    *,
    threshold: float = 4.5,
    fallback_threshold: float | None = None,
    protein_only: bool = True,
    overwrite: bool = False,
) -> PocketPreparationResult:
    """Extract and clean a binding-site PDB with optional fallback cutoff for empty pockets."""
    target = Path(bs_path)
    if not overwrite and has_atoms(target):
        return PocketPreparationResult(
            bs_path=target,
            residue_count=-1,
            cutoff_used=threshold,
            used_fallback=False,
            reused_existing=True,
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        residues = extract_pocket(
            protein_path=protein_path,
            ligand_path=ligand_path,
            threshold=threshold,
            save=True,
            output_path=tmp_path,
            protein_only=protein_only,
        )
        cutoff_used = threshold
        used_fallback = False

        if not residues and fallback_threshold is not None and fallback_threshold > threshold:
            residues = extract_pocket(
                protein_path=protein_path,
                ligand_path=ligand_path,
                threshold=fallback_threshold,
                save=True,
                output_path=tmp_path,
                protein_only=protein_only,
            )
            cutoff_used = fallback_threshold
            used_fallback = bool(residues)

        clean_pdb_file(tmp_path)
        os.replace(tmp_path, target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return PocketPreparationResult(
        bs_path=target,
        residue_count=len(residues),
        cutoff_used=cutoff_used,
        used_fallback=used_fallback,
        reused_existing=False,
    )


def prepare_chemical_features(
    bs_path: str | Path,
    glosa_dir: str | Path,
    *,
    java_bin: str = "java",
    timeout: int = 150,
    overwrite: bool = False,
) -> Path:
    """Generate `<stem>-cf.pdb` via GLoSA AssignChemicalFeatures.

    Upstream GLoSA's AssignChemicalFeatures strips the first occurrence of "pdb"
    from the input path when constructing the `-cf.pdb` output name, which breaks
    filenames whose identifier contains "pdb" (such as `2pdb`, `3pdb`, `7pdb`).
    To stay compatible with both patched and unpatched GLoSA installations, paths
    whose stem contains "pdb" are processed through a temporary neutral filename.
    """
    pocket = Path(bs_path).resolve()
    cf_path = chemical_features_path(pocket)
    if not overwrite and has_atoms(cf_path):
        return cf_path

    glosa_root = Path(glosa_dir).resolve()
    needs_safe_alias = "pdb" in pocket.stem.lower()

    if needs_safe_alias:
        with tempfile.TemporaryDirectory(prefix="bento-glosa-cf-") as tmp_dir:
            tmp_root = Path(tmp_dir)
            alias_pdb = tmp_root / "pocket_input.pdb"
            shutil.copyfile(pocket, alias_pdb)
            subprocess.run(
                [
                    java_bin,
                    "-XX:ActiveProcessorCount=1",
                    "-Xmx256m",
                    "-cp",
                    str(glosa_root),
                    "AssignChemicalFeatures",
                    str(alias_pdb),
                ],
                cwd=glosa_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            alias_cf = tmp_root / "pocket_input-cf.pdb"
            if not alias_cf.is_file():
                raise RuntimeError(
                    f"AssignChemicalFeatures did not produce {alias_cf.name} for {pocket}"
                )
            tmp_target = cf_path.with_name(f".{cf_path.name}.{os.getpid()}.tmp")
            shutil.copyfile(alias_cf, tmp_target)
            os.replace(tmp_target, cf_path)
        return cf_path

    subprocess.run(
        [
            java_bin,
            "-XX:ActiveProcessorCount=1",
            "-Xmx256m",
            "-cp",
            str(glosa_root),
            "AssignChemicalFeatures",
            str(pocket),
        ],
        cwd=glosa_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if not cf_path.is_file():
        raise RuntimeError(f"AssignChemicalFeatures did not produce {cf_path}")
    return cf_path


def build_glosa_binary(
    glosa_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    cxx: str = "g++",
    optimize: bool = True,
) -> Path:
    """Compile GLoSA `glosa.cpp` with deterministic IEEE-754 optimization flags."""
    glosa_root = Path(glosa_dir).resolve()
    source = glosa_root / "glosa.cpp"
    if not source.is_file():
        raise FileNotFoundError(f"glosa.cpp not found in {glosa_root}")

    target = Path(output_path).resolve() if output_path else glosa_root / "glosa"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_bin = target.with_name(f".{target.name}.{os.getpid()}.tmp")

    cmd = [cxx]
    if optimize:
        cmd.extend(["-O3", "-ffp-contract=off"])
    cmd.extend([str(source), "-o", str(tmp_bin)])

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        os.chmod(tmp_bin, 0o755)
        os.replace(tmp_bin, target)
    finally:
        if tmp_bin.exists():
            tmp_bin.unlink()

    return target
