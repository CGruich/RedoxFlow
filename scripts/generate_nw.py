#!/usr/bin/env python3
import re
import sys
import shutil
import argparse
from pathlib import Path
from typing import List

from rdkit import Chem
from redox_script_generator import RedoxPotentialScript

REACT_PREFIX = "react_"
PROD_PREFIX  = "prod_"

def ensure_dirs(scripts_root: Path) -> None:
    (scripts_root / "reactants").mkdir(parents=True, exist_ok=True)
    (scripts_root / "products").mkdir(parents=True, exist_ok=True)

def find_xyz_files(src_dirs: List[Path]) -> List[Path]:
    files: List[Path] = []
    for d in src_dirs:
        if not d.exists():
            print(f"[warn] source dir missing: {d}")
            continue
        for p in sorted(d.glob("*.xyz")):
            nm = p.name.lower()
            if nm.startswith(REACT_PREFIX) or nm.startswith(PROD_PREFIX):
                files.append(p)
    return files

def parse_smiles_from_xyz_header(xyz_path: Path) -> str:
    """
    Header looks like:
      REACTANT_ID=2 SMILES=C1CCCCC1 ENERGY=-4.068120 kcal/mol
    Return SMILES string or raise with context.
    """
    try:
        with xyz_path.open("r", encoding="utf-8") as f:
            _ = f.readline()            # atom count
            header = f.readline().strip()
    except Exception as e:
        raise RuntimeError(f"read error: {e}")

    m = re.search(r"\bSMILES=([^\s]+)", header)
    if not m:
        raise ValueError(f"header missing SMILES=...  (header: '{header}')")
    return m.group(1)

def get_charge_and_multiplicity_from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles}")
    charge = Chem.GetFormalCharge(mol)
    unpaired = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    multiplicity = unpaired + 1
    return charge, multiplicity

def write_nwchem_script(
    xyz_src: Path,
    smiles: str,
    is_reactant: bool,
    scripts_root: Path,
) -> None:
    """
    Create <scripts_root>/<reactants|products>/<basename>/ and place:
      - <basename>.xyz (copied)
      - <basename>.nw  (generated)
    Geometry path in .nw is just the local filename.
    scratch_dir and permanent_dir are set to this job folder.
    """
    parent_scripts_dir = scripts_root / ("reactants" if is_reactant else "products")
    parent_scripts_dir.mkdir(parents=True, exist_ok=True)

    base = xyz_src.stem           # e.g., 'react_1' or 'prod_8'
    job_dir = parent_scripts_dir / base
    job_dir.mkdir(parents=True, exist_ok=True)

    # Copy geometry into the job folder so the input references a local file
    geom_name = xyz_src.name
    geom_dst = job_dir / geom_name
    if geom_dst.resolve() != xyz_src.resolve():
        shutil.copy2(xyz_src, geom_dst)

    # Build NWChem input
    charge, mult = get_charge_and_multiplicity_from_smiles(smiles)

    # choose ONE combo:

    '''B3LYP: xc b3lyp
    PBE0: xc pbe0
    PBE96: xc xpbe96 cpbe96
    PW91: xc xperdew91 perdew91
    BHLYP: xc bhlyp
    Becke Half and Half: xc beckehandh
    BP86: xc becke88 perdew86
    BP91: xc becke88 perdew91
    BLYP: xc becke88 lyp'''

    '''(xcoarse||coarse||medium||fine||xfine||huge)'''
    '''6-311g*'''

    xc, grid, basis = "xpbe96 cpbe96","xcoarse", "def2-SV(P)"

    nw = RedoxPotentialScript(
        titleGeomOptimizer=f"{base}_redox",
        scratch_dir=str(job_dir),
        permanent_dir=str(job_dir),
        charge=charge,
        geometry=geom_name,             # geometry lives beside the .nw file
        basis=basis,
        maxIter=300,
        xyz=f"{base}_opt",
        xc=xc,
        mult=mult,
        grid=grid,
        disp="vdw 3",
        taskGeomOptimize="dft optimize",
        titleFreq=f"{base}_FREQ",
        taskFreq="dft freq",
        titleSolv=f"{base}_SOLV",
        dielec=78.4,
        minbem="3",
        ificos="1",
        do_gasphase="False",
        taskSolvEnergy="dft energy",
        inputScriptFolderTemplate=None,  # we write the .nw ourselves
    )

    script_text = str(nw)
    out_path = job_dir / f"{base}.nw"
    out_path.write_text(script_text, encoding="utf-8")
    print(f"[ok] wrote {out_path}  (geom={geom_name}, charge={charge}, mult={mult})")

def main():
    # CLI to configure input/output directories
    ap = argparse.ArgumentParser(description="Generate NWChem scripts for reactant/product XYZs.")
    ap.add_argument(
        "--reactants_in",
        type=str,
        default="/home/cameron/Documents/Github/RedoxFlow/memory/reactants",
        help="Directory containing reactant XYZ files (default: memory/reactants)",
    )
    ap.add_argument(
        "--products_in",
        type=str,
        default="/home/cameron/Documents/Github/RedoxFlow/memory/products",
        help="Directory containing product XYZ files (default: memory/products)",
    )
    ap.add_argument(
        "--scripts_out",
        type=str,
        default="/home/cameron/Documents/Github/RedoxFlow/memory/scripts",
        help="Root directory to write scripts (default: memory/scripts). "
             "Scripts go into <scripts_out>/reactants and <scripts_out>/products.",
    )
    args = ap.parse_args()

    reactants_dir = Path(args.reactants_in)
    products_dir = Path(args.products_in)
    scripts_root  = Path(args.scripts_out)

    ensure_dirs(scripts_root)

    src_dirs = [reactants_dir, products_dir]
    xyz_files = find_xyz_files(src_dirs)
    print(f"[info] found {len(xyz_files)} XYZ files in {', '.join(str(d) for d in src_dirs)}")
    if not xyz_files:
        print(f"[hint] put your 'react_*.xyz' in {reactants_dir} and 'prod_*.xyz' in {products_dir}")
        return

    failures = 0
    for p in xyz_files:
        is_reactant = p.name.lower().startswith(REACT_PREFIX)
        kind = "reactant" if is_reactant else "product" if p.name.lower().startswith(PROD_PREFIX) else "unknown"
        try:
            smiles = parse_smiles_from_xyz_header(p)
            if Chem.MolFromSmiles(smiles) is None:
                raise ValueError(f"RDKit cannot parse SMILES='{smiles}'")
            print(f"[do ] {kind:8s}  {p.name:25s}  SMILES={smiles}")
            write_nwchem_script(p, smiles, is_reactant=is_reactant, scripts_root=scripts_root)
        except Exception as e:
            failures += 1
            print(f"[ERR] {kind:8s}  {p} :: {e}")

    if failures:
        print(f"[done] {len(xyz_files)-failures} ok, {failures} failed")
        sys.exit(1)
    else:
        print(f"[done] {len(xyz_files)} ok, 0 failed")

if __name__ == "__main__":
    main()
