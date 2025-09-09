#!/usr/bin/env python3
import re
import sys
import shutil
from pathlib import Path

from rdkit import Chem
from redox_script_generator import RedoxPotentialScript

# -------- paths --------
ROOT = Path(".").resolve()
SRC_DIRS = [ROOT / "reactants", ROOT / "products"]   # where XYZs live
SCRIPTS_ROOT = ROOT / "scripts"
REACT_SCRIPTS_DIR = SCRIPTS_ROOT / "reactants"
PROD_SCRIPTS_DIR  = SCRIPTS_ROOT / "products"

REACT_PREFIX = "react_"
PROD_PREFIX  = "prod_"
# -----------------------

def ensure_dirs():
    REACT_SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    PROD_SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

def find_xyz_files():
    files = []
    for d in SRC_DIRS:
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

def write_nwchem_script(xyz_src: Path, smiles: str, is_reactant: bool):
    """
    Create scripts/<reactants|products>/<basename>/ and place:
      - <basename>.xyz (copied)
      - <basename>.nw  (generated)
    Geometry path in .nw is just the local filename.
    scratch_dir and permanent_dir are set to this job folder.
    """
    parent_scripts_dir = REACT_SCRIPTS_DIR if is_reactant else PROD_SCRIPTS_DIR
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
    xc, grid = "svwn", "xcoarse"     # ultra-fast
    # xc, grid = "pbe",  "coarse"    # fast GGA
    # xc, grid = "blyp", "coarse"    # alt GGA
    # xc, grid = "b3lyp","coarse"    # lighter than your default

    nw = RedoxPotentialScript(
        titleGeomOptimizer=f"{base}_redox",
        scratch_dir=str(job_dir),       # <-- now the job folder
        permanent_dir=str(job_dir),     # <-- now the job folder
        charge=charge,
        geometry=geom_name,             # geometry lives beside the .nw file
        basis="6-311g*",
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
    ensure_dirs()

    xyz_files = find_xyz_files()
    print(f"[info] found {len(xyz_files)} XYZ files in {', '.join(str(d) for d in SRC_DIRS)}")
    if not xyz_files:
        print("[hint] put your 'react_*.xyz' in ./reactants and 'prod_*.xyz' in ./products")
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
            write_nwchem_script(p, smiles, is_reactant=is_reactant)
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
