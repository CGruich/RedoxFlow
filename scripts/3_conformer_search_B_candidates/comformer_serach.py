#!/usr/bin/env python3
"""
conformer_search.py

Contains two main utilities:
 1) get_reduction_candidates(smiles, max_depth=2)
    - Rule-based reduction candidate generator using SMARTS -> product rules
    - Uses RDKit ChemicalReaction objects

 2) exhaustive_conformer_search(smiles_list, num_conformers=500, mmff_max_iter=2000)
    - For each SMILES in smiles_list, performs an conformer
      search using RDKit's EmbedMultipleConfs and MMFF optimization of each conformer.
    - Returns a list of (smiles, xyz_string) tuples where xyz_string is the lowest-energy
      conformer's XYZ block. If a given molecule fails, the tuple is (smiles, None).

Notes / Dependencies:
  - rdkit (rdkit-pypi)
  - selfies (only used if you want to accept SELFIES upstream; not required here)

"""

from typing import Dict, List, Optional, Set, Tuple
import logging

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdmolfiles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reduction")

# ---------------------------
# Reduction rules (SMARTS -> product)
# ---------------------------
REDUCTION_RULES: Dict[str, Tuple[str, str]] = {
    # ketone: R–C(=O)–R -> R–CH(OH)–R
    "ketone_to_alcohol": (
        "[C:1](=[O:2])([C:3])[C:4]",
        "[C:1]([O:2])([C:3])[C:4]"
    ),

    # aldehyde: R–CH=O -> R–CH2OH
    "aldehyde_to_alcohol": (
        "[CX3H1:1](=O)[#6:2]",
        "[C:1]([O])[#6:2]"
    ),

    # formaldehyde special: O=CH2 -> CH3OH
    "formaldehyde_to_methanol": (
        "[CX3H2:1](=O)",
        "[CH3:1][OH]"
    ),

    # carboxylic acid -> primary alcohol
    "acid_to_alcohol": (
        "[C:1](=O)[O;H1:2]",
        "[C:1][O:2]"
    ),

    "ester_to_alcohol": (
        "[C:1](=O)[O:2][C:3]",
        "[C:1][O][H].[O:2][C:3]"
    ),

    # imines -> amines
    "imine_to_amine": (
        "[C:1]=[N:2]",
        "[C:1][N:2]"
    ),

    "oxime_to_amine": (
        "[C:1]=[N:2][O:3]",
        "[C:1]-[N:2][O:3]"
    ),

    "enamine_to_amine": (
        "[C:1]=[C:2]-[N:3]",
        "[C:1]-[C:2]-[N:3]"
    ),

    "iminium_to_amine": (
        "[C:1]=[N+:2]",
        "[C:1][N:2]"
    ),

    "nitrile_to_amine": (
        "[C:1]#[N:2]",
        "[C:1][N:2]"
    ),

    "alkyne_to_alkene": (
        "[C:1]#[C:2]",
        "[C:1]=[C:2]"
    ),

    "alkene_to_alkane": (
        "[C:1]=[C:2]",
        "[C:1][C:2]"
    ),

    # aromatic -> saturated
    "aromatic_ring_to_cyclohexane": (
        "c1ccccc1",
        "C1CCCCC1"
    ),
}

# Pre-build RDKit ChemicalReaction objects. If a rule fails to parse, it will be skipped.
REACTIONS: Dict[str, AllChem.ChemicalReaction] = {}
for name, (react, prod) in REDUCTION_RULES.items():
    try:
        REACTIONS[name] = AllChem.ReactionFromSmarts(f"{react}>>{prod}")
    except Exception as e:
        logger.warning("Failed to build reaction '%s': %s", name, str(e))


# ---------------------------
# Reduction candidate generator
# ---------------------------

def get_reduction_candidates(smiles: str, max_depth: int = 2) -> Optional[List[str]]:
    """
    Given an input SMILES, returns a deduplicated list of possible reduced-product SMILES
    discovered by recursively applying the SMARTS-based reduction rules up to max_depth.

    Returns None if the initial SMILES cannot be parsed or no reductions are found.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.error("Invalid input SMILES: %s", smiles)
        return None

    # Use canonical smiles for seen set
    seen: Set[str] = set([Chem.MolToSmiles(mol, canonical=True)])
    results: Set[str] = set()

    def recurse(m: Chem.Mol, depth: int):
        if depth >= max_depth:
            return
        for name, rxn in REACTIONS.items():
            if rxn is None:
                continue
            try:
                products = rxn.RunReactants((m,))
            except Exception:
                continue
            for prod_tuple in products:
                for prod in prod_tuple:
                    try:
                        Chem.SanitizeMol(prod)
                    except Exception:
                        # skip invalid products
                        continue
                    s = Chem.MolToSmiles(prod, canonical=True)
                    if s not in seen:
                        seen.add(s)
                        results.add(s)
                        recurse(prod, depth + 1)

    recurse(mol, depth=0)
    return sorted(results) if results else None


# ---------------------------
# Exhaustive conformer search using RDKit
# ---------------------------

def exhaustive_conformer_search(
    smiles_list: List[str],
    num_conformers: int = 500,
    mmff_max_iter: int = 2000,
    prune_rms_thresh: float = 0.5,
) -> List[Tuple[str, Optional[str]]]:
    """
    For each SMILES in smiles_list, generate up to `num_conformers` 3D conformers using
    RDKit's EmbedMultipleConfs (ETKDG), optimize each conformer with MMFF, and return the
    lowest-energy conformer's XYZ block.

    - If a molecule fails at any point, the returned tuple will have None for the XYZ.

    Returns: [(smiles, xyz_or_none), ...]
    """
    out: List[Tuple[str, Optional[str]]] = []

    for smi in smiles_list:
        logger.info("Processing: %s", smi)
        try:
            mol = Chem.AddHs(Chem.MolFromSmiles(smi))
            if mol is None:
                raise ValueError("Invalid SMILES")

            params = AllChem.ETKDGv3()
            params.pruneRmsThresh = prune_rms_thresh  
            params.randomSeed = 0xF00D

            conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params)
            if not conf_ids:
                logger.warning("No conformers generated for %s", smi)
                out.append((smi, None))
                continue

            mmff_props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94')
            optimize_results = AllChem.MMFFOptimizeMoleculeConfs(
                mol,
                numThreads=0,
                maxIters=mmff_max_iter,
                mmffVariant='MMFF94'
            )

            energies = [res[1] for res in optimize_results]
            if not energies:
                logger.warning("MMFF optimization failed for %s", smi)
                out.append((smi, None))
                continue

            best_idx = int(min(range(len(energies)), key=lambda i: energies[i]))
            best_conf_id = conf_ids[best_idx]
            logger.info("Best energy for %s: %.4f (conf %d)", smi, energies[best_idx], best_conf_id)

            try:
                xyz_block = rdmolfiles.MolToXYZBlock(mol, confId=best_conf_id)
                out.append((smi, xyz_block))
            except Exception as e:
                logger.exception("Failed to convert to XYZ for %s: %s", smi, str(e))
                out.append((smi, None))

        except Exception as e:
            logger.exception("Failed processing %s: %s", smi, str(e))
            out.append((smi, None))

    return out


# ---------------------------
# Simple demo when run as script
# ---------------------------
if __name__ == "__main__":
    demo_smiles = [
        "CC(=O)C",      # isopropyl ketone -> alcohol
        "CC(=O)O",      # acetic acid -> alcohol
        "CC#N",         # nitrile -> amine
        "c1ccccc1"      # benzene -> cyclohexane
    ]

    # 1) Generate reduction candidates for acetone-like molecule
    test = "CC(=O)C"
    red = get_reduction_candidates(test, max_depth=2)
    print("Reductions for", test, "->", red)

    # 2) Exhaustive conformer search on reductions (if any)
    if red:
        conformer_results = exhaustive_conformer_search(red[:10], num_conformers=200)
        for smi, xyz in conformer_results:
            print(smi, "->", "XYZ found" if xyz else "FAILED")
    else:
        print("No reduction candidates found for demo molecule.")
