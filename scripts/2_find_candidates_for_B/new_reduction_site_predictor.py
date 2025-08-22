#!/usr/bin/env python3
"""
new_reduction_site_predictor.py

Given a molecule in SMILES (or SELFIES, decoded internally), generate
likely reduced candidates B based on functional group reduction rules.

Key features:
- SMARTS → product reaction definitions for typical organic reductions.
- Uses RDKit to apply transformations.
- Recursively reduces products (A → B → C …) up to max_depth.
- Returns unique SMILES candidates or None if no reductions apply.

Dependencies:
  pip install selfies rdkit-pypi
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Set, Tuple
import warnings
warnings.filterwarnings('ignore')

import selfies as sf
from rdkit import Chem
from rdkit.Chem import AllChem


# ---------------------------------------------------------------------------
# Reduction rules (SMARTS → product SMARTS)
# ---------------------------------------------------------------------------
REDUCTION_RULES: Dict[str, Tuple[str, str]] = {

    # ketone: R–C(=O)–R → R–CH(OH)–R
    "ketone_to_alcohol": (
        "[C:1](=[O:2])([C:3])[C:4]",
        "[C:1]([O:2])([C:3])[C:4]"
    ),

    # aldehyde: R–CH=O → R–CH2OH
    "aldehyde_to_alcohol": (
        "[CX3H1:1](=O)[#6:2]",    # carbonyl C with exactly 1 H, neighbor must be C
        "[C:1]([O])[#6:2]"        # reduction → alcohol, keep substituent
    ),

    # formaldehyde special case: O=CH2 → CH3OH
    "formaldehyde_to_methanol": (
        "[CX3H2:1](=O)",
        "[CH3:1][OH]"
    ),

    # carboxylic acid → primary alcohol
     "acid_to_alcohol": (
     "[C:1](=O)[O;H1:2]",    # must be an –OH group
     "[C:1][O:2]"            # reduced → alcohol
    ),

    "ester_to_alcohol": (
    "[C:1](=O)[O:2][C:3]",      # Ester: carbonyl C bonded to O, which is bonded to another C
    "[C:1][O][H].[O:2][C:3]"    # Reduction: carbonyl → alcohol (CH2OH), alkoxy → alcohol (ROH)
),

    # imines → amines
    "imine_to_amine": (
        "[C:1]=[N:2]",
        "[C:1][N:2]"
    ),

   "oxime_to_amine": (
    "[C:1]=[N:2][O:3]",   # C=NOH pattern
    "[C:1]-[N:2][O:3]"          # → C–NH₂
   ),

   "enamine_to_amine": (
    "[C:1]=[C:2]-[N:3]",   # enamine C=C–N
    "[C:1]-[C:2]-[N:3]"    # reduced to saturated amine
    ),

    # iminium → amine
    "iminium_to_amine": (
        "[C:1]=[N+:2]",
        "[C:1][N:2]"
    ),

    # nitriles → primary amine
    "nitrile_to_amine": (
        "[C:1]#[N:2]",
        "[C:1][N:2]"
    ),

    # alkyne → alkene
    "alkyne_to_alkene": (
        "[C:1]#[C:2]",
        "[C:1]=[C:2]"
    ),

    # alkene → alkane
    "alkene_to_alkane": (
        "[C:1]=[C:2]",
        "[C:1][C:2]"
    ),

    # Aromatic ring hydrogenation (benzene → cyclohexane)
    "aromatic_ring_to_cyclohexane": (
        "c1ccccc1",         # six-membered aromatic ring
        "C1CCCCC1"          # cyclohexane
    ),
}

# ---------------------------------------------------------------------------
# Build reaction objects
# ---------------------------------------------------------------------------
REACTIONS: Dict[str, AllChem.ChemicalReaction] = {
    name: AllChem.ReactionFromSmarts(f"{smarts}>>{product}")
    for name, (smarts, product) in REDUCTION_RULES.items()
}

# ---------------------------------------------------------------------------
# Reduction engine
# ---------------------------------------------------------------------------
def get_reduction_candidates(
    smiles: str, max_depth: int = 2
) -> Optional[List[Tuple[str, str, int]]]:
    """
    Given a reactant SMILES, generate reduced candidates via recursive rule application.

    Args:
        smiles: Input SMILES string
        max_depth: Maximum recursion depth

    Returns:
        List of tuples (rule_name, reduced_product_smiles, depth)
        or None if no reductions possible.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    seen: Set[str] = set([Chem.MolToSmiles(mol, canonical=True)])
    results: Set[Tuple[str, str, int]] = set()

    def recurse(m: Chem.Mol, depth: int):
        if depth >= max_depth:
            return
        for name, rxn in REACTIONS.items():
            products = rxn.RunReactants((m,))
            for prod_tuple in products:
                for prod in prod_tuple:
                    try:
                        Chem.SanitizeMol(prod)
                    except Exception:
                        continue
                    s = Chem.MolToSmiles(prod, canonical=True)
                    if s not in seen:
                        seen.add(s)
                        results.add((name, s, depth + 1))  # record depth level
                        recurse(prod, depth + 1)

    recurse(mol, depth=0)
    return list(results) if results else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
	description=(
			"Reduction Site Predictor\n\n"
			"This script takes a molecule (as SMILES or SELFIES) and predicts "
			"possible reduction products based on rule-based transformations. "
			"You can optionally specify a recursion depth (default=2) to apply reductions multiple times.\n"
			" Example Usage:\n"
           		"  python new_reduction_site_predictor.py --smiles \"CC(=O)O\"\n"
            		"  python new_reduction_site_predictor.py --selfies \"[C][C][=O][O]\"\n"
            		"  python new_reduction_site_predictor.py --smiles \"CCC#CCC\" --depth 2\n"
		    ),
			formatter_class=argparse.RawTextHelpFormatter
		    )
    parser.add_argument(
			"--smiles", type=str, 
			help="Input molecule in SMILES format eg. 'CC(=O)O'."
		       )
    parser.add_argument("--selfies", type=str, 
			help="Input molecule in SELFIES format eg. '[C][C][=O][O]'."
			)
    parser.add_argument("--depth", type=int, default=2, 
			help="Maximum recursion depth for reductions (default=2)")
    args = parser.parse_args()

    if args.smiles:
        mol = Chem.MolFromSmiles(args.smiles)
        print(f"Input: {args.smiles}")
    elif args.selfies:
        smi = sf.decoder(args.selfies)
        mol = Chem.MolFromSmiles(smi)
        print(f"Input SELFIES: {args.selfies}\nDecoded SMILES: {smi}")
    else:
        raise ValueError("Must provide either --smiles or --selfies")

    if mol is None:
        raise ValueError("Invalid input molecule")

    results = get_reduction_candidates(Chem.MolToSmiles(mol), max_depth=args.depth)
    if not results:
        print("No reducible sites found.")
    else:
        print("Reduced candidates:")
        for name, smi, depth in results:
            print(f"  depth {depth} | {name}: {smi}")


if __name__ == "__main__":
    main()

