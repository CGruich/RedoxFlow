#!/usr/bin/env python3
"""
run_reactions_recursive.py

Combine:
  (1) Batch SMARTS reaction runner that loads reactions from JSON and molecules from CSV,
  (2) Recursive reduction/transform runner that re-applies the same reactions to products,
  (3) After products are written, automatically prune reactants_memory.csv
      using products_memory.csv (if present).

INPUT (STRICT for reaction-running):
  CSV file with two required columns:
    - 'index'  (unique identifier per molecule; must be unique)
    - 'smiles' (reactant SMILES)

Example:
  python run_reactions_recursive.py \
      --molecules molecules.csv \
      --reactions reactions.json \
      --out products_memory.csv \
      --depth 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
import numpy as np

# ---------------------------------------------------------------------------
# Silence RDKit warnings
# ---------------------------------------------------------------------------
RDLogger.DisableLog("rdApp.warning")

# ---------------------------------------------------------------------------
# Reaction loading
# ---------------------------------------------------------------------------
def _compile_reaction_from_entry(name: str, entry: dict):
    try:
        if "smarts" in entry and isinstance(entry["smarts"], str):
            rxn_smarts = entry["smarts"]
        elif "reactant" in entry and "product" in entry:
            rxn_smarts = f"{entry['reactant']}>>{entry['product']}"
        else:
            return name, None
        rxn = AllChem.ReactionFromSmarts(rxn_smarts)
        return name, rxn
    except Exception:
        return name, None


def load_reactions(path: str) -> Dict[str, AllChem.ChemicalReaction]:
    with open(path) as f:
        raw = json.load(f)

    compiled: Dict[str, AllChem.ChemicalReaction] = {}

    if isinstance(raw, dict):
        for name, entry in raw.items():
            _, rxn = _compile_reaction_from_entry(name, entry if isinstance(entry, dict) else {"smarts": entry})
            if rxn is not None:
                compiled[name] = rxn
    elif isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name") or entry.get("id") or entry.get("rule") or "unnamed"
            _, rxn = _compile_reaction_from_entry(name, entry)
            if rxn is not None:
                compiled[name] = rxn
    else:
        raise ValueError("Unsupported reactions.json format")
    return compiled

# ---------------------------------------------------------------------------
# Molecules CSV
# ---------------------------------------------------------------------------
def load_molecules_csv(path: str) -> List[Tuple[str, str]]:
    if not path.lower().endswith(".csv"):
        raise ValueError(f"Input molecules must be a .csv file with 'index' and 'smiles' columns. Got: {path}")

    df = pd.read_csv(path, dtype={"index": "string", "smiles": "string"})
    missing = [c for c in ("index", "smiles") if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required column(s): {missing}. Expected columns: 'index', 'smiles'.")

    df["index"] = df["index"].astype("string").str.strip()
    df["smiles"] = df["smiles"].astype("string").str.strip()
    df = df.dropna(subset=["index", "smiles"])
    df = df[(df["index"] != "") & (df["smiles"] != "")]

    if not df["index"].is_unique:
        dupes = df[df["index"].duplicated(keep=False)]["index"].tolist()
        raise ValueError(
            "Column 'index' must be unique. "
            f"Found {len(dupes)} duplicate index values; examples: {dupes[:10]!r} ..."
        )

    return [(str(idx), str(smi)) for idx, smi in zip(df["index"], df["smiles"])]

# ---------------------------------------------------------------------------
# One-step application
# ---------------------------------------------------------------------------
def apply_reactions_once(
    mol: Chem.Mol,
    reactions: Dict[str, AllChem.ChemicalReaction],
    parent_smiles: str,
) -> List[Tuple[str, str]]:
    results: List[Tuple[str, str]] = []
    seen_here: Set[str] = set()

    for rname, rxn in reactions.items():
        if rxn is None:
            continue
        try:
            prods = rxn.RunReactants((mol,))
        except Exception:
            continue

        for prod_tuple in prods:
            for prod in prod_tuple:
                try:
                    Chem.SanitizeMol(prod)
                except Exception:
                    continue
                smi = Chem.MolToSmiles(prod, canonical=True)
                if smi == parent_smiles:
                    continue
                if smi in seen_here:
                    continue
                seen_here.add(smi)
                results.append((rname, smi))
    return results

# ---------------------------------------------------------------------------
# Recursive BFS per root
# ---------------------------------------------------------------------------
def run_recursive_for_root(
    root_name: str,
    root_smiles: str,
    reactions: Dict[str, AllChem.ChemicalReaction],
    max_depth: int,
) -> List[dict]:
    root_mol = Chem.MolFromSmiles(root_smiles)
    if root_mol is None:
        return []

    seen_global: Set[str] = {Chem.MolToSmiles(root_mol, canonical=True)}
    queue: List[Tuple[Chem.Mol, str, int]] = [(root_mol, root_smiles, 0)]
    rows: List[dict] = []

    while queue:
        parent_mol, parent_smi, depth = queue.pop(0)
        if depth >= max_depth:
            continue

        for rname, child_smi in apply_reactions_once(parent_mol, reactions, parent_smi):
            if child_smi in seen_global:
                continue
            seen_global.add(child_smi)

            rows.append(
                {
                    "index": root_name,
                    "Reactant_SMILES": root_smiles,
                    "Depth": depth + 1,
                    "Parent_SMILES": parent_smi,
                    "Reaction_name": rname,
                    "Reaction_scheme": f"{parent_smi} >> {child_smi}",
                    "Product_SMILES": child_smi,
                }
            )

            if depth + 1 < max_depth:
                child_mol = Chem.MolFromSmiles(child_smi)
                if child_mol is not None:
                    queue.append((child_mol, child_smi, depth + 1))
    return rows

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=(
            "Run SMARTS reactions on molecules from a STRICT CSV (columns: 'index', 'smiles'), "
            "then recursively apply the same reactions to newly formed products up to a specified depth. "
            "After writing products, prune reactants_memory.csv using products_memory.csv if available."
        )
    )
    ap.add_argument("--molecules", type=str, required=True,
                    help="Input molecules .csv (REQUIRED) with columns: 'index' (unique), 'smiles'.")
    ap.add_argument("--reactions", type=str, default="reactions.json",
                    help="JSON file with reactions (supports 'smarts' or 'reactant'+'product').")
    ap.add_argument("--out", type=str, default="reaction_products.csv",
                    help="Output CSV path for products (e.g., products_memory.csv).")
    ap.add_argument("--depth", type=int, default=1,
                    help="Recursion depth (1 = only direct products; 2+ = recursive).")
    args = ap.parse_args()

    # Run reactions
    reactions = load_reactions(args.reactions)
    if not reactions:
        raise ValueError(f"No valid reactions compiled from {args.reactions}")

    molecules = load_molecules_csv(args.molecules)
    if not molecules:
        raise ValueError(f"No molecules found in {args.molecules}")

    all_rows: List[dict] = []
    for compound_index, reactant_smiles in molecules:
        if Chem.MolFromSmiles(reactant_smiles) is None:
            continue
        all_rows.extend(
            run_recursive_for_root(compound_index, reactant_smiles, reactions, max_depth=args.depth)
        )

    # Write products
    df = pd.DataFrame(all_rows)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")
    
if __name__ == "__main__":
    main()
