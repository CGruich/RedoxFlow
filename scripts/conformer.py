#!/usr/bin/env python3
"""
conformer_pick_lowest_mmff94.py

Brute-force conformer search (OpenBabel) for each Product_SMILES,
MMFF94-minimize all conformers, pick the lowest-energy conformer per product,
then choose the lowest-energy product per Reactant_name.

Also saves (ONE FILE PER REACTANT):
  - products/prod_<Reactant_name>.xyz  (winner product conformer)
  - reactants/react_<Reactant_name>.xyz  (best conformer of the actual Reactant_SMILES)

Output CSV columns: Reactant_name, Reactant_SMILES, Product_SMILES

Usage:
  python conformer_pick_lowest_mmff94.py --in data.tsv --out winners.csv
  # Optional knobs:
  #   --max-confs 300   (upper bound of conformers to enumerate)
  #   --min-steps 1000  (minimization steps per conformer)   [ignored in obabel CLI path]
  #   --seed 1337                                       [ignored in obabel CLI path]
"""

import argparse
import csv
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from collections import namedtuple
from typing import Optional, Tuple, Dict, List

Record = namedtuple(
    "Record",
    [
        "Reactant_name",
        "Reactant_SMILES",
        "Depth",
        "Parent_SMILES",
        "Reaction_name",
        "Reaction_scheme",
        "Product_SMILES",
        "row_index",
    ],
)

def sniff_delimiter(path: str) -> str:
    with open(path, "r", newline="", encoding="utf-8") as f:
        sample = f.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,;|")
        return dialect.delimiter
    except Exception:
        # Heuristic: prefer tab if tabs appear; else comma
        return "\t" if ("\t" in sample and sample.count("\t") >= sample.count(",")) else ","

def read_table(path: str):
    delim = sniff_delimiter(path)
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delim)
        required = {
            "Reactant_name", "Reactant_SMILES", "Depth", "Parent_SMILES",
            "Reaction_name", "Reaction_scheme", "Product_SMILES"
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        rows = []
        for i, row in enumerate(reader):
            rows.append(
                Record(
                    row["Reactant_name"],
                    row["Reactant_SMILES"],
                    row["Depth"],
                    row["Parent_SMILES"],
                    row["Reaction_name"],
                    row["Reaction_scheme"],
                    row["Product_SMILES"],
                    i,
                )
            )
    return rows

# ---------- obabel / obenergy CLI helpers ----------

def _parse_obenergy_all(text: str) -> List[float]:
    """
    Parse *all* MMFF94 energies from obenergy output, in order.
    Matches lines like:
      'Total MMFF94 energy = 12.3456 kcal/mol'
      'MMFF94 energy: 12.3456'
    Returns a list of floats (possibly empty).
    """
    if not text:
        return []
    pat = re.compile(r'(?i)(?:total\s+)?(?:mmff94|uff)?\s*energy[^-\d]*(-?\d+(?:\.\d+)?)\s*(?:kcal|kcal/mol)?')
    vals = [float(x) for x in pat.findall(text)]
    return vals

def _parse_multi_xyz(stream_text: str) -> List[Tuple[str, str]]:
    """
    Parse a multi-XYZ text into a list of (title, xyz_block_text).
    Assumes standard XYZ blocks:
      line 0: N
      line 1: comment/title (we put a numeric ID here)
      next N lines: atoms
    """
    lines = stream_text.splitlines()
    i = 0
    blocks: List[Tuple[str, str]] = []
    nlines = len(lines)
    while i < nlines:
        while i < nlines and not lines[i].strip():
            i += 1
        if i >= nlines:
            break
        try:
            n = int(lines[i].strip())
        except Exception:
            break
        if i + 1 >= nlines:
            break
        title = lines[i + 1].strip()
        start = i
        end = i + 2 + n
        if end > nlines:
            break
        block = "\n".join(lines[start:end]) + "\n"
        blocks.append((title, block))
        i = end
    return blocks

def _run_obabel_to_multi_xyz(
    smi_path: Path,
    *,
    nconf: int,
    obabel_bin: str,
    timeout: int = 600,
) -> Optional[str]:
    """
    Run `obabel input.smi -oxyz --gen3d --best --conformer --nconf N --score energy --weighted`
    and return the captured multi-XYZ text, or None on failure.
    """
    cmd = [
        obabel_bin, str(smi_path), "-oxyz",
        "--gen3d", "--best", "--conformer", "--nconf", str(nconf),
        "--score", "energy", "--weighted",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError:
        sys.stderr.write("ERROR: 'obabel' binary not found on PATH. Install Open Babel (e.g., conda-forge openbabel).\n")
        return None
    except subprocess.TimeoutExpired:
        sys.stderr.write("  [diag] obabel call timed out\n")
        return None

    if proc.returncode != 0:
        sys.stderr.write(f"  [diag] obabel nonzero exit ({proc.returncode}): {proc.stderr.strip()}\n")
        return None

    return proc.stdout

def _run_obenergy_on_multi_xyz_file(
    xyz_path: Path,
    *,
    obenergy_bin: str,
    timeout: int = 300,
) -> List[float]:
    """
    Run `obenergy -ff MMFF94 <xyz_path>` and return a list of energies (kcal/mol) in order.
    """
    cmd = [obenergy_bin, "-ff", "MMFF94", str(xyz_path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError:
        sys.stderr.write("ERROR: 'obenergy' binary not found on PATH. Install Open Babel (e.g., conda-forge openbabel).\n")
        return []
    except subprocess.TimeoutExpired:
        sys.stderr.write("  [diag] obenergy call timed out\n")
        return []

    if proc.returncode != 0 and not proc.stdout:
        sys.stderr.write(f"  [diag] obenergy nonzero exit ({proc.returncode}): {proc.stderr.strip()}\n")
        # still try to parse whatever it printed
    energies = _parse_obenergy_all(proc.stdout) + _parse_obenergy_all(proc.stderr)
    return energies

def _run_obenergy_on_xyz_block(
    xyz_text: str,
    *,
    obenergy_bin: str,
    timeout: int = 60,
) -> Optional[float]:
    """
    Fallback: write one XYZ block to a temp file and run obenergy on it.
    """
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "one.xyz"
        p.write_text(xyz_text, encoding="utf-8")
        vals = _run_obenergy_on_multi_xyz_file(p, obenergy_bin=obenergy_bin, timeout=timeout)
        if vals:
            return vals[0]
    return None

def _rewrite_xyz_comment(xyz_block: str, new_comment: str) -> str:
    """
    Replace the comment (2nd line) of an XYZ block with new_comment.
    """
    lines = xyz_block.splitlines()
    if len(lines) < 2:
        return xyz_block
    lines[1] = new_comment
    return "\n".join(lines) + ("\n" if not xyz_block.endswith("\n") else "")

def _sanitize(name: str, maxlen: int = 80) -> str:
    # Keep it filesystem-safe
    name = re.sub(r"[^A-Za-z0-9._+-]+", "_", name.strip())
    if len(name) > maxlen:
        name = name[:maxlen]
    return name

def _batch_best_conformer_results(
    smiles_list: List[str],
    *,
    nconf: int,
    obabel_bin: str,
    obenergy_bin: str,
    batch_size: int = 1000,
) -> Dict[str, Tuple[Optional[float], Optional[str]]]:
    """
    Batch unique SMILES:
      - write .smi with 'SMILES<TAB>ID' (ID is 0..)
      - obabel to multi-XYZ (best conformer per SMILES)
      - write multi-XYZ to file, run single obenergy -ff MMFF94 over it
      - map energies back to SMILES by order/title
      - if counts mismatch, fallback per-block only for those missing

    Returns {smiles: (energy or None, xyz_block or None)}.
    """
    results: Dict[str, Tuple[Optional[float], Optional[str]]] = {s: (None, None) for s in smiles_list}
    if not smiles_list:
        return results

    for chunk_start in range(0, len(smiles_list), batch_size):
        chunk = smiles_list[chunk_start: chunk_start + batch_size]
        id_to_smiles = {str(i): s for i, s in enumerate(chunk)}

        with tempfile.TemporaryDirectory() as td:
            smi_path = Path(td) / "batch.smi"
            with open(smi_path, "w", encoding="utf-8") as f:
                for i, s in enumerate(chunk):
                    f.write(f"{s}\t{i}\n")

            # 1) obabel -> multi-XYZ (stdout)
            xyz_text = _run_obabel_to_multi_xyz(smi_path, nconf=nconf, obabel_bin=obabel_bin)
            if not xyz_text:
                continue

            # Parse XYZ into ordered blocks (title is our numeric ID)
            blocks = _parse_multi_xyz(xyz_text)
            if not blocks:
                continue

            # 2) Run a single obenergy over the whole batch
            xyz_path = Path(td) / "batch.xyz"
            xyz_path.write_text(xyz_text, encoding="utf-8")
            batch_Es = _run_obenergy_on_multi_xyz_file(xyz_path, obenergy_bin=obenergy_bin)

            # Map by order; if mismatch, we'll fallback selectively
            per_idx_energy: List[Optional[float]] = [None] * len(blocks)
            for k, E in enumerate(batch_Es[:len(blocks)]):
                per_idx_energy[k] = E

            # Fallback per-block where missing
            missing_idxs = [k for k, v in enumerate(per_idx_energy) if v is None]
            for k in missing_idxs:
                _title, xyz_block = blocks[k]
                e1 = _run_obenergy_on_xyz_block(xyz_block, obenergy_bin=obenergy_bin)
                per_idx_energy[k] = e1

            # Assign back to SMILES via title (ID)
            for k, ((title, xyz_block), E) in enumerate(zip(blocks, per_idx_energy)):
                smi = id_to_smiles.get(title)
                if smi is not None:
                    results[smi] = (E, xyz_block)

    return results

# ---------- main pipeline ----------

def main():
    p = argparse.ArgumentParser(
        description="Pick lowest-energy product per Reactant_name using OpenBabel MMFF94."
    )
    p.add_argument("--in", dest="in_path", required=True, help="Input TSV/CSV table.")
    p.add_argument("--out", dest="out_path", required=True, help="Output CSV path (winners).")
    p.add_argument(
        "--max-confs",
        type=int,
        default=200,
        help="Max conformers to enumerate per product (default: 200).",
    )
    p.add_argument(
        "--min-steps",
        type=int,
        default=1000,
        help="MMFF94 minimization steps per conformer (default: 1000). [ignored by obabel path]",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for conformer search (default: 1337). [ignored by obabel path]",
    )
    p.add_argument(
        "--obabel",
        dest="obabel_bin",
        default="obabel",
        help="Path to the obabel executable (default: 'obabel').",
    )
    p.add_argument(
        "--obenergy",
        dest="obenergy_bin",
        default="obenergy",
        help="Path to the obenergy executable (default: 'obenergy').",
    )
    p.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=1000,
        help="How many unique SMILES to process per obabel batch (default: 1000).",
    )
    args = p.parse_args()

    # Quick checks: obabel & obenergy available?
    missing = []
    if shutil.which(args.obabel_bin) is None:
        missing.append(f"'{args.obabel_bin}'")
    if shutil.which(args.obenergy_bin) is None:
        missing.append(f"'{args.obenergy_bin}'")
    if missing:
        sys.stderr.write(
            "ERROR: Missing required executables on PATH: "
            + ", ".join(missing)
            + ". Install Open Babel (e.g., conda-forge openbabel) or pass explicit paths via --obabel/--obenergy.\n"
        )
        sys.exit(4)

    # Read rows
    try:
        rows = read_table(args.in_path)
    except Exception as e:
        sys.stderr.write(f"ERROR: failed to read table '{args.in_path}': {e}\n")
        sys.exit(2)

    if not rows:
        sys.stderr.write("WARNING: no rows found in input.\n")

    # ---------- CHANGED: derive output directories from --out ----------
    out_path = Path(args.out_path).resolve()
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)  # ensure parent directory exists

    products_dir = out_dir / "products"
    reactants_dir = out_dir / "reactants"
    products_dir.mkdir(parents=True, exist_ok=True)
    reactants_dir.mkdir(parents=True, exist_ok=True)
    # ---------------------------------------------------------------

    # Prepare unique Product_SMILES (preserve first-seen order)
    seen = set()
    unique_product_smiles: List[str] = []
    for rec in rows:
        ps = (rec.Product_SMILES or "").strip()
        if ps and ps not in seen:
            seen.add(ps)
            unique_product_smiles.append(ps)

    # Batch-evaluate energies + grab best-conformer XYZ for unique Product_SMILES
    product_results: Dict[str, Tuple[Optional[float], Optional[str]]] = _batch_best_conformer_results(
        unique_product_smiles,
        nconf=args.max_confs,
        obabel_bin=args.obabel_bin,
        obenergy_bin=args.obenergy_bin,
        batch_size=args.batch_size,
    )

    # Winners keyed by Reactant_name -> (best_E, Reactant_SMILES, Product_SMILES)
    winners: Dict[str, Tuple[float, str, str]] = {}
    printed: set[str] = set()  # diagnostics per unique Product_SMILES

    for rec in rows:
        psmi = (rec.Product_SMILES or "").strip()
        if not psmi:
            continue

        E, _xyz_block = product_results.get(psmi, (None, None))

        if psmi not in printed:
            sys.stderr.write(
                f"[row {rec.row_index}] Product_SMILES='{psmi}' -> "
                f"{'FAIL' if (E is None) else f'E={E:.6f} kcal/mol'}\n"
            )
            printed.add(psmi)

        if E is None:
            continue

        key = rec.Reactant_name
        current = winners.get(key)
        if (current is None) or (E < current[0]):
            winners[key] = (E, (rec.Reactant_SMILES or "").strip(), psmi)

    # Write output CSV: Reactant_name, Reactant_SMILES, Product_SMILES
    try:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Reactant_name", "Reactant_SMILES", "Product_SMILES"])
            for reactant_name in sorted(winners.keys(), key=lambda s: (str(s))):
                best_E, r_smi, p_smi = winners[reactant_name]
                writer.writerow([reactant_name, r_smi, p_smi])
        sys.stderr.write(f"Wrote {len(winners)} winners to {out_path}\n")
    except Exception as e:
        sys.stderr.write(f"ERROR: failed to write output '{out_path}': {e}\n")
        sys.exit(3)

    # ---- Save PRODUCT winners (one per reactant) ----
    for reactant_name, (best_E, r_smi, p_smi) in winners.items():
        E_p, xyz_block_p = product_results.get(p_smi, (None, None))
        if E_p is None or not xyz_block_p:
            continue
        reactant_id = _sanitize(str(reactant_name)) or "reactant"
        comment_p = f"REACTANT_ID={reactant_id} SMILES={p_smi} ENERGY={E_p:.6f} kcal/mol"
        xyz_out_p = _rewrite_xyz_comment(xyz_block_p, comment_p)
        (products_dir / f"prod_{reactant_id}.xyz").write_text(xyz_out_p, encoding="utf-8")

    # ---- Compute & Save REACTANT conformers (one per reactant) ----
    # Collect unique reactant SMILES among winners
    reactant_name_to_smi: Dict[str, str] = {str(name): smi for name, (_E, smi, _p) in winners.items() if smi.strip()}
    unique_reactant_smiles: List[str] = list(dict.fromkeys(reactant_name_to_smi.values()))

    reactant_results: Dict[str, Tuple[Optional[float], Optional[str]]] = _batch_best_conformer_results(
        unique_reactant_smiles,
        nconf=args.max_confs,
        obabel_bin=args.obabel_bin,
        obenergy_bin=args.obenergy_bin,
        batch_size=args.batch_size,
    )

    # Save each reactant's conformer
    for reactant_name, r_smi in reactant_name_to_smi.items():
        E_r, xyz_block_r = reactant_results.get(r_smi, (None, None))
        reactant_id = _sanitize(str(reactant_name)) or "reactant"
        if E_r is None or not xyz_block_r:
            sys.stderr.write(f"[diag] reactant '{reactant_id}' SMILES='{r_smi}' -> FAIL\n")
            continue
        sys.stderr.write(f"[diag] reactant '{reactant_id}' SMILES='{r_smi}' -> E={E_r:.6f} kcal/mol\n")
        comment_r = f"REACTANT_ID={reactant_id} SMILES={r_smi} ENERGY={E_r:.6f} kcal/mol"
        xyz_out_r = _rewrite_xyz_comment(xyz_block_r, comment_r)
        (reactants_dir / f"react_{reactant_id}.xyz").write_text(xyz_out_r, encoding="utf-8")

if __name__ == "__main__":
    main()
