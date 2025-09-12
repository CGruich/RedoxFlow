from __future__ import annotations

from langflow.load import run_flow_from_json
from typing import Any, Optional, Dict, List
import json
import os
from rdkit import RDLogger
from rdkit import rdBase
from extract_redox_variables import RedoxPotentialMiner
from calculate_nw import redox_potential_BornHaber
from pathlib import Path
import re
import pandas as pd

class RedoxFlow:
    """
    Object wrapper for preparing RedoxFlow agent prompts, running the LangFlow flow,
    and pretty-printing the Python job outputs from the agent.

    Parameters
    ----------
    git_root : Optional[str]
        Filesystem path to your RedoxFlow repo root. If not provided, defaults to $REDOXGIT.
    flow_json_path : Optional[str]
        Full path to the LangFlow JSON. If not provided, defaults to
        {git_root}/scripts/redoxflow.json
    """

    def __init__(self, git_root: Optional[str] = None, flow_json_path: Optional[str] = None):
        self.git_root = git_root or os.environ.get("REDOXGIT")
        if not self.git_root:
            raise EnvironmentError(
                "RedoxFlow: git_root not provided and environment variable REDOXGIT is not set."
            )
        self.flow_json_path = flow_json_path or os.path.join(self.git_root, "scripts", "redoxflow.json")

    def load_flow(self, path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load and return the LangFlow JSON as a Python dict.

        Parameters
        ----------
        path : Optional[str]
            Path to the flow JSON. Defaults to self.flow_json_path.

        Returns
        -------
        dict
        """
        flow_path = path or self.flow_json_path
        with open(flow_path, "r") as f:
            return json.load(f)

    def print_agent_stdouts(self, result: Any, show_stderr: bool = True) -> Dict[str, Any]:
        """
        Pretty-print the stdout (and optionally stderr) for each python job
        reported by the ChemPipeline component in the agent result.

        Parameters
        ----------
        result : Any
            The object returned by run_flow_from_json(...)
        show_stderr : bool
            Whether to also print stderr blocks.

        Returns
        -------
        dict
            Parsed payload from the agent message (for further inspection).
        """
        # 1) Extract the JSON string produced by the ChemPipeline component
        text = result[0].outputs[0].results["message"].data["text"]
        payload = json.loads(text)

        # 2) Normalize runs list (supports both python_runs and legacy python_run)
        runs = payload.get("python_runs") or []
        if not runs and payload.get("python_run"):  # legacy single
            runs = [payload["python_run"]]

        if not runs:
            print("No python runs found in the agent output.")
            return payload  # return in case you want to inspect other fields

        # 3) Pretty print stdout (and optionally stderr) for each run
        sep = "_" * 96
        for i, r in enumerate(runs, 1):
            cmd = r.get("cmd", "<no cmd>")
            rc = r.get("rc", None)
            stdout = r.get("stdout", "") or ""
            stderr = r.get("stderr", "") or ""

            job = "AGENT JOB"
            if "reduce.py" in cmd:
                job = "REDUCTION CANDIDATE SEARCH"
            elif "conformer.py" in cmd:
                job = "CONFORMER SEARCH"
            elif "generate_nw.py" in cmd:
                job = "SCRIPT GENERATION"

            print(sep)
            print(f"[{job}]\n rc={rc}  cmd: {cmd}")
            print("\n--- Output ---")
            print(stdout if stdout.strip() else "")
            if show_stderr:
                print("\n")
                print(stderr if stderr.strip() else "")
        print(sep)

        return payload

    def validate_params(self, max_heavy_atoms_per_reactant: int, min_unique_elements: int) -> None:
        """
        Ensures:
        - max_heavy_atoms_per_reactant >= 4
        - min_unique_elements <= 3

        Raises:
        ValueError with the exact phrasing requested.
        """
        if max_heavy_atoms_per_reactant < 4:
            raise ValueError("max_heavy_atoms_per_reactant must match '>= 4' for proof-of-concept demonstration.")
        if min_unique_elements > 3:
            raise ValueError("min_unique_elements must match '<= 3' for proof-of-concept demonstration.")

    def prepare_scripts(
        self,
        num_generated_reactants: int = 10,
        max_heavy_atoms_per_reactant: int = 16,
        min_uniq_elements: int = 1,
        temperature: float = 0.425,
        device: str = "auto",
        verbose: bool = True,
    ) -> Any:
        """
        Build the agent prompt and run the LangFlow flow to:
          1) generate reduction candidates,
          2) perform conformer generation,
          3) write NWChem input scripts.
        """

        self.validate_params(max_heavy_atoms_per_reactant=max_heavy_atoms_per_reactant, 
                             min_unique_elements=min_uniq_elements)

        # Suppress RDKit logs (mirrors original behavior)
        RDLogger.DisableLog("rdApp.*")
        rdBase.DisableLog("rdApp.*")

        flow = self.load_flow(self.flow_json_path)

        # Build the prompt using self.git_root
        gr = self.git_root
        prompt = {
            "count": num_generated_reactants,
            "max_atoms": max_heavy_atoms_per_reactant,
            "min_unique_elements": min_uniq_elements,
            "elements": "CNOF",
            "temperature": temperature,
            "top_p": 0.9,
            "repetition_penalty": 1.10,
            "num_return_sequences": 96,
            "device": device,
            "python_jobs": [
                {
                    "script": f"{gr}/scripts/reduce.py",
                    "args": [
                        "--molecules", f"{gr}/memory/reactants_memory.csv",
                        "--reactions", f"{gr}/scripts/reactions.json",
                        "--out", f"{gr}/memory/reduction_candidates.csv",
                        "--depth", "2",
                    ],
                },
                {
                    "script": f"{gr}/scripts/conformer.py",
                    "args": [
                        "--in", f"{gr}/memory/reduction_candidates.csv",
                        "--out", f"{gr}/memory/products_memory.csv",
                        "--max-confs", "200",
                        "--min-steps", "1000",
                        "--seed", "1337",
                        "--batch-size", "1000",
                    ],
                },
                {
                    "script": f"{gr}/scripts/generate_nw.py",
                    "args": [
                        "--reactants_in", f"{gr}/memory/reactants",
                        "--products_in", f"{gr}/memory/products",
                        "--scripts_out", f"{gr}/memory/scripts",
                    ],
                },
            ],
            "memory_path": f"{gr}/memory/reactants_memory.csv",
        }

        result = run_flow_from_json(flow, input_value=json.dumps(prompt), cache=False, disable_logs=True)
        if verbose:
            self.print_agent_stdouts(result)
        return result

    def extract_variables(self, sim_root: str | None = None, sim_ext: str = ".out") -> None:
        """Extraction of variables from outputs or files.

        Scans each folder inside '{sim_path}/reactants' and '{sim_path}/products'
        (plus those roots themselves) for files ending with `sim_ext`, and for each
        match:
            miner = RedoxPotentialMiner(stdoutFilepath=ABS_PATH)
            miner.variableDF = {}
            miner.search()

        Stores miners in:
            self.miners = {
                "reactants": [ ... ],
                "products":  [ ... ],
            }
        Records errors (if any) in:
            self.miner_errors = {
                "reactants": [(filepath, exc), ...],
                "products":  [(filepath, exc), ...],
            }
        """
        sim_path = Path(sim_root) if sim_root is not None else Path(self.git_root)

        react_path = sim_path / "reactants"
        prod_path  = sim_path / "products"

        ext = sim_ext if sim_ext.startswith(".") else f".{sim_ext}"
        ext = ext.lower()

        # Prepare containers
        self.miners = {"reactants": [], "products": []}
        self.miner_errors = {"reactants": [], "products": []}

        def _add_file(path: Path, bucket_key: str) -> None:
            try:
                miner = RedoxPotentialMiner(stdoutFilepath=str(path.resolve()))
                miner.variableDF = {}
                miner.search() 
                self.miners[bucket_key].append(miner)
            except Exception as e:
                self.miner_errors[bucket_key].append((str(path), e))

        def _scan_bucket(root: Path, bucket_key: str) -> None:
            if not root.exists():
                print(f"⚠️ Path not found: {root}")
                return

            # 1) Files directly in the root
            for p in root.iterdir():
                if p.is_file() and p.suffix.lower() == ext:
                    _add_file(p, bucket_key)

            # 2) For each immediate subfolder, recurse within it
            for child in root.iterdir():
                if child.is_dir():
                    for f in child.rglob(f"*{ext}"):
                        if f.is_file() and f.suffix.lower() == ext:
                            _add_file(f, bucket_key)

        _scan_bucket(react_path, "reactants")
        _scan_bucket(prod_path,  "products")

    def calculate_redox(self, n_reduction: int = None, 
                        reference_state: float = 0.0,
                        sim_root: str | None = None, 
                        sim_ext: str = ".csv") -> None:
        """
        Pair react_{X}_redox{ext} with prod_{X}_redox{ext} (X must match) and compute:
            redox_potential_BornHaber(react_df, prod_df, h2_df, nRduction=..., referenceState=...)

        H2 dataframe is loaded (if available) from:  {base}/H2/**/H2_redox{ext}
        """
        base = Path(sim_root) if sim_root is not None else Path(self.git_root)
        react_root = base / "reactants"
        prod_root  = base / "products"
        H2_root    = base / "H2"

        ext = sim_ext if sim_ext.startswith(".") else f".{sim_ext}"
        ext = ext.lower()

        # Containers
        self.redox_results: dict = {}
        self.redox_unmatched: dict = {"react_only": [], "prod_only": []}
        self.redox_errors: list = []
        self.h2_source: dict = {"path": None, "loaded": False, "error": None}

        # Filenames must be exactly react_{X}_redox{ext} / prod_{X}_redox{ext}
        react_pat = re.compile(rf"^react_(.+?)_redox{re.escape(ext)}$", re.IGNORECASE)
        prod_pat  = re.compile(rf"^prod_(.+?)_redox{re.escape(ext)}$",  re.IGNORECASE)

        def _collect(root: Path, kind: str) -> dict[str, list[Path]]:
            """Collect files keyed by X; values are lists of Paths (handle duplicates)."""
            mapping: dict[str, list[Path]] = {}
            if not root.exists():
                print(f"⚠️ Path not found: {root}")
                return mapping

            def consider(p: Path):
                if not (p.is_file() and p.suffix.lower() == ext):
                    return
                m = (react_pat if kind == "react" else prod_pat).match(p.name)
                if not m:
                    return
                X = m.group(1)
                mapping.setdefault(X, []).append(p.resolve())

            # scan root files
            for p in root.iterdir():
                consider(p)
            # scan each immediate subfolder recursively
            for child in root.iterdir():
                if child.is_dir():
                    for f in child.rglob(f"*{ext}"):
                        consider(f)
            return mapping

        react_map = _collect(react_root, "react")
        prod_map  = _collect(prod_root,  "prod")

        react_keys = set(react_map)
        prod_keys  = set(prod_map)
        matched    = sorted(react_keys & prod_keys)
        self.redox_unmatched["react_only"] = sorted(react_keys - prod_keys)
        self.redox_unmatched["prod_only"]  = sorted(prod_keys - react_keys)

        def _pick_one(paths: list[Path]) -> Path:
            # newest by mtime; tiebreaker by path string
            return sorted(paths, key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)[0]

        # Load H2 dataframe: must be named exactly "H2_redox{ext}" somewhere under H2_root
        h2_df = None
        h2_path: str | None = None
        try:
            candidates = list(H2_root.rglob(f"H2_redox{ext}")) if H2_root.exists() else []
            if candidates:
                h2_file = _pick_one(candidates)
                h2_df = pd.read_csv(h2_file)
                h2_path = str(h2_file.resolve())
                self.h2_source.update({"path": h2_path, "loaded": True, "error": None})
            else:
                self.h2_source.update({"path": None, "loaded": False, "error": f"No H2_redox{ext} found"})
        except Exception as e:
            self.h2_source.update({"path": h2_path, "loaded": False, "error": repr(e)})
            # proceed with h2_df = None; downstream may error and be recorded per-X

        # Compute redox for matched pairs
        for X in matched:
            try:
                react_path = _pick_one(react_map[X])
                prod_path  = _pick_one(prod_map[X])

                react_df = pd.read_csv(react_path)
                prod_df  = pd.read_csv(prod_path)

                value = redox_potential_BornHaber(
                    react_df,
                    prod_df,
                    h2_df,
                    nReduction=n_reduction,
                    referenceState=reference_state,
                )

                self.redox_results[X] = {
                    "react_path": str(react_path),
                    "prod_path":  str(prod_path),
                    "h2_path":    h2_path,
                    "n_reduction (# participating electrons)": n_reduction,
                    "reference_state": reference_state,
                    "redox potential (J/C or V)": value,
                }
            except Exception as e:
                self.redox_errors.append((
                    X,
                    [str(p) for p in react_map.get(X, [])],
                    [str(p) for p in prod_map.get(X, [])],
                    repr(e),
                ))
