from langflow.load import run_flow_from_json
import json
import os
from rdkit import RDLogger
from rdkit import rdBase

def load_flow(path):
    with open(path, "r") as f:
        return json.load(f)

def print_agent_stdouts(result, show_stderr=True):

    """
    result: the object returned by run_flow_from_json(...)
    show_stderr: set True to also print stderr blocks
    """
    # 1) Extract the JSON string produced by ChemPipeline component
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

        if 'reduce.py' in cmd:
            job = 'REDUCTION CANDIDATE SEARCH'
        elif 'conformer.py' in cmd:
            job = 'CONFORMER SEARCH'
        elif 'generate_nw.py' in cmd:
            job = 'SCRIPT GENERATION'
        print(sep)
        print(f"[{job}]\n rc={rc}  cmd: {cmd}")
        print("\n--- Output ---")
        print(stdout if stdout.strip() else "")
        if show_stderr:
            print("\n")
            print(stderr if stderr.strip() else "")
    print(sep)

def prepare_scripts(num_generated_reactants: int = 10,
                    max_heavy_atoms_per_reactant: int = 16,
                    min_uniq_elements: int = 1,
                    temperature: float = 0.425,
                    device: str = 'auto',
                    verbose=True):
    

    RDLogger.DisableLog("rdApp.*")   # suppress error/warn/info
    rdBase.DisableLog("rdApp.*")     # some RDKit builds use this hook too

    flow_json_path = f"{os.environ['REDOXGIT']}/scripts/redoxflow.json"
    
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
            "script": f"{os.environ['REDOXGIT']}/scripts/reduce.py",
            "args": [
            "--molecules",f"{os.environ['REDOXGIT']}/memory/reactants_memory.csv",
            "--reactions", f"{os.environ['REDOXGIT']}/scripts/reactions.json",
            "--out", f"{os.environ['REDOXGIT']}/memory/reduction_candidates.csv",
            "--depth", "2"
            ]
        },
        {
            "script": f"{os.environ['REDOXGIT']}/scripts/conformer.py",
            "args": [
            "--in", f"{os.environ['REDOXGIT']}/memory/reduction_candidates.csv",
            "--out", f"{os.environ['REDOXGIT']}/memory/products_memory.csv",
            "--max-confs", "200",
            "--min-steps", "1000",
            "--seed", "1337",
            "--batch-size", "1000"
            ]
        },
        {
            "script": f"{os.environ['REDOXGIT']}/scripts/generate_nw.py",
            "args": [
            "--reactants_in", f"{os.environ['REDOXGIT']}/memory/reactants",
            "--products_in", f"{os.environ['REDOXGIT']}/memory/products",
            "--scripts_out", f"{os.environ['REDOXGIT']}/memory/scripts"
            ]
        }
        ],

        "memory_path": f"{os.environ['REDOXGIT']}/memory/reactants_memory.csv"
        }
    
    # 1) Load flow dict
    flow = load_flow(flow_json_path)

    result = run_flow_from_json(flow, input_value=json.dumps(prompt), cache=False, disable_logs=True)

    if verbose:
        print_agent_stdouts(result)
    
    return result
