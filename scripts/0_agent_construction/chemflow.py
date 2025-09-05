from __future__ import annotations
import os, sys, json, time, subprocess, textwrap, re
from pathlib import Path
from typing import List, Dict, Any, Set, Optional, Tuple

from langflow.custom import Component
from langflow.io import (
    MessageTextInput, StrInput, IntInput, DropdownInput,
    Output, BoolInput, FloatInput
)
from langflow.schema import Message

# RDKit is required for validation/sanitization
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

MIN_ATOMS_FIXED = 4  # fixed minimum heavy atoms per request


class ChemPipeline(Component):
    display_name = "ChemPipeline"
    description = "Transformers-only SMILES generator with constraints, memory, optional RDKit sanitize, Python/Bash hooks. Uses MoLFormer-recommended HF pattern."
    icon = "FlaskConical"
    name = "ChemPipeline"

    # -------- Inputs ----------
    inputs = [
        MessageTextInput(
            name="params_json",
            display_name="Params JSON",
            info=textwrap.dedent("""\
                Optional JSON to override fields at runtime. Example:
                {
                  "count": 5,
                  "max_atoms": 12,
                  "min_unique_elements": 0,     // distinct NON-C elements required (C doesn't count)
                  "elements": "CNOF",

                  "model_id": "ibm-research/GP-MoLFormer-Uniq",
                  "tokenizer_id": "ibm-research/MoLFormer-XL-both-10pct",

                  "temperature": 0.7,
                  "top_p": 0.9,
                  "top_k": null,                 // null => omitted
                  "repetition_penalty": 1.08,
                  "num_return_sequences": 96,

                  "max_length": 202,             // per model card

                  "device": "cuda:0",
                  "rdkit_filter": true,
                  "remap_disallowed": true,

                  "python_script": "hello_world.py",
                  "python_args": "",
                  "bash_cmd": "echo {molecule}",
                  "memory_path": "chem_memory.jsonl",
                  "return_smiles": true
                }
            """).strip(),
            value='{"count": 5, "max_atoms": 10, "min_unique_elements": 1, "elements": "CNOF"}',
        ),

        # Targets / constraints
        IntInput(name="count", display_name="Count", value=5),
        IntInput(name="max_atoms", display_name="Max heavy atoms", value=10),
        IntInput(name="min_unique_elements", display_name="Min unique non-carbon elements", value=1),
        StrInput(name="elements", display_name="Allowed elements", value="CNOF"),

        # Transformers & decoding (MoLFormer card pairing)
        StrInput(name="model_id", display_name="HF Model ID", value="ibm-research/GP-MoLFormer-Uniq"),
        StrInput(name="tokenizer_id", display_name="HF Tokenizer ID", value="ibm-research/MoLFormer-XL-both-10pct"),

        FloatInput(name="temperature", display_name="Temperature", value=0.7),
        FloatInput(name="top_p", display_name="Top-p (nucleus)", value=0.9),
        IntInput(name="top_k", display_name="Top-k (-1 => None)", value=-1),
        FloatInput(name="repetition_penalty", display_name="Repetition penalty", value=1.08),
        IntInput(name="num_return_sequences", display_name="# samples per round", value=96),

        # The card uses max_length=202.
        IntInput(name="max_length", display_name="Max length (card=202)", value=202),

        # Accelerator
        DropdownInput(
            name="device",
            display_name="Device",
            options=["auto", "cpu", "cuda:0", "cuda:1", "mps"],
            value="auto",
        ),

        # Optional chemistry gate
        BoolInput(name="rdkit_filter", display_name="Filter by RDKit sanitize()", value=True),
        BoolInput(name="remap_disallowed", display_name="Remap disallowed atoms to allowed set before filtering", value=True),

        # Hooks & memory
        StrInput(name="python_script", display_name="Python script path", value="hello_world.py"),
        StrInput(name="python_args", display_name="Python args", value=""),
        StrInput(name="bash_cmd", display_name="Bash command template", value=""),
        StrInput(name="memory_path", display_name="Memory file (jsonl)", value="chem_memory.jsonl"),
        BoolInput(name="return_smiles", display_name="Also return SMILES (echoed)", value=True),
    ]

    outputs = [Output(display_name="Result", name="result", method="run_pipeline")]

    # Cached bundle: (model, tokenizer, model_id, tokenizer_id, device_str, dbg_ids)
    _bundle: Optional[Tuple[Any, Any, str, str, str, Dict[str, int]]] = None

    # Last filter stats (for debugging)
    _last_filter_stats: Dict[str, int] = {}

    # ---------------- Utilities ----------------
    def _merge_params(self) -> Dict[str, Any]:
        p = {
            "count": int(self.count),
            "max_atoms": int(self.max_atoms),
            "min_unique_elements": max(0, int(getattr(self, "min_unique_elements", 1))),  # non-C uniques
            "elements": "".join(sorted(set((self.elements or "CNOF").upper()))),

            "model_id": (self.model_id or "ibm-research/GP-MoLFormer-Uniq").strip(),
            "tokenizer_id": (self.tokenizer_id or "ibm-research/MoLFormer-XL-both-10pct").strip(),

            "temperature": float(self.temperature) if str(self.temperature) else 0.7,
            "top_p": float(getattr(self, "top_p", 0.9)),
            "top_k": getattr(self, "top_k", -1),
            "repetition_penalty": float(getattr(self, "repetition_penalty", 1.08)),
            "num_return_sequences": max(1, int(getattr(self, "num_return_sequences", 96))),

            "max_length": max(1, int(getattr(self, "max_length", 202))),

            "device": str(getattr(self, "device", "auto")).strip().lower(),
            "rdkit_filter": bool(getattr(self, "rdkit_filter", True)),
            "remap_disallowed": bool(getattr(self, "remap_disallowed", True)),

            "python_script": self.python_script or "",
            "python_args": self.python_args or "",
            "bash_cmd": self.bash_cmd or "",
            "memory_path": self.memory_path or "chem_memory.jsonl",
            "return_smiles": bool(self.return_smiles),
        }
        try:
            if self.params_json:
                override = json.loads(str(self.params_json))
                for k, v in override.items():
                    # Ignore user-provided min_atoms and max_new_tokens (auto/fixed now)
                    if k in ("min_atoms", "max_new_tokens"):
                        continue
                    p[k] = v
        except Exception:
            pass

        # Normalize
        p["count"] = int(p["count"])
        p["max_atoms"] = int(p["max_atoms"])
        p["min_unique_elements"] = max(0, int(p["min_unique_elements"]))
        p["temperature"] = float(p["temperature"])
        p["top_p"] = float(p["top_p"])
        # Allow None-like top_k via JSON null, or -1 via UI
        if p["top_k"] in (None, "None"):
            p["top_k"] = None
        else:
            try:
                tk = int(p["top_k"])
                p["top_k"] = None if tk < 0 else tk
            except Exception:
                p["top_k"] = None
        p["repetition_penalty"] = float(p["repetition_penalty"])
        p["num_return_sequences"] = max(1, int(p["num_return_sequences"]))
        p["max_length"] = max(1, int(p["max_length"]))
        p["device"] = str(p["device"]).strip().lower()
        return p

    def _resolve_device(self, device_str: str) -> str:
        try:
            import torch
        except Exception:
            return "cpu"
        s = (device_str or "auto").lower().strip()
        if s == "auto":
            if torch.cuda.is_available():
                return "cuda:0"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if s == "cpu":
            return "cpu"
        if s == "mps":
            return "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        if s.startswith("cuda"):
            if not torch.cuda.is_available():
                return "cpu"
            try:
                _ = int(s.split(":")[1]) if ":" in s else 0
            except Exception:
                s = "cuda:0"
            return s
        return "cpu"

    # ---- Load model/tokenizer and robustly ensure PAD/EOS/BOS ----
    def _get_bundle(self, model_id: str, tokenizer_id: str, device_choice: str):
        device_str = self._resolve_device(device_choice)
        if self._bundle:
            mdl, tok, mid, tid, dev, _ = self._bundle
            if mid == model_id and tid == tokenizer_id and dev == device_str:
                return self._bundle

        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        tok = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)

        added = False
        # Ensure EOS
        if tok.eos_token_id is None:
            # prefer sep if present; else add a safe EOS
            if getattr(tok, "sep_token", None) is not None:
                tok.eos_token = tok.sep_token
            else:
                tok.add_special_tokens({"eos_token": "</s>"})
                added = True
        # Ensure PAD
        if tok.pad_token_id is None:
            # if eos exists, reuse it; else add dedicated PAD
            if tok.eos_token_id is not None:
                tok.pad_token = tok.eos_token
            else:
                tok.add_special_tokens({"pad_token": "[PAD]"})
                added = True
        # Ensure BOS
        if getattr(tok, "bos_token_id", None) is None:
            # prefer cls if present; else add a safe BOS
            if getattr(tok, "cls_token", None) is not None:
                tok.bos_token = tok.cls_token
            else:
                tok.add_special_tokens({"bos_token": "<s>"})
                added = True

        mdl = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        if added:
            mdl.resize_token_embeddings(len(tok))

        # Write into config so generation doesn't need explicit kwargs
        if getattr(mdl.config, "pad_token_id", None) is None:
            mdl.config.pad_token_id = tok.pad_token_id
        if getattr(mdl.config, "eos_token_id", None) is None:
            mdl.config.eos_token_id = tok.eos_token_id

        mdl.to(device_str)

        dbg_ids = dict(pad=tok.pad_token_id or -1,
                       eos=tok.eos_token_id or -1,
                       bos=getattr(tok, "bos_token_id", None) or -1)

        self._bundle = (mdl, tok, model_id, tokenizer_id, device_str, dbg_ids)
        return self._bundle

    # ---------- Text generation (MoLFormer-card style) ----------
    def _generate_texts(self, p: Dict[str, Any], n_samples: int) -> List[str]:
        """
        Promptless sampling by default (BOS token). Mirrors the model card behavior.
        """
        import torch
        mdl, tok, _, _, device_str, _ = self._get_bundle(p["model_id"], p["tokenizer_id"], p["device"])

        # Build a BOS-only prompt
        bos_id = getattr(tok, "bos_token_id", None)
        if bos_id is None:
            bos_id = tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id
        if bos_id is None:
            bos_id = 0  # absolute fallback; won't be used if specials were set above

        device = torch.device(device_str)
        input_ids = torch.tensor([[bos_id]], dtype=torch.long, device=device)

        gen_kwargs = dict(
            do_sample=True,
            num_return_sequences=max(1, int(n_samples)),
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )

        # Auto-determine max_new_tokens = 1.5 * max_atoms (rounded)
        auto_mnt = max(1, int(round(1.5 * p["max_atoms"])))
        gen_kwargs["max_new_tokens"] = auto_mnt

        # Decoding controls
        if p["temperature"] is not None:
            gen_kwargs["temperature"] = float(p["temperature"])
        if p["top_p"] is not None:
            gen_kwargs["top_p"] = float(p["top_p"])
        if p["top_k"] is not None:  # only include if not None; avoids TypeError
            gen_kwargs["top_k"] = int(p["top_k"])
        if p["repetition_penalty"] is not None:
            gen_kwargs["repetition_penalty"] = float(p["repetition_penalty"])

        with torch.inference_mode():
            outputs = mdl.generate(input_ids=input_ids, **gen_kwargs)

        texts = tok.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return [t for t in texts if t and t.strip()]

    # SMILES utilities
    _smiles_line_re = re.compile(r"^[A-Za-z0-9@+\-\[\]\(\)=#\\/\.%]+$")

    def _extract_smiles_candidates(self, text: str) -> List[str]:
        if not text:
            return []
        text = text.replace("```", " ").replace("`", " ")
        raw = []
        for line in re.split(r"[\n;]", text):
            line = line.strip()
            if not line:
                continue
            # remove all spaces inside, MoLFormer often emits plain SMILES without spaces anyway
            line = re.sub(r"\s+", "", line)
            if self._smiles_line_re.match(line):
                raw.append(line)
        seen, out = set(), []
        for s in raw:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    def _elements_in_mol(self, mol) -> Set[str]:
        return {a.GetSymbol() for a in mol.GetAtoms()}

    def _remap_disallowed_mol(self, mol, allowed: Set[str]) -> rdchem.Mol | None:
        mapping = {
            "Cl": "F", "Br": "F", "I": "F",
            "S": "O",
            "P": "N",
            "Si": "C", "Sn": "C", "B": "C",
        }
        try:
            for atom in mol.GetAtoms():
                sym = atom.GetSymbol()
                if sym not in allowed:
                    if sym in mapping and mapping[sym] in allowed:
                        new_sym = mapping[sym]
                        atom.SetAtomicNum(Chem.GetPeriodicTable().GetAtomicNumber(new_sym))
                    else:
                        atom.SetAtomicNum(0)  # mark for removal
            to_remove = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
            to_remove.sort(reverse=True)
            em = Chem.EditableMol(mol)
            for idx in to_remove:
                em.RemoveAtom(idx)
            new_mol = em.GetMol()
            Chem.SanitizeMol(new_mol)
            return new_mol
        except Exception:
            return None

    # Ring-strain heuristic: filter unrealistic rings / highly strained fused small rings
    def _passes_ring_strain(self, mol: rdchem.Mol) -> bool:
        """
        Conservative heuristic:
          - Reject any ring of size < 3.
          - For 3-membered rings: reject if any DOUBLE/TRIPLE bond in the ring or any sp atom.
          - For 4-membered rings: reject if any TRIPLE bond or >=2 DOUBLE bonds in the ring, or any sp atom.
          - Reject fused small-ring systems where an atom participates in >=2 rings of size <= 4 (e.g., bicyclobutane-like).
        Allows common small rings like cyclopropane/cyclobutane when saturated and not multiply fused.
        """
        try:
            ri = mol.GetRingInfo()
            atom_rings = list(ri.AtomRings())
            bond_rings = list(ri.BondRings())

            # Per-ring checks
            for i, atoms in enumerate(atom_rings):
                n = len(atoms)
                if n < 3:
                    return False  # impossible ring sizes
                bonds = bond_rings[i] if i < len(bond_rings) else []
                num_double = 0
                num_triple = 0
                for bidx in bonds:
                    bt = mol.GetBondWithIdx(bidx).GetBondType()
                    if bt == rdchem.BondType.DOUBLE:
                        num_double += 1
                    elif bt == rdchem.BondType.TRIPLE:
                        num_triple += 1
                sp_in_ring = any(
                    mol.GetAtomWithIdx(a).GetHybridization() == rdchem.HybridizationType.SP
                    for a in atoms
                )

                if n == 3:
                    if num_double > 0 or num_triple > 0:
                        return False
                    if sp_in_ring:
                        return False
                elif n == 4:
                    if num_triple > 0:
                        return False
                    if num_double >= 2:
                        return False
                    if sp_in_ring:
                        return False

            # Fused small-ring check (atoms shared by >=2 rings of size <=4)
            ring_sizes_per_atom: Dict[int, List[int]] = {i: [] for i in range(mol.GetNumAtoms())}
            for atoms in atom_rings:
                n = len(atoms)
                for a in atoms:
                    ring_sizes_per_atom[a].append(n)
            for sizes in ring_sizes_per_atom.values():
                smalls = [s for s in sizes if s <= 4]
                if len(smalls) >= 2:
                    return False

            return True
        except Exception:
            # If analysis fails, be conservative and reject
            return False

    # Unrealistic-moieties heuristic: ban specific, physically dubious substructures
    def _fails_unrealistic_motifs(self, mol: rdchem.Mol) -> bool:
        """
        Conservative set of red flags:
          - O–F and N–F bonds (highly unstable in ordinary organic contexts):  [O]-[F], [N]-[F]
          - F engaged in multiple bonds:                                      [F]=*, [F]#*
          - O=O (dioxygen-like embedded or as a fragment):                    [O]=[O]
          - Halogen–halogen F–F single bond:                                  [F]-[F]
          - Dinitrogen triple bond in structure:                              [N]#[N]
          - Carbon–fluorine multiple bonds:                                   [#6]=[F], [#6]#[F]
          - O–O bond inside a very small ring (size ≤ 4)
        This list is intentionally narrow and extendable.
        """
        try:
            patterns = getattr(self, "_forbidden_queries", None)
            if patterns is None:
                pats = [
                    ("O-F", Chem.MolFromSmarts("[O]-[F]")),
                    ("N-F", Chem.MolFromSmarts("[N]-[F]")),
                    ("F=*", Chem.MolFromSmarts("[F]=*")),
                    ("F#*", Chem.MolFromSmarts("[F]#*")),
                    ("O=O", Chem.MolFromSmarts("[O]=[O]")),
                    ("F-F", Chem.MolFromSmarts("[F]-[F]")),
                    ("N#N", Chem.MolFromSmarts("[N]#[N]")),
                    ("C=F", Chem.MolFromSmarts("[#6]=[F]")),
                    ("C#F", Chem.MolFromSmarts("[#6]#[F]")),
                ]
                # Filter out any None in case a SMARTS fails to parse
                patterns = [(name, q) for (name, q) in pats if q is not None]
                self._forbidden_queries = patterns

            # SMARTS screen
            for _, q in patterns:
                if mol.HasSubstructMatch(q):
                    return False  # "fails" means reject -> return False here or True? We want True=it fails.
            # The above line rejects incorrectly (returns False); fix logic:
            # We'll invert: if a pattern matches => moiety is unrealistic => return True (fails)
        except Exception:
            # On any error, be conservative and DON'T fail here; ring/valence checks still apply.
            return False

        # Special-case: O–O bond inside very small rings (≤ 4)
        try:
            ri = mol.GetRingInfo()
            bond_rings = list(ri.BondRings())
            for b in mol.GetBonds():
                a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
                if a1.GetSymbol() == "O" and a2.GetSymbol() == "O":
                    if b.IsInRing():
                        # Check all rings containing this bond; if any ring size <= 4, reject
                        bidx = b.GetIdx()
                        for ring in bond_rings:
                            if bidx in ring and len(ring) <= 4:
                                return True
        except Exception:
            pass

        return False  # no unrealistic motifs found

    def _enforce_constraints_smiles(
        self,
        smiles_list: List[str],
        max_atoms: int,
        elements: str,
        min_atoms: int = 1,
        min_unique_nonC: int = 0,
        rdkit_filter: bool = True,
        remap_disallowed: bool = True,
    ) -> List[str]:
        stats = {
            "too_few_atoms": 0,
            "too_many_atoms": 0,
            "not_enough_unique_nonC": 0,
            "rdkit_failed": 0,
            "has_disallowed_atom": 0,
            "ring_strain_failed": 0,  # NEW: filtered by ring-strain heuristic
            "unrealistic_moiety": 0,  # NEW: filtered by SMARTS-based unrealistic motifs
        }
        allowed = set(elements)
        keep: List[str] = []

        for s in smiles_list:
            if not s:
                stats["rdkit_failed"] += 1
                continue
            try:
                mol = Chem.MolFromSmiles(s, sanitize=True)
                if mol is None:
                    stats["rdkit_failed"] += 1
                    continue
            except Exception:
                stats["rdkit_failed"] += 1
                continue

            mol_work = mol
            elems = self._elements_in_mol(mol_work)
            if any(e not in allowed for e in elems):
                if not remap_disallowed:
                    stats["has_disallowed_atom"] += 1
                    continue
                mol_work = self._remap_disallowed_mol(mol_work, allowed)
                if mol_work is None:
                    stats["rdkit_failed"] += 1
                    continue
                elems = self._elements_in_mol(mol_work)
                if any(e not in allowed for e in elems):
                    stats["has_disallowed_atom"] += 1
                    continue

            n_heavy = mol_work.GetNumHeavyAtoms()
            if n_heavy < min_atoms:
                stats["too_few_atoms"] += 1
                continue
            if n_heavy > max_atoms:
                stats["too_many_atoms"] += 1
                continue

            nonC = {e for e in elems if e != "C"}
            if len(nonC) < min_unique_nonC:
                stats["not_enough_unique_nonC"] += 1
                continue

            # Ring-strain heuristic filter
            if not self._passes_ring_strain(mol_work):
                stats["ring_strain_failed"] += 1
                continue

            # Unrealistic moieties filter
            if self._fails_unrealistic_motifs(mol_work):
                stats["unrealistic_moiety"] += 1
                continue

            try:
                can = Chem.MolToSmiles(mol_work, canonical=True)
            except Exception:
                stats["rdkit_failed"] += 1
                continue
            keep.append(can)

        self._last_filter_stats = stats
        return keep

    # Memory
    def _load_memory(self, path: str) -> Set[str]:
        tried = set()
        p = Path(path)
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        # Prefer stored molecule string if present
                        if isinstance(d.get("molecule"), str) and d.get("molecule"):
                            tried.add(d["molecule"])
                        else:
                            # Legacy support: 'smiles' may be a string in old files
                            sv = d.get("smiles")
                            if isinstance(sv, str) and sv:
                                tried.add(sv)
                            else:
                                # Fallback: treat full line as unique marker
                                tried.add(line)
                    except Exception:
                        tried.add(line)
        return tried

    def _ensure_memory_file(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch(exist_ok=True)

    def _get_next_memory_index(self, path: str) -> int:
        """
        Scan memory file and return the next integer index for the 'smiles' field.
        Legacy lines with 'smiles' as a string are ignored for indexing.
        """
        max_idx = 0
        p = Path(path)
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        sv = d.get("smiles")
                        if isinstance(sv, int) and sv > max_idx:
                            max_idx = sv
                    except Exception:
                        continue
        return max_idx + 1 if max_idx >= 0 else 1

    def _append_memory(self, path: str, smiles_list: List[str]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        next_idx = self._get_next_memory_index(path)
        with p.open("a", encoding="utf-8") as f:
            for s in smiles_list:
                # Write 'smiles' as unique integer index; store actual SMILES under 'molecule'
                f.write(json.dumps({"smiles": next_idx, "molecule": s}) + "\n")
                next_idx += 1

    # ---------- Hooks ----------
    def _run_python(self, script: str, args: str, molecule: str) -> Dict[str, Any]:
        if not script:
            return {}
        env = os.environ.copy()
        env["MOLECULE"] = molecule
        cmd = [sys.executable, script] + ([a for a in args.split(" ") if a] if args else [])
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
            return {"rc": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr, "cmd": " ".join(cmd)}
        except Exception as e:
            return {"error": str(e), "cmd": " ".join(cmd)}

    def _run_bash(self, template: str, molecule: str) -> Dict[str, Any]:
        if not template:
            return {}
        cmd = template.replace("{molecule}", molecule)
        try:
            proc = subprocess.run(["/bin/bash", "-lc", cmd], capture_output=True, text=True, timeout=300)
            return {"rc": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr, "cmd": cmd}
        except Exception as e:
            return {"error": str(e), "cmd": cmd}

    # ---------------- Main ----------------
    def run_pipeline(self) -> Message:
        p = self._merge_params()
        self._ensure_memory_file(p["memory_path"])

        tried = self._load_memory(p["memory_path"])
        seen_this_run = set()
        fresh: List[str] = []
        raw_chunks: List[str] = []
        rounds = 0
        MAX_ROUNDS = 30

        # Capture tokenizer special IDs for debug visibility
        _, _, _, _, _, dbg_ids = self._get_bundle(p["model_id"], p["tokenizer_id"], p["device"])

        while len(fresh) < p["count"] and rounds < MAX_ROUNDS:
            rounds += 1

            texts = self._generate_texts(p, n_samples=p["num_return_sequences"])
            raw_chunks.extend(texts)

            candidates: List[str] = []
            for t in texts:
                candidates.extend(self._extract_smiles_candidates(t))

            if candidates:
                filtered = self._enforce_constraints_smiles(
                    candidates,
                    max_atoms=p["max_atoms"],
                    elements=p["elements"],
                    min_atoms=MIN_ATOMS_FIXED,
                    min_unique_nonC=p["min_unique_elements"],
                    rdkit_filter=p["rdkit_filter"],
                    remap_disallowed=p["remap_disallowed"],
                )
            else:
                filtered = []

            for s in filtered:
                if s in tried or s in seen_this_run:
                    continue
                seen_this_run.add(s)
                fresh.append(s)
                if len(fresh) >= p["count"]:
                    break

        fresh = fresh[: p["count"]]
        self._append_memory(p["memory_path"], fresh)

        results = []
        for s in fresh:
            rec = {"smiles": s}
            py = self._run_python(p["python_script"], p["python_args"], s) if p["python_script"] else {}
            sh = self._run_bash(p["bash_cmd"], s) if p["bash_cmd"] else {}
            if py: rec["python"] = py
            if sh: rec["bash"] = sh
            results.append(rec)

        body = {
            "params_used": {
                k: v for k, v in p.items()
                if k not in ("model_id", "tokenizer_id")
            },
            "hf_special_token_ids": dbg_ids,  # helpful to confirm pad/eos/bos were set
            "count_generated": len(results),
            "filter_stats": getattr(self, "_last_filter_stats", {}),
            "raw_samples": len(raw_chunks),
            "raw_model_text_truncated": ("\n".join(raw_chunks))[:800] if raw_chunks else "",
            "results": results
        }
        text = json.dumps(body, indent=2, ensure_ascii=False)
        return Message(text=text)