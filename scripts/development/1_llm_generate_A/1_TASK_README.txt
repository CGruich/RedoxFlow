Task:
Have an LLM generate a reactant A from which to calculate redox potential.

- The chemical should at least be somewhat realistic.

- The chemical is ideally generated from SELFIES (which, unlike SMILEs, guarantees chemical validity upon random generation)

- LLM should be small enough to store on local machine. OR should be directly callable from HuggingFace

- LLM should ideally be chemistry-tuned or made for chemistry in mind.

- LLM should be able to generate reactant based on:
	- Specified atom types (e.g., C, N, O, F)
	- Number of atoms per-molecule (e.g., 10)
	- Number of chemicals per request (e.g., 5)

Inputs:
	- Specified atom types
	- Number of atoms per-molecule
	- Number of chemicals per request

Outputs:
	- SELFIES representation of reactant A, which can later trivially be converted into SMILEs for downstream code.

Notes:
	- Probably have to write logic to limit atom types
	- Probably have to write logic to limit number of atoms per-molecule
	- Probably have to write logic to limit number of chemicals per request
	- Must be able to read from previously tried molecule memory (.json)

Suggestions:
	- Text-prompt policy that specifically limits the number of SELFIES tokens used
	- Base SELFIES tokens off of functional groups, not individual atoms.
	- Text-prompt policy can enforce # of atoms, # of atom types via dictionary prompt
		- e.g., {SELFIES: 'Some_Functional_Group', num_atoms: X, atom_types_in_SELFIES: ['C', 'N', 'O', 'F']}
			"Strictly generate tokens line-by-line based on max_atoms, using num_atoms to accumulate the count... etc."

Members:
Cameron

DELIVERABLE:
Either demonstrated code in LangFlow or a Jupyter Notebook, unclear which makes more sense at the moment, depends on (1) - (8)
