Task:
Write a python function to:
Take the set of possible reduction candidates for B from script (2) and exhaustively conformer search ALL candidates. Brute force all possibilities.

Inputs:
A list of SMILEs strings representing all possible candidates that B could look like.

Outputs:
The same last of SMILEs strings that B could look like AND the XYZ geometry of the lowest-energy conformer.
e.g., [(SMILES_1, XYZ_1), 
	(SMILES_2, XYZ_2) ... ]

Suggestions:
	- Use OpenBabel (comes with MMFF94 force-field, conformer searching can be one-line potentially)
	- Change OpenBabel's default conformer search from an ML model to the brute-force option that does every possibility.
	- If any molecule fails conformer search, return None to keep track of failed molecules in the pipeline

Members:
TBD

Deliverable:
Jupyter notebook demonstration of function within same folder
