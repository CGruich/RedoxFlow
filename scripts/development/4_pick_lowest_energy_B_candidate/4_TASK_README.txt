Task:
Write a function that:

Given a list of molecules in the form of:
[(SMILES_1, XYZ_1), 
	(SMILES_2, XYZ_2), 
	(SMILES3, XYZ_3), ...]

Find the thermodynamically most likely (i.e., lowest energy) molecule via cheap force-field energy minimization.
Return the lowest energy structure in the form of [(SMILES_LOWEST_ENERGY, XYZ_LOWEST_ENERGY)]

Inputs:
[(SMILES_1, XYZ_1), 
        (SMILES_2, XYZ_2), 
        (SMILES3, XYZ_3), ...]

Outputs:
One molecule in the format
[(SMILES_LOWEST_ENERGY, XYZ_LOWEST_ENERGY)]

Suggestions:
	- Use OpenBabel MMFF94 force field on inputs (cheap, should run in one-line, validated against empirical data)

Members:
TBD

Deliverable:
Jupyter notebook demonstration of function within the same folder
