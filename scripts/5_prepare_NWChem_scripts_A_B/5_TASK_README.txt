Task:
Write a python function (or class) that:

Produces an output text file (.nw extension) that formats a chemical into a template redox potential simulation script.
The simulation script is constructed the same for any given chemical.

A sample redox script template class is given to provide a frame of reference for making the scripts automatically.
A sample redox simulation script is given to show you what the final outputted .nw script should look like.

This simulation script will need to specify:
	- Charge
	- A geometry file (.xyz)
	- A basis set
	- Maximum number of iterations to optimize the calculation
	- Exchange-correlation functional
	- Spin multiplicity
	- 'Grid', or coarseness of the DFT simulation
	- Dispersion correctly
	
	And for implicit solvation step will need to specify:
	- Dielectric constant
	- minbem setting
	- ificos setting
	- do_gasphase setting

If this looks confusing, reference the sample .nw script to see how all these settings are set.

Note that different molecules may have different:
	- Charge (can auto-calculate from SMILEs with RDKit)
	- Spin multiplicity (can auto-calculate from SMILEs with RDKit)

Inputs:
	Molecule .xyz file
	SMILEs (either directly or retrieved by RDKit SMILES -> XYZ conversion)

Outputs:
	.nw input simulation script that matches the example template

Members:
TBD

Deliverables:
Jupyter notebook of class (or function) demonstrated within the same folder.
