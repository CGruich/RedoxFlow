Task:
Write a python function that:
Takes a successful NWChem simulation (.out file) and auto-reads the variables relevant for redox potential calculation.
Return the variables from the auto-reader through the Python function.

If no variables are found (i.e., the simulation crashes and there's nothing to get), return None in the function.

A sample auto-reader script for .out files is given for reference.
A sample .out file of a successful simulation is given for reference.

Inputs:
Successful NWChem simulation (.out)

Outputs:
Returned variables auto-extracted from the job output (see auto-reader example for variables that are returned).

Suggestions
	- Stick to the auto-reader example as much as possible, otherwise it can get pretty complicated.

Members: TBD

Deliverable:
Jupyter notebook demonstration of the function successfully returning variables after reading a .out file.
