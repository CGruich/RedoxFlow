Task:
Write a function that:

Takes a pre-prepared .nw input simulation script and launches an NWChem simulation, saving the simulation output to a '.out' file extension. Returns True if the simulation succeeded, returns False if it failed.

An example of a proper .nw input script and proper .out NWChem simulation output is provided for reference.

Inputs:
Pre-installed NWChem
Already prepared NWChem .nw input script

Outputs:
Simulation output printed to a .out file

Suggestions:
	- Avoid ASE, the current automation tools for auto-extracting variables that we have does not work through ASE
	- Launch the .nw input script into NWChem directly
	- Strict .out file extension

	- Run an intentionally garbage molecule simulation to see what a failed job looks like
	- Determine how to auto-identify a failed NWChem simulation from the printed output
	- For molecules that failed NWChem simulation, return False to keep track of molecules that failed in the pipeline.

Members:
TBD

Deliverable:
.py (or .py inside a bash script) file that launches the simulation, with an example .out file of a successful simulation
No need to submit the finished task in a Jupyter notebook. Just the .py (or bash) script and a successful simulation example.
