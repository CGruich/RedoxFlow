Task:

Take the variables auto-extracted from the NWChem simulation output for both A and B and calculate redox potential via thermocycle/algebraic calculations. Return the value

An example redox potential calculator function is provided.
An example set of variables auto-retrieved from simulation for a reactant A and a product B is provided.

Notes:
	- Try to stick to the calculator given, otherwise it can get pretty complicated
	- Note that H2 used in the calculation is equivalent to 2H+ + 2e- (because CHE method)
		- If two hydrogens were added to A to produce B, then you use H2 in the thermocycle algebra
		- If one hydrogen was added to A to produce B, then you use 1/2 H2 in the thermocycle algebra
		- If three hydrogens were added to A to produce B, then you use 3/2 H2 in the thermocycle algebra
		- ETC.

Inputs:
Redox potential-relevant variables for thermocycle calculation for both A and B.

Outputs:
A single redox potential number (V vs. SHE) returned by the function.

Members:
TBD

Deliverable:
A Jupyter notebook within the same folder demonstrating the function used successfully for one redox potential calculation.
Within the same notebook, another example of the function used successfully but this time for a different amount of hydrogen participating in the reaction (just the same calculation but with different stoichiometry).
