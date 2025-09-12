# RedoxFlow

RedoxFlow is an agentic proof-of-concept that prepares redox potential simulations of solvated chemicals en-masse for electrochemical discovery.

Redox potential quantifies how willing an organic chemical is to being reduced or oxidized. It is useful in many electrochemical applications, such as:

* Batteries & Flow Batteries (e.g., anode/cathode couple selection, screening redox-active electrolytes/mediators)
* Photosensors & Bioelectrochemistry (e.g., select an organic via redox potential that avoids oxygen/hydrogen interference)
* Environmental Electrochemistry (e.g., quantify the redox potential of contaminants to predict likelihood of harmful byproducts)
* Redox-swing Separations & CO2 Capture (e.g., select redox potential to quantify the binding/unbinding window for chemical separations)

A common method for computing redox potentials is the computational hydrogen electrode (CHE) method. Using this method often amounts to calculating a thermocycle. The project premise is that an agentic AI can prepare the simulation scripts for this method en-masse for large-scale chemical search and discovery.

Such a thermocycle is arduous by-hand and not conducive to high-throughput chemistry workflows. 
For a model reduction reaction `A + xH^+ + ye^- → B`, a researcher typically must: 

* (1) Prepare a simulation script for `A`
* (2) Prepare a simulation script for `B`
* (3) Conformer search `A` and `B` to determine reasonable initial geometries for the molecules
* (4) Prepare a simulation script for `H~2` as a proxy for `H^+` and `e^-` as allowed by the CHE method
* (5) Prepare some flavor of solvation correction
* (6) Run the simulations
* (7) Extract the relevant variables from simulations
* (8) Calculate the redox potential `E = -ΔG/zF`, where `E` is redox potential, `ΔG` is solvated free energy of reaction derived from simulations, `z` is the number of participating electrons, and `F` is Faraday's constant.

 RedoxFlow automates simulation preparation (i.e., steps (1)-(5)). The agent is wrapped in an interface that automates calculation steps (7), (8).

RedoxFlow auto-prepares simulation scripts by:
* (1) Generating reactants via a foundational model (ibm-research/GP-MoLFormer-Uniq) designed for _de novo_ generation of molecules
* (2) Predicting an array of reduced products (in our proof-of-concept, using transparent reaction rules)
* (3) Conformer searching the reactants/products
* (4) Writing the finalized molecules and associated simulation scripts for lowest-energy conformers 

Through a function-based interface, the researcher has access to several knobs that control the chemical diversity of generated reactant:
*Number of generated reactant candidates (`num_generated_candidates`)
*Maximum non-H atoms per generated reactant (`max_heavy_atoms_per_reactant`)
*Minimum number of non-carbon elements per generated reactant (`min_uniq_elements`)
*Electronegative presence/token volatility in generated reactants (`temperature`)

Upon first-use, the agent creates persistent memory to remember generated reactants (`RedoxFlow/memory/reactants_memory.csv`) and products (`RedoxFlow/memory/product_memory.csv`) to filter redundantly generated chemicals. Reaction rules (`RedoxFlow/scripts/reactions.json`) can be transparently adjusted to tailor predicted products. The agent is loaded directly from `redoxflow.json` and thus can be deployed locally or on HPC filesystems to prepare simulations.

Installation:
`cd RedoxFlow`

`mamba env create -p ../redoxflow -f ./env/redoxflow.yml`

`conda activate ../redoxflow`

Demonstration:

`RedoxFlow/scripts/agent_interface_local_simplified.ipynb`

Importance of Redox Potentials

Redox potentials are quantities that represent how easily oxidizable or reducible an organic chemical is. 
