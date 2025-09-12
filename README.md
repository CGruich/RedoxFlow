# RedoxFlow

RedoxFlow is a proof-of-concept agentic workflow that auto-prepares NWChem simulation scripts for redox potential calculation. Redox potential quantifies how willing an organic chemical is to being reduced or oxidized and thus is useful in many electrochemical applications, such as:

* Batteries & Flow Batteries (e.g., anode/cathode couple selection, screening redox-active electrolytes/mediators)
* Photosensors & Bioelectrochemistry (e.g., select an organic via redox potential that avoids oxygen/hydrogen interference)
* Environmental Electrochemistry (e.g., quantify the redox potential of contaminants to predict likelihood of harmful byproducts)
* Redox-swing Separations & CO2 Capture (e.g., select redox potential to quantify the binding/unbinding window for chemical separations)
* and more (e.g., electrosynthesis, radical-based polymer chemistry, organic semiconductors, etc.)

Such a quantity is generallyA common method for computing redox potentials is the computational hydrogen electrode (CHE) method. RedoxFlow auto-prepares simulations for this method by (1) generating reactants via a foundational model, (2) predicting an array of reduced products, (3) conformer searching the reactants/products, and (4) writing the finalized molecules and associated simulation scripts for lowest-energy conformers. 


The motivation for this agent The proof-of-concept automates simulation script preparation for this redox potential method start-to-finish. The agent has persistent memory to remember generated reactants and products and filter redundant generation. 

Installation:
`cd RedoxFlow`

`mamba env create -p ../redoxflow -f ./env/redoxflow.yml`

`conda activate ../redoxflow`

Demonstration:

`RedoxFlow/scripts/agent_interface_local_simplified.ipynb`

Importance of Redox Potentials

Redox potentials are quantities that represent how easily oxidizable or reducible an organic chemical is. 
