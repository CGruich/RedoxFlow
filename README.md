# RedoxFlow: Agentic workflow for preparing simulations in high-throughput redox-potential screening 

RedoxFlow generates candidate organic molecules, proposes reduced products, and auto-prepares simulation inputs to compute thermodynamic (Nernstian) redox potentials via the **Computational Hydrogen Electrode (CHE)** method.

---

## Why Redox Potentials?
Redox potential gauges how readily a molecule is oxidized or reduced—a key lever for:
- **Batteries / Flow batteries** (electrolyte & couple selection)
- **Bio/Photoelectrochemistry** (avoid O₂/H₂ interference)
- **Environmental electrochemistry** (transformations & byproducts)
- **Redox-swing separations / CO₂ capture** (binding–unbinding windows)

CHE computes potentials from a thermodynamic cycle; RedoxFlow automates the tedious setup and extraction.

---

## TL;DR
- **Generates reactants** (_de novo_) with a lightweight foundation model (`ibm-research/GP-MoLFormer-Uniq`)
- **Predicts reduced products** (transparent rule-based POC)
- **Does conformer search** and picks lowest-energy structures
- **Writes simulation scripts** for both states and **computes** \(E = -ΔG/(zF)\)

---

## Install & Demo

```
bash
git clone <this-repo> RedoxFlow
cd RedoxFlow
mamba env create -p ../redoxflow -f env/redoxflow.yml
conda activate ../redoxflow
```

## Proof-of-Concept Restrictions

To show that the agentic workflow works start-to-finish, we restrict our agent to generate molecules and prepare simulation scripts for:
* CNOF-containing molecules
* A model reduction reaction `A + xH^+ + ye^- → B`
    * No degradation products considered
    * By nature of the model reaction, we only consider PCET-reactions.

For a round-trip demonstration of redox potential calculation with the agent/embedded class interface, we:
* Generate 10 simulations (5 reactants/5 products) and thus 5 redox potentials
    * PBE Functional
    * def2-SV(P) basis set
    * Pure water (dielectric constant ~ 78.4)
    * Room temperature
