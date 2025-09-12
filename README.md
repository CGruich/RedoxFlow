# RedoxFlow: Agentic workflow for preparing simulations in high-throughput redox-potential screening 

RedoxFlow generates candidate organic molecules, proposes reduced products, and auto-prepares simulation inputs to compute thermodynamic (Nernstian) redox potentials via the **Computational Hydrogen Electrode (CHE)** method.

---

## Why Redox Potentials?
Redox potential quantifies how readily a molecule is oxidized or reduced. Mapping **E** across chemical space is a fast screener of **thermodynamic driving force** (via ΔG = −zFE): it lets you quickly rank candidates, estimate feasible cell voltages, and check compatibility with solvent/electrolyte stability windows and pH before doing any heavy kinetic/mechanistic work. For PCET steps, CHE also exposes the pH-dependence (Nernst slope), enabling screening across operating conditions.

Redox potentials are especially useful in:
- **Batteries / Flow batteries:** target voltage windows (∆E ≈ E_cath − E_an) while respecting solvent/electrolyte stability.
- **Bio/Photoelectrochemistry:** choose mediators/analytes whose E minimizes O₂/H₂ interference and matches desired driving force.
- **Environmental electrochemistry:** assess spontaneity/selectivity along the redox ladder for contaminant transformations.
- **Redox-swing separations / CO₂ capture:** tune E to set binding–unbinding windows and energy efficiency.

CHE computes potentials from a thermodynamic cycle; RedoxFlow automates the setup, extraction, and calculation.

---

## TL;DR
- **Generates reactants** (_de novo_) with a lightweight foundation model (`ibm-research/GP-MoLFormer-Uniq`)
- **Predicts reduced products** (transparent rule-based POC)
- **Does conformer search** and picks lowest-energy structures
- **Writes simulation scripts** for both states and **computes** \(E = -ΔG/(zF)\) for completed simulations

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
