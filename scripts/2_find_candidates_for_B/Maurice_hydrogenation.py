import json
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
import pandas as pd

# ---------------------------------------------------------------------------
# FOR NOW SILENCING THIS ERROR!!!
# ---------------------------------------------------------------------------
RDLogger.DisableLog('rdApp.warning')

# ---------------------------------------------------------------------------
# Load molecules and reactions
# ---------------------------------------------------------------------------

# For simple testing:
# with open("molecules.json") as f:
#     molecules = json.load(f)

# For big testing:
molecules = pd.read_csv("RedDBv2_reaction.tab", sep="\t")["reactant_smiles"]
molecules = molecules.to_dict()


with open("reactions.json") as f:
    reactions = json.load(f)

# ---------------------------------------------------------------------------
# List to collect all reaction results
# ---------------------------------------------------------------------------
reaction_results = []

# ---------------------------------------------------------------------------
# Process each molecule with each reaction
# ---------------------------------------------------------------------------
for compound_name, reactant_smiles in molecules.items():
    reactant_molecule = Chem.MolFromSmiles(reactant_smiles)

    # Skipping invalid SMILES
    if reactant_molecule is None:
        continue
    # Skipping invalid SMARTS
    for reaction_name, reaction_dict in reactions.items():
        reaction_object = AllChem.ReactionFromSmarts(reaction_dict["smarts"])
        if reaction_object is None:
            continue

        reaction_products = reaction_object.RunReactants((reactant_molecule,))
        seen_products = set()  # Tracking unique products
        for product_tuple in reaction_products:
            for product_molecule in product_tuple:
                product_smiles = Chem.MolToSmiles(product_molecule, canonical=True)
                # Skipping unchanged molecules and duplicates
                if product_smiles == reactant_smiles or product_smiles in seen_products:
                    continue
                seen_products.add(product_smiles)

                # If everything is alright, then we can store the reaction result
                reaction_results.append({
                    "Reactant_name": compound_name,
                    "Reactant_SMILES": reactant_smiles,
                    "Reaction_name": reaction_name,
                    "Reaction_scheme": f"{reactant_smiles} >> {product_smiles}",
                    "Product_SMILES": product_smiles
                })

# ---------------------------------------------------------------------------
# Save results to CSV
# ---------------------------------------------------------------------------
df = pd.DataFrame(reaction_results)
df.to_csv("reaction_products.csv", index=False)

