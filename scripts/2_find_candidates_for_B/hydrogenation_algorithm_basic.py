from rdkit import Chem
from rdkit.Chem import AllChem

reaction_templates = {
    "Alkene → Alkane": AllChem.ReactionFromSmarts("[C:1]=[C:2]>>[C:1]-[C:2]"),
    "Alkyne → Alkene": AllChem.ReactionFromSmarts("[C:1]#[C:2]>>[C:1]=[C:2]"),
    "Carbonyl (C=O) → Alcohol": AllChem.ReactionFromSmarts("[C:1]=O>>[C:1][OH]"),
    "Nitro (–NO2) → Amine": AllChem.ReactionFromSmarts("[N+:1](=O)[O-]>>[NH2:1]")
}

molecule_library = {
    "1,4-Benzoquinone": "O=C1C=CC(=O)C=C1",
    "1,4-Naphthoquinone": "O=C1C=CC(=O)c2ccccc12",
    "Anthraquinone": "O=C2c1ccccc1C(=O)c3ccccc23",
    "2,6-Dimethyl-1,4-benzoquinone": "CC1=CC(=O)C(=O)C=C1C",
    "1,4-Dimethoxybenzene": "COc1ccc(OC)cc1",
    "2,5-Di-tert-butyl-1,4-dimethoxybenzene": "COc1cc(C(C)(C)C)cc(OC)c1C(C)(C)C",
    "4-Cyanopyridine": "N#Cc1ccncc1",
    "Phenazine": "n1c2ccccn2ccc3cccnc13",
    "Nitrobenzene": "O=[N+]([O-])c1ccccc1"
}

def create_molecule_from_smiles(smiles_string):
    return Chem.MolFromSmiles(smiles_string)

def get_unique_molecule_smiles(molecule_list):
    seen_molecules, unique_molecules = set(), []
    for molecule in molecule_list:
        canonical_smiles = Chem.MolToSmiles(molecule, canonical=True)
        if canonical_smiles not in seen_molecules:
            seen_molecules.add(canonical_smiles)
            unique_molecules.append(canonical_smiles)
    return unique_molecules

for compound_name, smiles_string in molecule_library.items():
    reactant_molecule = create_molecule_from_smiles(smiles_string)
    print(f"\n{compound_name} ({smiles_string})")
    for reaction_name, reaction_object in reaction_templates.items():
        reaction_products = reaction_object.RunReactants((reactant_molecule,))
        if reaction_products:
            valid_product_molecules = []
            for product_tuple in reaction_products:
                for product_molecule in product_tuple:
                    try:
                        Chem.SanitizeMol(product_molecule)
                        valid_product_molecules.append(product_molecule)
                    except:
                        pass
            unique_product_smiles = get_unique_molecule_smiles(valid_product_molecules)
            if unique_product_smiles:
                for product_smiles in unique_product_smiles:
                    print(f"  {reaction_name}: {smiles_string} >> {product_smiles}")