from rdkit import Chem
import sys

infile = sys.argv[1]

for i, mol in enumerate(Chem.SDMolSupplier(infile)):
    if mol is None:
        smi = "NA"
    else:
        smi = Chem.MolToSmiles(mol)
    with open(sys.argv[1].replace(".sdf", ".smi"), "a") as f:
        f.write(f"{smi} {i}\n")
    if i % 10000 == 0:
        print(i)
