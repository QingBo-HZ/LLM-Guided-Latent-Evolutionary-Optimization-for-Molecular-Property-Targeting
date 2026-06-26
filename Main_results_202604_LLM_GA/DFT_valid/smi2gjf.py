import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

INPUT_CSV = "/root/autodl-tmp/sweeteners_evolve/Main_results_202604_LLM_GA/DFT_valid/llm_no4.csv"

OUT_DIR = "gjf_files_add"
os.makedirs(OUT_DIR, exist_ok=True)

METHOD = "B3LYP/6-31G(d)"
NPROC = 16
MEM = "16GB"

df = pd.read_csv(INPUT_CSV)

print("Rows:", len(df))

for i, row in df.iterrows():

    smiles = row["smiles"]

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    AllChem.EmbedMolecule(mol, AllChem.ETKDG())

    AllChem.UFFOptimizeMolecule(mol)

    conf = mol.GetConformer()

    mol_name = f"mol_{i:03d}"
    out_path = os.path.join(OUT_DIR, mol_name + ".gjf")

    with open(out_path, "w") as f:

        f.write(f"%chk={mol_name}.chk\n")
        f.write(f"%nprocshared={NPROC}\n")
        f.write(f"%mem={MEM}\n")
        f.write(f"#p {METHOD} opt freq\n\n")

        f.write(mol_name + "\n\n")

        f.write("0 1\n")

        for atom in mol.GetAtoms():

            pos = conf.GetAtomPosition(atom.GetIdx())

            f.write(
                f"{atom.GetSymbol():<2} {pos.x:>12.6f} {pos.y:>12.6f} {pos.z:>12.6f}\n"
            )

        f.write("\n")

    print("write:", out_path)