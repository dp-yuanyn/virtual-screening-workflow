from typing import List, Dict
import os
import json
from io import StringIO, BytesIO
import pandas as pd
from rdkit import Chem


def convert_decompose_list(decompose_list:List[Dict]) -> List:
    for decompose_item in decompose_list:
        decompose_item["dG"] = decompose_item.pop("TOTAL")
        decompose_item["dGVdw"] = decompose_item.pop("Van der Waals")
        decompose_item["dGEle"] = decompose_item.pop("Electrostatic")
        decompose_item["dGPolar"] = decompose_item.pop("Polar Solvation")
        decompose_item["dGNopolar"] = decompose_item.pop("Non-Polar Solvation")
        decompose_item["dGMM"] = decompose_item.pop("Internal")
    return decompose_list


def read_pbgbsa_results(results_dir_list:List[str]):
    results_list = []
    for results_dir in results_dir_list:
        name = os.path.basename(results_dir)
        energy_table_path = os.path.join(results_dir, "Energy.csv")
        if not os.path.exists(energy_table_path):
            print(f"Ligand {name} failed, skip")
            continue
        energy_data = pd.read_csv(energy_table_path, index_col=None).to_dict(orient="records")[0]
        pbgbsa_item = {
            "dG": energy_data.get("TOTAL"),
            "dGComplex": energy_data.get("complex"),
            "dGFreeProtein": energy_data.get("receptor"),
            "dGFreeLigand": energy_data.get("ligand"),
            "dGMM": energy_data.get("Internal"),
            "dGVdw": energy_data.get("Van der Waals"),
            "dGEle": energy_data.get("Electrostatic"),
            "dGPolar": energy_data.get("Polar Solvation"),
            "dGNopolar": energy_data.get("Non-Polar Solvation"),
            "dGGas": energy_data.get("Gas"),
            "dGSolv": energy_data.get("Solvation"),
        }
        decompose_table_path = os.path.join(results_dir, "Dec.csv")
        if os.path.exists(decompose_table_path):
            decompose_data = pd.read_csv(decompose_table_path).to_dict(orient="records")
            decompose_data = convert_decompose_list(decompose_data)
            pbgbsa_item["operation"] = decompose_data
        results_list.append((name, pbgbsa_item))
    return results_list


def find_ligand_inner_name(mol:Chem.rdchem.Mol) -> str:
    alias_list = ["hit_id", "cas", "CAS", "CdId", "cdid", "_Name"]
    for alias in alias_list:
        if mol.HasProp(alias) and mol.GetProp(alias).strip():
            return mol.GetProp(alias).strip()
    return "None"
 

def process_sdf_content(sdf_content:str) -> tuple:
    bio = BytesIO(sdf_content.encode())
    mol = next(Chem.ForwardSDMolSupplier(bio, removeHs=False))

    smiles = Chem.MolToSmiles(mol)
    name = find_ligand_inner_name(mol)
    
    return name, smiles