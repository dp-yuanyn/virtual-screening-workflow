from typing import List
from pathlib import Path
import os


def split_ligand_files(ligand_files:List[str], count_symbol:str="$$$$", batch_size:int=5000):
    tmp_content_list = []
    tmp_content = ""
    for ligand_file in ligand_files:
        file_name = Path(ligand_file).stem
        with open(ligand_file, "r", errors="ignore") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                tmp_content += line
                if line.strip().startswith(count_symbol):
                    tmp_content_list.append((tmp_content, file_name))
                    tmp_content = ""
                    if len(tmp_content_list) >= batch_size:
                        yield tmp_content_list
                        tmp_content_list = []
            if tmp_content:
                tmp_content_list.append((tmp_content, file_name))
    if tmp_content_list:
        yield tmp_content_list
    return


def _count_sdf_file_num_by_rdkit(ligand_file:str) -> int:
    from rdkit import Chem
    return len(Chem.SDMolSupplier(ligand_file))


def count_sdf_files_num_by_rdkit(ligand_files:List[str]) -> int:
    from multiprocessing import Pool

    with Pool(os.cpu_count()) as pool:
        count_list = pool.map(_count_sdf_file_num_by_rdkit, ligand_files)
    return sum(count_list)