from typing import List
from pathlib import Path

from dflow.python import (
    OP, 
    OPIO, 
    Artifact,
    upload_packages,
)

if "__file__" in locals():
    upload_packages.append(__file__)


def read_ligand_str_wrap(content_fname_tuple):
    from io import BytesIO
    from rdkit import Chem
    content, filename = content_fname_tuple
    mols = []
    try:
        mol = next(Chem.ForwardSDMolSupplier(BytesIO(content.encode()), removeHs=False, strictParsing=False))
        mol.SetProp("name", filename)
        mols.append(mol)
    except:
        mols.append(None)
    return mols


def get_mol_content(mol):
    import re
    from io import StringIO
    from rdkit import Chem

    label_value = -1
    mol_name = re.sub(r"[\s]", "", mol.GetProp("_Name") if mol.HasProp("_Name") else "")
    file_name = mol.GetProp("name", "") if mol.HasProp("name") else ""
    if not mol_name:
        mol_name = file_name
    if file_name.startswith("active"):
        label_value = 1
    elif file_name.startswith("inactive"):
        label_value = 0
    label_str = {-1: "unknown", 0: "inactive", 1: "active"}[label_value]
    mol.SetProp("active_label", label_str)

    sio = StringIO()
    with Chem.SDWriter(sio) as writer:
        writer.write(mol)
    content_list = [sio.getvalue()]
    return (mol_name, label_value, content_list)


@OP.function
def read_ligands_op(
        ligands_dir: Artifact(Path),
        label_table: Artifact(Path, optional=True),
        batch_size: int = 18000,
) -> {"ligands_json_list": Artifact(List[Path], archive=None),
      "label_json": Artifact(Path, optional=True)}:
    import os
    import glob
    import time
    import re
    import math
    from multiprocessing import Pool
    from functools import partial
    import pandas as pd
    from tools.stdio import MetaIO, StdIO
    from tools.read_ligand import count_sdf_files_num_by_rdkit, split_ligand_files

    sdf_files = glob.glob(os.path.join(ligands_dir, "**", "*.sdf"), recursive=True)
    start_time = time.time()
    total_num = count_sdf_files_num_by_rdkit(sdf_files)
    print(f"Count ligand num {total_num}, time {time.time() - start_time} seconds")
    real_batch_size = math.ceil(total_num / math.ceil(total_num / batch_size))
    print(f"Real batch size: {real_batch_size}")

    label_map = dict()
    content_json_list = []
    for ind, sdf_content_fname_list in enumerate(split_ligand_files(sdf_files, batch_size=real_batch_size)):
        print(f"batch {ind}")
        stdio = StdIO()
        start_time = time.time()
        with Pool(os.cpu_count()) as pool:
            mols = pool.map(read_ligand_str_wrap, sdf_content_fname_list)
        mols = sum(mols, [])
        mols = [mol for mol in mols if mol]
        print(f"num mols after sanitize a batch: {len(mols)}")
        start_time = time.time()
        with Pool(os.cpu_count()) as pool:
            for mol_name, label_value, content_list in pool.imap_unordered(get_mol_content, mols):
                label_map[mol_name] = label_value
                stdio.insert_records({mol_name: {"contents": content_list}})
        one_json_path = f"ligands_{ind}.json"
        stdio.write_content_json(one_json_path)
        content_json_list.append(Path(one_json_path))
        print(f"Get SDF Content Time a Batch:{time.time()-start_time}")

    if label_table:
        label_df = pd.read_csv(label_table)
        if "name" in label_df.columns and "label" in label_df.columns:
            for _, row in label_df.iterrows():
                label_map[str(row["name"])] = int(row["label"])

    label_json_path = None
    if label_map:
        start_time = time.time()
        meta_io = MetaIO()
        for name, label in label_map.items():
            meta_io.add_meta(name, {"name": name, "label": label})
        label_json_path = Path("label.json")
        meta_io.write_json(label_json_path)
        print(f"Write Meta Time:{time.time()-start_time}")

    return OPIO({"ligands_json_list": content_json_list, 
                 "label_json": label_json_path})