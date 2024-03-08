from typing import List
from pathlib import Path

from dflow.python import (
    OP, 
    OPIO, 
    Artifact, 
    Parameter,
    upload_packages,
)

if "__file__" in locals():
    upload_packages.append(__file__)


RADIUS = { "1":1.08, 
        "2":1.4, 
        "5":1.47, 
        "6":1.49,
        "7":1.41,
        "8":1.4,
        "9":1.39,
        "11":1.84,
        "12":2.05,
        "13":2.11,
        "14":2.1,
        "15":1.92,
        "16":1.82,
        "17":1.83,
        "19":2.05,
        "20":2.21,
        "25":1.97,
        "26":1.94,
        "30":2.1,
        "35":1.98,
        "53":2.23, }
VSET = -1.0


@OP.function
def gen_hbond_bpf_op(receptor_path:Artifact(Path), 
                     hbond_sites:Parameter(str), 
                     scoring_func:Parameter(str)) -> {"bias_file": Artifact(Path)}:
    from tools.constrain import HBondBias

    runner = HBondBias(receptor_file=str(receptor_path), hbond_sites=hbond_sites, scoring_function=scoring_func)

    runner.gen_hbond_bias("hbond_bias.bpf")

    return OPIO({"bias_file": Path("hbond_bias.bpf")})


@OP.function
def merge_bpf_op(bias_content_json:Artifact(Path), bpf_file:Artifact(Path)) -> {"bias_content_json": Artifact(Path)}:
    import json

    with open(bpf_file, "r") as f:
        bpf_lines = f.readlines()

    with open(bias_content_json, "r") as f:
        bias_content_list = json.load(f)

    if len(bpf_lines) > 1:
        bpf_content = "".join(bpf_lines[1:])
        for i in range(len(bias_content_list)):
            bias_content_list[i]["content"] += bpf_content

    with open("new_result_bias.json", "w") as f:
        json.dump(bias_content_list, f)

    return OPIO({"bias_content_json": Path("new_result_bias.json")})


@OP.function
def gen_substructure_bpf_op(ref_sdf_file:Artifact(Path), ind_list:Parameter(list)) -> {"bias_file": Artifact(Path)}:
    from rdkit import Chem
    from tools.constrain import AtomType

    ind_list = [int(i) for i in ind_list]
    ref_mol = Chem.SDMolSupplier(ref_sdf_file.as_posix(), removeHs=False, sanitize=True)[0]
    atom_info_map = AtomType().get_docking_atom_types(ref_mol)

    with open("substructure_bias.bpf", "w") as f:
        f.write("x y z Vset r type atom\n")
        for atom in ref_mol.GetAtoms():
            atom_idx = atom.GetIdx()
            if atom_idx in ind_list:
                atom_type = atom_info_map[atom_idx]
                atomic_num = atom.GetAtomicNum()
                atom_radius = RADIUS[str(atomic_num)]
                position = ref_mol.GetConformer().GetAtomPosition(atom_idx)
                #f.write("%6.3f %6.3f %6.3f %s %s map %s\n"%(position.x, position.y, position.z, str(Vset), str(atom_radius), atom_type))
                f.write(f'{position.x:6.3f} {position.y:6.3f} {position.z:6.3f} {VSET:6.2f} {atom_radius:6.2f} map {atom_type:<2s}\n')
    
    return OPIO({"bias_file": Path("substructure_bias.bpf")})


@OP.function
def get_substructure_ind_op(ref_sdf_file:Artifact(Path), 
                            ind_list:Parameter(list), 
                            ligand_content_json:Artifact(Path)
) -> {"meta_json": Artifact(Path)}:
    import os
    import json
    from tqdm import tqdm
    from rdkit import Chem
    from tools.stdio import StdIO, MetaIO

    ind_list = [int(i) for i in ind_list]
    ref_mol = Chem.SDMolSupplier(ref_sdf_file.as_posix(), removeHs=False, sanitize=True)[0]
    sub_mol = Chem.RWMol()
    old_to_new_atom_indices = dict()
    for idx in ind_list:
        atom = ref_mol.GetAtomWithIdx(idx)
        atom_idx = sub_mol.AddAtom(atom)
        old_to_new_atom_indices[idx] = atom_idx
    for bond in ref_mol.GetBonds():
        begin_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        if begin_atom_idx in old_to_new_atom_indices and end_atom_idx in old_to_new_atom_indices:
            begin_atom_new_idx = old_to_new_atom_indices[begin_atom_idx]
            end_atom_new_idx = old_to_new_atom_indices[end_atom_idx]
            bond_type = bond.GetBondType()
            sub_mol.AddBond(begin_atom_new_idx, end_atom_new_idx, bond_type)
    sub_mol = Chem.Mol(sub_mol)

    meta_io = MetaIO()
    substructure_match_ind_key = "substructure_match_ind"

    content_io = StdIO(json_path=ligand_content_json)
    for name, one_ligand_content_dict in tqdm(content_io.to_dict().items()):
        one_ligand_content_list = one_ligand_content_dict["contents"]
        for pose_ind, content in enumerate(one_ligand_content_list):
            target_mol = Chem.MolFromMolBlock(content, removeHs=False, sanitize=True)
            match_indices_list = target_mol.GetSubstructMatches(sub_mol)
            match_str = ""
            if match_indices_list:
                for match_indices in match_indices_list:
                    match_str += ",".join([str(i) for i in match_indices])
                    match_str += ";"
                if match_str[-1] == ";":
                    match_str = match_str[:-1]
            print(match_str)
            meta_io.add_meta(name, {substructure_match_ind_key: match_str})
    
    result_meta_path = Path("meta.json")
    meta_io.write_json(result_meta_path)
    return OPIO({"meta_json": result_meta_path})


@OP.function
def gen_shape_bpf_op(ref_sdf_file:Artifact(Path), shape_scale:Parameter(float)) -> {"bias_file": Artifact(Path)}:
    from rdkit import Chem
    from tools.constrain import AtomType

    ref_mol = Chem.SDMolSupplier(ref_sdf_file.as_posix(), removeHs=False, sanitize=True)[0]
    atom_info_map = AtomType().get_docking_atom_types(ref_mol)

    with open("shape_bias.bpf", "w") as f:
        f.write("x y z Vset r type atom\n")
        for atom in ref_mol.GetAtoms():
            atom_idx = atom.GetIdx()
            atom_type = atom_info_map[atom_idx]
            atomic_num = atom.GetAtomicNum()
            atom_radius = RADIUS[str(atomic_num)]
            position = ref_mol.GetConformer().GetAtomPosition(atom_idx)
            f.write(f'{position.x:6.3f} {position.y:6.3f} {position.z:6.3f} {VSET:6.2f} {atom_radius*shape_scale:6.2f} map {atom_type:<2s}\n')
    
    return OPIO({"bias_file": Path("shape_bias.bpf")})


@OP.function
def gen_mcs_bpf_op(ref_sdf_file:Artifact(Path), 
                   ligand_content_json:Artifact(Path)
) -> {"bias_content_json": Artifact(Path), 
      "meta_json": Artifact(Path)}:
    import os
    from tqdm import tqdm
    from functools import partial
    from multiprocessing import Pool
    from rdkit import Chem
    from tools.stdio import StdIO, MetaIO
    from tools.constrain import gen_mcs_index_and_bpf

    mcs_match_ind_key = "mcs_match_ind"
    meta_io = MetaIO()
    bias_content_io = StdIO()

    content_io = StdIO(json_path=ligand_content_json)
    name_content_list = [(name, content) for name, one_ligand_content_dict \
                         in content_io.to_dict().items() for \
                            content in one_ligand_content_dict["contents"]]
    
    ref_mol = Chem.SDMolSupplier(ref_sdf_file.as_posix(), removeHs=False, sanitize=True)[0]
    print(os.cpu_count())
    with Pool(os.cpu_count()) as pool:
        for name, mcs_ind_str, bias_content in \
            tqdm(pool.imap_unordered(partial(gen_mcs_index_and_bpf, 
                                        ref_mol=ref_mol), 
                                        name_content_list)):
            meta_io.add_meta(name, {mcs_match_ind_key: mcs_ind_str})
            bias_content_io.append_content(name, bias_content)

    result_bias_json_path = Path("result_bias.json")
    bias_content_io.write_content_json(result_bias_json_path)
    
    result_meta_path = Path("meta.json")
    meta_io.write_json(result_meta_path)
    
    return OPIO({
        "bias_content_json": result_bias_json_path, 
        "meta_json": result_meta_path,
    })