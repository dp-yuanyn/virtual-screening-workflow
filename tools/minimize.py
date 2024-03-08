import os
import traceback
import time
from io import BytesIO, StringIO
from multiprocessing import Pool
from functools import partial
from rdkit import Chem
from rdkit.Chem.rdchem import AtomPDBResidueInfo
from rdkit.Chem import rdmolops
from rdkit.Chem import ChemicalForceFields
from rdkit.Geometry import Point3D
import MDAnalysis as mda
from openmm import app


HDONER = "[$([O,S;+0]),$([N;$(Na),$(NC=[O,S]);H2]),$([N;$(N[S,P]=O)]);!H0]"
UNWANTED_H = "[#1;$([#1][N;+1;H2]),$([#1][N;!H2]),$([#1][#6])]"


def check_ligand(sdf_content:str, pattern:Chem.rdchem.Mol) -> bool:
    try:
        bio = BytesIO(bytes(sdf_content, encoding="utf-8"))
        mol = next(Chem.ForwardSDMolSupplier(bio, removeHs=False))
        match = mol.HasSubstructMatch(pattern)
        return match is not None
    except:
        print(f'check ligand error: {traceback.format_exc()}')
        return False


def constrain_minimize(mol:Chem.rdchem.Mol, constrain_list:list[int]):
    ff_property = ChemicalForceFields.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')
    ff = ChemicalForceFields.MMFFGetMoleculeForceField(mol, ff_property, confId=0, ignoreInterfragInteractions=False)

    for query_atom_idx in constrain_list:
        ff.MMFFAddPositionConstraint(query_atom_idx, 0.0, 1000)

    ff.Initialize()

    max_minimize_iteration = 5
    for _ in range(max_minimize_iteration):
        minimize_seed = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
        if minimize_seed == 0:
            break
    
    return mol


def get_polar_hydrogen_list(mol:Chem.rdchem.Mol, pattern, unwated_pattern):
    wanted_hydrogen_id_list = []
    polar_hydrogen_list = []
    matches = mol.GetSubstructMatches(pattern)
    for match in matches:
        atom = mol.GetAtomWithIdx(match[0])
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 1:
                polar_hydrogen_list.append(neighbor.GetIdx())
    unwated_matches = mol.GetSubstructMatches(unwated_pattern)
    for i in polar_hydrogen_list:
        if i not in [j[0] for j in unwated_matches]:
            wanted_hydrogen_id_list.append(i)
    
    return wanted_hydrogen_id_list


def mol_to_content(mol:Chem.rdchem.Mol) -> str:
    sio = StringIO()
    with Chem.SDWriter(sio) as writer:
        writer.write(mol)
    return sio.getvalue()


def get_pocket_mol(mol, protein_universe, pattern, unwanted_H_pattern):
    mol_pdb_str = Chem.MolToPDBBlock(mol)
    ligand_universe = mda.Universe(StringIO(mol_pdb_str), format='pdb')
    merge_pdb = mda.Merge(ligand_universe.atoms, protein_universe.atoms)
    minimize_atom_idx_list = get_polar_hydrogen_list(mol, pattern, unwanted_H_pattern)
    if len(minimize_atom_idx_list) == 0:
        return None
    else:
        minimize_atom_string = ' '.join([str(idx) for idx in minimize_atom_idx_list])
        pro_pocket = merge_pdb.select_atoms(f'byres protein and around 4.0 (index {minimize_atom_string})') #around H or heavy atoms?
        if len(pro_pocket) == 0:
            return None
        if len(pro_pocket) > 0:
            mda_to_rdkit = mda._CONVERTERS['RDKIT']().convert
            pro_pocket_mol = mda_to_rdkit(pro_pocket)
            Chem.GetSymmSSSR(pro_pocket_mol)
            pro_pocket_mol.UpdatePropertyCache(strict=False)
            return pro_pocket_mol


def set_pro_lig_name(pro_pocket_mol, mol):
    for atom in pro_pocket_mol.GetAtoms():
        atom.GetMonomerInfo().SetResidueName("PRO")

    for atom in mol.GetAtoms():
        monomer_info = atom.GetMonomerInfo()

        # If there is no monomer information, create it
        if monomer_info is None:
            # Create a new residue info object with the desired residue name
            residue_info = AtomPDBResidueInfo()
            residue_info.SetResidueName("LIG")
            atom.SetMonomerInfo(residue_info)
        else:
            monomer_info.SetResidueName("LIG")
    complex = rdmolops.CombineMols(mol, pro_pocket_mol)
    Chem.GetSymmSSSR(complex) # trick to avoid SSSR error
    complex.UpdatePropertyCache(strict=False)
    return complex


def get_complex_constrain_list(complex, unwanted_H_pattern):
    constrain_list = []        
    unwanted_H_atom_idx_list = list(complex.GetSubstructMatches(unwanted_H_pattern))
    unwanted_H_atom_idx_list = [unwanted_H_atom_idx_tuple[0] for unwanted_H_atom_idx_tuple in unwanted_H_atom_idx_list]
    for atom in complex.GetAtoms():
        if atom.GetMonomerInfo().GetResidueName() == "PRO":
            constrain_list.append(atom.GetIdx())
        if atom.GetMonomerInfo().GetResidueName() == "LIG":
            if atom.GetSymbol() != "H":
                constrain_list.append(atom.GetIdx())
            if atom.GetIdx() in unwanted_H_atom_idx_list:
                constrain_list.append(atom.GetIdx())
    
    return constrain_list


def get_ligand_constrain_list(mol, unwanted_H_pattern):
    constrain_list = []
    unwanted_H_atom_idx_list = list(mol.GetSubstructMatches(unwanted_H_pattern))
    unwanted_H_atom_idx_list = [unwanted_H_atom_idx_tuple[0] for unwanted_H_atom_idx_tuple in unwanted_H_atom_idx_list]
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "H":
            constrain_list.append(atom.GetIdx())
        if atom.GetIdx() in unwanted_H_atom_idx_list:
            constrain_list.append(atom.GetIdx())
    
    return constrain_list


def get_minimized_ligand_content(complex_mol, ori_mol):
    coord_dict = {}
    for atom in complex_mol.GetAtoms():
        if atom.GetMonomerInfo().GetResidueName() == "LIG":
            coord_dict[atom.GetIdx()] = complex_mol.GetConformer().GetAtomPosition(atom.GetIdx())
            
    lig_conf = Chem.Conformer()
    for idx in range(len(ori_mol.GetAtoms())):
        atom_coords = coord_dict[idx]
        atom_coords_point_3D = Point3D(atom_coords[0], atom_coords[1], atom_coords[2])
        lig_conf.SetAtomPosition(idx, atom_coords_point_3D)

    ori_mol.RemoveAllConformers()
    ori_mol.AddConformer(lig_conf)

    return mol_to_content(ori_mol)


def minimize_ligand(sdf_content, protein_universe, pattern, unwanted_H_pattern) -> str:
    try:
        bio = BytesIO(bytes(sdf_content, encoding="utf-8"))
        mol = next(Chem.ForwardSDMolSupplier(bio, removeHs=False))
        pro_pocket_mol = get_pocket_mol(mol, protein_universe, pattern, unwanted_H_pattern)
        if pro_pocket_mol is None:
            # do nothing
            #constrain_list = get_ligand_constrain_list(mol, unwanted_H_pattern)
            #mol = constrain_minimize(mol, constrain_list)
            return mol_to_content(mol)

        complex = set_pro_lig_name(pro_pocket_mol, mol)
        constrain_list = get_complex_constrain_list(complex, unwanted_H_pattern)
        try:
            complex = constrain_minimize(complex, constrain_list)
        except:
            print(f'constrain minimization complex error: {traceback.format_exc()}')
        return get_minimized_ligand_content(complex, mol)
    except:
        print(f'minimize ligand error: {traceback.format_exc()}')
        return sdf_content


def minimize_batch(pdbfile:str, ligand_contents:list[str]):
    pattern = Chem.MolFromSmarts(HDONER)
    unwanted_H_pattern = Chem.MolFromSmarts(UNWANTED_H)
    try:
        pdbfile = app.PDBFile(pdbfile)
        protein_universe = mda.Universe(pdbfile)
    except:
        print(f'protein error: {traceback.format_exc()}')
        return []

    # process ligands to check if they have hydrogen bond donor need to be minimized
    start = time.time()    
    with Pool(os.cpu_count()) as pool:
        check_list = pool.map(partial(check_ligand, pattern=pattern), ligand_contents)
    end = time.time()
    print(f'check ligand time: {end-start}')

    running_ind_list = [i for i in range(len(ligand_contents)) if check_list[i]]
    running_content_list = [ligand_contents[i] for i in range(len(ligand_contents)) if check_list[i]]

    # minimize ligands that need to be minimized
    start = time.time()
    with Pool(os.cpu_count()) as pool:
        minimized_content_list = pool.map(partial(minimize_ligand, 
                                                  protein_universe=protein_universe, 
                                                  pattern=pattern, 
                                                  unwanted_H_pattern=unwanted_H_pattern), 
                                                  running_content_list)
    end = time.time()
    print(f'minimize ligand time: {end-start}')
    
    assert len(minimized_content_list) == len(running_ind_list), "minimized results length not match"

    for running_ind, ori_ind in enumerate(running_ind_list):
        ligand_contents[ori_ind] = minimized_content_list[running_ind]
    
    return ligand_contents
