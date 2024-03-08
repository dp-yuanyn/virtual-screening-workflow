import MDAnalysis as mda
import prolif as plf
import itertools
import pandas as pd
import math
from rdkit import Chem
import openmm.app as app

cal_fp_type = [
 'HBAcceptor',
 'HBDonor',
 'Anionic',
 'Cationic',
 'CationPi',
 'PiCation',
 'PiStacking',
 'EdgeToFace',
 'FaceToFace',
 'XBAcceptor',
 'XBDonor'
]

parameters = {'HBAcceptor': {
                'DHA_angle': (120,180)
                },
              'HBDonor': {
                'DHA_angle': (120,180)
                },
              'EdgeToFace': {
                'distance': 5.5, 
                'intersect_radius': 3.0,
                'plane_angle': (60, 90)
                },
              'FaceToFace': {
                'distance': 5.5,
                'plane_angle': (0, 30) 
                },
              'CationPi': {
                'distance': 6.6,
                },
              'PiCation': {
                'distance': 6.6,
                },
              'XBDonor': {
                'distance': 3.5,
                'acceptor': "[#7,#8,P,S,Se,Te,a;!+{1-}]~[*]",
                'donor': "[#6,#7,Si,F,Cl,Br,I]-[Cl,Br,I,At]",
                'AXD_angle': (130,180),
                'XAR_angle': (80,140)
                }, 
}

def adjust_resid_mda(u, protein):
    '''adjust resid in mda universe to match openmm PDBFile'''
    # Create a mapping from mda resids to openmm resids
    resid_mapping = {mda_resid: openmm_resid.id for mda_resid, openmm_resid in zip(u.residues.resids, protein.topology.residues())}

    # Adjust mda resids
    for residue in u.residues:
        residue.resid = resid_mapping[residue.resid]

def get_ifp_df(protein_path:str, sdf_path:str):
    protein = app.PDBFile(protein_path)
    u = mda.Universe(protein)
    adjust_resid_mda(u, protein)
    protein_mol = plf.Molecule.from_mda(u)
    mol_supp = plf.sdf_supplier(sdf_path)
    fp = plf.Fingerprint(cal_fp_type, parameters=parameters)
    fp.run_from_iterable(mol_supp, protein_mol)
    df = fp.to_dataframe()
    return df


def convert_df(df:pd.DataFrame):
    df = df.copy(deep=True)
    origin_header = df.columns.to_list()
    header = [f'{tup[1].split(".")[1]}:{tup[1].split(".")[0][:3]}:{tup[1].split(".")[0][3:]}:{tup[2]}' for tup in origin_header]
    df.columns = header
    df = df.fillna(0).astype(int)
    return df

#protein_path = '7M94_protein.pdb'
#sdfs_path = 'big.sdf'
#df = get_ifp_df(protein_path, sdfs_path)
#df_ifp = convert_df(df)
#df_ifp.to_csv('out2.csv')
