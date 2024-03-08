from typing import List, Dict, Tuple
import os
import math
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFMCS
from Bio.PDB import PDBParser


class HBondBias:
    def __init__(self, receptor_file:str, hbond_sites:str, scoring_function:str):
        self.receptor_file = receptor_file
        self.hbond_sites = hbond_sites
        HBOND_DIST = 1.9
        HBOND_DIST_VINA = 2.9
        if scoring_function == "ad4":
            HBOND_DIST_VINA = 1.9
        self.ideal_interactions = self._get_ideal_interactions(HBOND_DIST, HBOND_DIST_VINA)
        self.ideal_interactions_bb = self._get_ideal_interactions_bb(HBOND_DIST, HBOND_DIST_VINA)
    
    def _get_ideal_interactions(self, hbond_dist:float, hbond_dist_vina:float):

        ''' Returns a list of lists with ideal H bond interaction definitions:
            distance, angle, diherdal.
        '''

        # side chain interactions (acceptors)
        # sintax: [resname,[atom1,atom2,atom3],[distance,angle,dihedral],type]
        sc_acceptors = [['GLU',['CG','CD','OE1'],[hbond_dist_vina,180,0],'acc'],
                        ['GLU',['CG','CD','OE1'],[hbond_dist_vina,210,0],'acc'],
                        ['GLU',['CG','CD','OE1'],[hbond_dist_vina,240,0],'acc'],
                        ['GLU',['CG','CD','OE1'],[hbond_dist_vina,150,0],'acc'],
                        ['GLU',['CG','CD','OE1'],[hbond_dist_vina,120,0],'acc'],

                        ['GLU',['CG','CD','OE2'],[hbond_dist_vina,180,0],'acc'],
                        ['GLU',['CG','CD','OE2'],[hbond_dist_vina,210,0],'acc'],
                        ['GLU',['CG','CD','OE2'],[hbond_dist_vina,240,0],'acc'],
                        ['GLU',['CG','CD','OE2'],[hbond_dist_vina,150,0],'acc'],
                        ['GLU',['CG','CD','OE2'],[hbond_dist_vina,120,0],'acc'],

                        ['ASP',['CB','CG','OD1'],[hbond_dist_vina,180,0],'acc'],
                        ['ASP',['CB','CG','OD1'],[hbond_dist_vina,210,0],'acc'],
                        ['ASP',['CB','CG','OD1'],[hbond_dist_vina,240,0],'acc'],
                        ['ASP',['CB','CG','OD1'],[hbond_dist_vina,150,0],'acc'],
                        ['ASP',['CB','CG','OD1'],[hbond_dist_vina,120,0],'acc'],

                        ['ASP',['CB','CG','OD2'],[hbond_dist_vina,180,0],'acc'],
                        ['ASP',['CB','CG','OD2'],[hbond_dist_vina,210,0],'acc'],
                        ['ASP',['CB','CG','OD2'],[hbond_dist_vina,240,0],'acc'],
                        ['ASP',['CB','CG','OD2'],[hbond_dist_vina,150,0],'acc'],
                        ['ASP',['CB','CG','OD2'],[hbond_dist_vina,120,0],'acc'],

                        ['ASN',['CB','CG','OD1'],[hbond_dist_vina,180,0],'acc'],
                        ['ASN',['CB','CG','OD1'],[hbond_dist_vina,210,0],'acc'],
                        ['ASN',['CB','CG','OD1'],[hbond_dist_vina,240,0],'acc'],
                        ['ASN',['CB','CG','OD1'],[hbond_dist_vina,150,0],'acc'],
                        ['ASN',['CB','CG','OD1'],[hbond_dist_vina,120,0],'acc'],

                        ['GLN',['CG','CD','OE1'],[hbond_dist_vina,180,0],'acc'],
                        ['GLN',['CG','CD','OE1'],[hbond_dist_vina,210,0],'acc'],
                        ['GLN',['CG','CD','OE1'],[hbond_dist_vina,240,0],'acc'],
                        ['GLN',['CG','CD','OE1'],[hbond_dist_vina,150,0],'acc'],
                        ['GLN',['CG','CD','OE1'],[hbond_dist_vina,120,0],'acc'],

                        ['SER',['CA','CB','OG'], [hbond_dist_vina,109.5,'alpha'],'acc'],
                        ['SER',['CA','CB','OG'], [hbond_dist_vina,109.5,'beta'], 'acc'],

                        ['THR',['CA','CB','OG1'],[hbond_dist_vina,109.5,'alpha'],'acc'],
                        ['THR',['CA','CB','OG1'],[hbond_dist_vina,109.5,'beta'], 'acc'],

                        ['TYR',['CE1','CZ','OH'],[hbond_dist_vina,109.5,'alpha'],'acc'],
                        ['TYR',['CE1','CZ','OH'],[hbond_dist_vina,109.5,'beta'], 'acc'],

                        ['HID',['CG','CD2','NE2'], [hbond_dist_vina,125.4,180],'acc'],
                        ['HIE',['NE2','CE1','ND1'],[hbond_dist_vina,125.4,180],'acc'],
                        
                        ['HOH',['H1','H2','O'],    [hbond_dist_vina,109.5,120],'acc'],
                        ['HOH',['H1','H2','O'],    [hbond_dist_vina,109.5,240],'acc'],
                        ['WAT',['H1','H2','O'],    [hbond_dist_vina,109.5,120],'acc'],
                        ['WAT',['H1','H2','O'],    [hbond_dist_vina,109.5,240],'acc']]

        # side chain interactions (donors)
        # sintax: [resname,[atom1,atom2,atom3],[distance,angle,dihedral],type]
        sc_donors = [['ASN',['CG','ND2','HD21'],[hbond_dist,180,0],'don'],
                    ['ASN',['CG','ND2','HD22'],[hbond_dist,180,0],'don'],

                    ['GLN',['CD','NE2','HE21'],[hbond_dist,180,0],'don'],
                    ['GLN',['CD','NE2','HE22'],[hbond_dist,180,0],'don'],

                    ['ARG',['CZ','NE','HNE'],  [hbond_dist,180,0],'don'],
                    ['ARG',['CZ','NH1','HH11'],[hbond_dist,180,0],'don'],
                    ['ARG',['CZ','NH1','HH12'],[hbond_dist,180,0],'don'],
                    ['ARG',['CZ','NH2','HH21'],[hbond_dist,180,0],'don'],
                    ['ARG',['CZ','NH2','HH22'],[hbond_dist,180,0],'don'],
                    ['ARG',['CZ','NE','HE'],   [hbond_dist,180,0],'don'],

                    ['SER',['CB','OG','HG'],[hbond_dist,180,0],'don'],

                    ['THR',['CB','OG1','HG1'],[hbond_dist,180,0],'don'],

                    ['TYR',['CZ','OH','HH'],[hbond_dist,180,0],'don'],

                    ['TRP',['CD1','NE1','HE1'],[hbond_dist,180,0],'don'],

                    ['LYS',['CE','NZ','HZ1'],[hbond_dist,180,0],'don'],
                    ['LYS',['CE','NZ','HZ2'],[hbond_dist,180,0],'don'],
                    ['LYS',['CE','NZ','HZ3'],[hbond_dist,180,0],'don'],

                    ['HIE',['CE1','NE2','HE2'],[hbond_dist,180,0],'don'],
                    ['HID',['CE1','ND1','HD1'],[hbond_dist,180,0],'don'],
                    ['HIP',['CE1','NE2','HE2'],[hbond_dist,180,0],'don'],
                    ['HIP',['CE1','ND1','HD1'],[hbond_dist,180,0],'don'],
                    
                    ['HOH',['H1','O','H2'],[hbond_dist,180,0],'don'],
                    ['HOH',['H2','O','H1'],[hbond_dist,180,0],'don'],
                    ['WAT',['H1','O','H2'],[hbond_dist,180,0],'don'],
                    ['WAT',['H2','O','H1'],[hbond_dist,180,0],'don'],]

        lists = sc_acceptors + sc_donors
        
        return lists

    # define ideal H bond interactions for protein residues (backbone)
    def _get_ideal_interactions_bb(self, hbond_dist:float, hbond_dist_vina:float):
    
        ''' Returns a list of lists with ideal H bond interaction definitions:
            distance, angle, diherdal.
        '''

        # backbone interactions (carbonyl)
        bb_acceptors = [['XXX',['CA','C','O'],[hbond_dist_vina,180,0],'acc'],
                        ['XXX',['CA','C','O'],[hbond_dist_vina,210,0],'acc'],
                        ['XXX',['CA','C','O'],[hbond_dist_vina,240,0],'acc'],
                        ['XXX',['CA','C','O'],[hbond_dist_vina,150,0],'acc'],
                        ['XXX',['CA','C','O'],[hbond_dist_vina,120,0],'acc']]

        # backbone interactions (NH) -Pro excluded-
        bb_donors = [['XXX',['CA','N','H'],[hbond_dist,180,0],'don']]
        
        lists = bb_acceptors + bb_donors
        
        return lists

    # get dihedral angle between 4 points defining 2 planes
    def get_dihedral(self, p1, p2, p3, p4):

        v1 = -1.0*(p2 - p1)
        v2 = p3 - p2
        v3 = p4 - p3

        # normalize v2 so that it does not influence magnitude of vector
        # rejections that come next
        v2 /= np.linalg.norm(v2)

        # vector rejections
        # v = projection of v1 onto plane perpendicular to v2
        #   = v1 minus component that aligns with v2
        # w = projection of v3 onto plane perpendicular to v2
        #   = v3 minus component that aligns with v2
        v = v1 - np.dot(v1, v2)*v2
        w = v3 - np.dot(v3, v2)*v2

        # angle between v and w in a plane is the torsion angle
        # v and w may not be normalized but that's fine since tan is y/x
        x = np.dot(v, w)
        y = np.dot(np.cross(v2, v), w)
        
        return np.degrees(np.arctan2(y, x))


    # get 4th atom position given distance, angle and dihedral
    def pos4(self, x1, y1, z1, x2, y2, z2, x3, y3, z3, r4, a4, d4):
        
        """
        Returns x,y,z coordinates of point 4 satisfying:
        - distance between points 3-4 = r4
        - angle between points 2-3-4 = a4
        - dihedral between points 1-2-3-4 = d4
        """
        
        xejx = (y3-y2)*(z1-z2) - (z3-z2)*(y1-y2)
        yejx = -1 * ((x3-x2)*(z1-z2) - (z3-z2)*(x1-x2))
        zejx = (x3-x2)*(y1-y2)-(y3-y2)*(x1-x2)
        
        rejx = math.sqrt(math.pow(xejx,2) + math.pow(yejx,2) + math.pow(zejx,2))
        
        l1 = xejx/rejx
        
        r23 = math.sqrt(math.pow(x3-x2,2) + math.pow(y3-y2,2) + math.pow(z3-z2,2))
        m1 = yejx/rejx
        n1 = zejx/rejx
        
        l2 = (x3-x2)/r23
        m2 = (y3-y2)/r23
        n2 = (z3-z2)/r23
        
        xejz = yejx*(z3-z2) - zejx*(y3-y2)
        yejz = -1 * (xejx*(z3-z2) - zejx*(x3-x2))
        zejz = xejx*(y3-y2) - yejx*(x3-x2)
        
        rejz = math.sqrt(math.pow(xejz,2) + math.pow(yejz,2) + math.pow(zejz,2))
        
        l3 = xejz/rejz
        m3 = yejz/rejz
        n3 = zejz/rejz
        
        d4 = d4*math.pi/180
        a4 = 180-a4
        a4 = a4*math.pi/180
        
        z = r4 * math.sin(a4) * math.cos(d4)
        x = r4 * math.sin(a4) * math.sin(d4)
        y = r4 * math.cos(a4)
        
        y = y + r23
        
        x4 = l1*x + l2*y + l3*z + x2
        y4 = m1*x + m2*y + m3*z + y2
        z4 = n1*x + n2*y + n3*z + z2

        return (x4, y4, z4)

    # get a line with bpf file format
    def write_bpf_format(self, htype:str, coordinates:List[float], 
                         vset:float=-1.5, r:float=2.0):
        x = coordinates[0]
        y = coordinates[1]
        z = coordinates[2]
        bpf_line = f'{x:6.3f} {y:6.3f} {z:6.3f} {vset:6.2f} {r:6.2f} {htype:3s}'
        return bpf_line
 
    def gen_hbond_bias(self, out_bias_file:str):
        parser = PDBParser(PERMISSIVE=1)
        structure = parser.get_structure("protein", self.receptor_file)
        model = structure[0]

        os.makedirs(os.path.dirname(os.path.abspath(out_bias_file)), exist_ok=True)

        bpf_lines = list()
        count = 1
        for hb in self.hbond_sites.split(","):
            chainid, resid, atomname = hb.split(":")
            # get the chain	
            chain = model[chainid]
            try:
                # raises error if it is water
                insertion_code = ' '
                if resid[-1].isalpha():
                    insertion_code = resid[-1]     
                    resid = resid[:-1]   
                residue = chain[' ', int(resid), insertion_code]
            except:
                # for water residues
                residue = chain['W', int(resid), ' ']
            # get residue name
            resname = residue.get_resname()

            for atom_ in residue:
                if atom_.get_name() == atomname:
                    atom = atom_
                    break
                
            # side chain HB interactions
            for interaction in self.ideal_interactions:    
                # if atom is interactor according to ideal interactions previously defined
                if resname == interaction[0] and atomname == interaction[1][2]:
                    # get coordinates of the three atoms defining the interaction
                    coords1 = residue[interaction[1][0]].get_vector()
                    coords2 = residue[interaction[1][1]].get_vector()
                    coords3 = atom.get_vector()

                    # alcohols need special treatment to get lone pairs location as acceptors
                    if resname in ['SER', 'THR', 'TYR'] and interaction[2][2] in ['alpha', 'beta']:
                        # get H atom coordinates
                        if resname == 'SER':
                            coords4 = residue['HG'].get_vector()
                        elif resname == 'THR':
                            coords4 = residue['HG1'].get_vector()
                        elif resname == 'TYR':
                            coords4 = residue['HH'].get_vector()
                        # calculate C-C-O-H dihedral
                        p1 = np.asarray([coords1[0],coords1[1],coords1[2]])
                        p2 = np.asarray([coords2[0],coords2[1],coords2[2]])
                        p3 = np.asarray([coords3[0],coords3[1],coords3[2]])
                        p4 = np.asarray([coords4[0],coords4[1],coords4[2]])
                        dihedral = self.get_dihedral(p1, p2, p3, p4)
                        if interaction[2][2] == 'alpha':
                            ang = dihedral + 120
                        elif interaction[2][2] == 'beta':
                            ang = dihedral + 240
                        position = self.pos4(coords1[0], coords1[1], coords1[2], 
                            coords2[0], coords2[1], coords2[2], coords3[0], 
                            coords3[1], coords3[2], interaction[2][0], 
                            interaction[2][1], ang)
                        bpf_line = self.write_bpf_format('don', position)
                        count += 1
                        bpf_lines.append(bpf_line)
                            
                    # for non alcohol groups (and alcohols as donors)
                    else:
                        if interaction[3] in ['don', 'acc']:
                            # if protein atom act as donor, write an acceptor site according to ideal interaction
                            position = self.pos4(coords1[0], coords1[1], coords1[2], 
                                coords2[0], coords2[1], coords2[2], coords3[0], 
                                coords3[1], coords3[2], interaction[2][0], 
                                interaction[2][1], interaction[2][2])
                            if interaction[3] == 'don':
                                bpf_line = self.write_bpf_format('acc', position)
                            # if protein atom act as acceptor, write a donor site according to ideal interaction
                            elif interaction[3] == 'acc':
                                bpf_line = self.write_bpf_format('don', position)
                            count += 1
                            bpf_lines.append(bpf_line)
                
            # backbone HB interactions
            if resname != 'HOH' and resname != 'WAT':
                for interaction in self.ideal_interactions_bb:
                    # if atom is interactor according to ideal interactions previously defined
                    if atomname == interaction[1][2]:
                        # get coordinates of the three atoms defining the interaction
                        coords1 = residue[interaction[1][0]].get_vector()
                        coords2 = residue[interaction[1][1]].get_vector()
                        coords3 = atom.get_vector()
                        # pos4 needs the coordinates of the three atoms defining the interaction + distance + angle + dihedral
                        position = self.pos4(coords1[0], coords1[1], coords1[2], 
                            coords2[0], coords2[1], coords2[2], coords3[0], 
                            coords3[1], coords3[2], interaction[2][0], 
                            interaction[2][1], interaction[2][2])
                        # if interactor is carbonyl				
                        # write a donor interactor at 5 different locations surrounding the carbonyl (ideal = 120 and 240 deg)
                        if interaction[3] in ['acc', 'don']:
                            if interaction[3] == 'acc':
                                bpf_line = self.write_bpf_format('don', position)
                            # if interactor is NH
                            # write an acceptor interactor at HBOND_DIST angstroms from the H in the straight line defined by N-H 
                            elif interaction[3] == 'don':
                                bpf_line = self.write_bpf_format('acc', position)
                            count += 1
                            bpf_lines.append(bpf_line)
        
        with open(out_bias_file, 'w') as f:
            f.write("\tx\ty\tz\tVset\tr\ttype\n")
            for line in bpf_lines:
                f.write(line + '\n')


class AtomType(object):
    def __init__(self):
        self.atom_type_convert_map = {
            '[#1]': 'H', 
            '[#1][#7,#8,#9,#15,#16]': 'HD', 
            '[#5]': 'B', 
            '[C]': 'C', 
            '[c]': 'A', 
            '[#7]': 'NA', 
            '[#8]': 'OA', 
            '[#9]': 'F', 
            '[#12]': 'Mg', 
            '[#14]': 'Si', 
            '[#15]': 'P', 
            '[#16]': 'S', 
            '[#17]': 'Cl', 
            '[#20]': 'Ca', 
            '[#25]': 'Mn', 
            '[#26]': 'Fe', 
            '[#30]': 'Zn', 
            '[#35]': 'Br', 
            '[#53]': 'I', 
            '[#7X3v3][a]': 'N', 
            '[#7X3v3][#6X3v4]': 'N', 
            '[#7+1]': 'N', 
            '[SX2]': 'SA'}

    def get_docking_atom_types(self, mol:Chem.rdchem.Mol) -> Dict[int, str]:
        atom_ind_type_map = dict()
        for smarts, atom_type in self.atom_type_convert_map.items():
            pattern_mol = Chem.MolFromSmarts(smarts)
            pattern_matches = mol.GetSubstructMatches(pattern_mol)
            for pattern_match in pattern_matches:
                atom_ind = pattern_match[0]
                atom_ind_type_map[atom_ind] = atom_type
        return atom_ind_type_map


def gen_mcs_index_and_bpf(name_content_tuple:Tuple[str, str], 
                          ref_mol:Chem.rdchem.Mol) -> Tuple[str, str, str]:
    radius_map = {
        "1": 1.08, 
        "2": 1.4, 
        "5": 1.47, 
        "6": 1.49,
        "7": 1.41,
        "8": 1.4,
        "9": 1.39,
        "11": 1.84,
        "12": 2.05,
        "13": 2.11,
        "14": 2.1,
        "15": 1.92,
        "16": 1.82,
        "17": 1.83,
        "19": 2.05,
        "20": 2.21,
        "25": 1.97,
        "26": 1.94,
        "30": 2.1,
        "35": 1.98,
        "53": 2.23
    }
    vset = -1.0

    name, content = name_content_tuple

    target_mol = Chem.MolFromMolBlock(content, removeHs=False, sanitize=True)
    mcs = rdFMCS.FindMCS([ref_mol, target_mol], timeout=10)
    query = Chem.MolFromSmarts(mcs.smartsString)
    match_indexs = target_mol.GetSubstructMatch(query)
    atom_info_map = AtomType().get_docking_atom_types(target_mol)
    bias_content = "x y z Vset r type atom\n"
    for atom in target_mol.GetAtoms():
        atom_idx = atom.GetIdx()
        if atom_idx in match_indexs:
            atom_type = atom_info_map[atom_idx]
            atomic_num = atom.GetAtomicNum()
            atom_radius = radius_map[str(atomic_num)]
            position = target_mol.GetConformer().GetAtomPosition(atom_idx)
            bias_content += f"{position.x:6.3f} {position.y:6.3f} {position.z:6.3f} {vset:6.2f} {atom_radius:6.2f} map {atom_type:<2s}\n"
    return name, ",".join([str(i) for i in match_indexs]), bias_content