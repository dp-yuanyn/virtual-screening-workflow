from pathlib import Path
from dflow import (
    Step,
    Steps,
    InputArtifact,
    OutputArtifact,
    Executor,
)
from dflow.python import (
    OP, 
    OPIO, 
    Artifact,
    Parameter,
    PythonOPTemplate,
    upload_packages,
)

if "__file__" in locals():
    upload_packages.append(__file__)


@OP.function
def get_pocket_op(receptor:Artifact(Path), docking_params:Parameter(dict)) -> {"pocket":Artifact(Path)}:
    receptor_path_str = receptor.as_posix()
    import os
    import sys
    from pymol import cmd

    scoring_func = docking_params["scoring"]
    center_x, center_y, center_z = docking_params["center_x"], docking_params["center_y"], docking_params["center_z"]
    size_x, size_y, size_z = docking_params["size_x"], docking_params["size_y"], docking_params["size_z"]

    x_min = center_x - size_x / 2
    x_max = center_x + size_x / 2
    y_min = center_y - size_y / 2
    y_max = center_y + size_y / 2
    z_min = center_z - size_z / 2
    z_max = center_z + size_z / 2

    pocket_path = Path("pocket.pdb")
    cmd.remove('all')
    cmd.load(receptor_path_str, 'init_protein')
    cmd.select('pok_res', f'byres ((polymer.protein or solvent) and (x > {x_min} and x < {x_max} and y > {y_min} and y < {y_max} and z > {z_min} and z < {z_max}))')
    cmd.save(pocket_path,'pok_res')

    return OPIO({"pocket":pocket_path})