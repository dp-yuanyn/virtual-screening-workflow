from typing import List
from pathlib import Path
from dflow.python import (
    OP, 
    OPIO, 
    OPIOSign,
    Artifact,
    Parameter,
    upload_packages,
)

if "__file__" in locals():
    upload_packages.append(__file__)


@OP.function
def run_pbgbsa_op(receptor_file:Artifact(Path), 
                  content_json:Artifact(Path),
                  meta_json:Artifact(Path),
                  pbgbsa_params:Parameter(dict)
) -> {"meta_json": Artifact(Path)}:
    import os
    import json
    import math
    import uuid
    import subprocess
    from multiprocessing import Pool
    from tools.stdio import StdIO, MetaIO
    from tools.pbgbsa import read_pbgbsa_results

    ligands_dir = Path("ligands_dir")
    ligands_dir.mkdir(parents=True, exist_ok=True)
    content_io = StdIO(json_path=content_json)
    input_ligands = content_io.write_pose_content_to_file(ligands_dir)

    results_dir = Path("results_dir")
    results_dir.mkdir(parents=True, exist_ok=True)
    param_file = results_dir.joinpath("pbgbsa_params.json")
    with open(str(param_file), "w") as f:
        json.dump(pbgbsa_params, f)
    cmd = f"unigbsa-pipeline -i {str(receptor_file.absolute())} -d {str(ligands_dir.absolute())} \
        -c {str(param_file.absolute())} -nt {os.cpu_count()}"
    print(cmd)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    res = subprocess.run(cmd, shell=True, 
                         capture_output=True, text=True, 
                         env=env, cwd=results_dir)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)
    
    meta_io = MetaIO(json_path=meta_json)
    ligand_results_dir_list = [os.path.join(results_dir, os.path.splitext(os.path.basename(input_ligand))[0]) for input_ligand in input_ligands]
    batch_size = math.ceil(len(ligand_results_dir_list)/(os.cpu_count()*8))
    batch_ligand_results_dir_list = [ligand_results_dir_list[i:i+batch_size] for i in range(0, len(ligand_results_dir_list), batch_size)]
    with Pool(processes=os.cpu_count()) as pool:
        result_list = pool.map(read_pbgbsa_results, batch_ligand_results_dir_list)
    result_list = sum(result_list, [])
    for name, pbgbsa_item in result_list:
        name, _, pose_ind = name.rpartition("_pose")
        pose_ind = int(pose_ind)
        meta_io.update_pose_by_dict(name, pose_ind, pbgbsa_item)
    
    result_meta_path = Path(f"meta_{uuid.uuid4().hex}.json")
    meta_io.write_json(result_meta_path)
    return OPIO({"meta_json": result_meta_path})