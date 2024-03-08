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


@OP.function
def karmadock_op(
        receptor:Artifact(Path),
        content_json:Artifact(Path), 
        meta_json:Artifact(Path),
) -> {"meta_json":Artifact(Path)}:
    import os
    import json
    import uuid
    import time
    import subprocess
    import logging
    from tools.stdio import StdIO, MetaIO

    ligands_dir = Path("./ligands_dir")
    ligands_dir.mkdir(parents=True, exist_ok=True)
    content_io = StdIO(content_json)
    logging.info("Start write files")
    sdf_files, batch_names = content_io.write_sdf_content_to_big_file(ligands_dir)
    print(sdf_files)
    print(batch_names)
    logging.info("Finish write files")

    sdf_name_ind_map = {os.path.splitext(os.path.basename(name))[0]: i for i, name in enumerate(sdf_files)}

    results_dir = Path("./results_dir")
    results_dir.mkdir(parents=True, exist_ok=True)

    # command
    cmd = [
        "python",
        "/opt/KarmaDock/utils/rescoring.py",
        "--ligand_path", ligands_dir.as_posix(),
        "--protein_file", receptor.as_posix(),
        "--model_file", "/opt/KarmaDock/trained_models/karmadock_screening.pkl",
        "--out_dir", results_dir.as_posix(),
        "--batch_size", "256",
        "--random_seed", "181129",
        "--autobox"
    ]
    logging.info(f"command: {json.dumps(cmd)}")
    resp = subprocess.run(cmd, capture_output=True, text=True)
    print(resp.stdout)
    if resp.stderr:
        print(resp.stderr)

    score_tag = "karmadock_score"
    score_stdio = MetaIO(meta_json)
    score_stdio.init_key(score_tag, 0)
    # read scores
    with open(os.path.join(results_dir, "score.csv"), "r") as f:
        for line in f.readlines():
            if line:
                if line.startswith("ligand_name,score"): continue
                calc_name, score = line.strip().split(",")
                score = float(score)
                sdfname, _, ind = calc_name.rpartition("-")
                sdf_ind = sdf_name_ind_map[sdfname]
                ind = int(ind)
                name = batch_names[sdf_ind][ind]
                name, _, conf_id = name.rpartition("_pose")
                conf_id = int(conf_id)
                score_stdio.update_pose_by_dict(name, conf_id, 
                                                {score_tag: score})

    meta_path = Path("meta_karma.json")
    score_stdio.write_json(meta_path)
    return OPIO({"meta_json": meta_path})