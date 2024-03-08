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


@OP.function
def rescore_gnina_op(receptor:Artifact(Path), 
                     content_json:Artifact(Path),
                     meta_json:Artifact(Path),
                     sdf_batch:Parameter(int, default=128)
) -> {"meta_json":Artifact(Path)}:
    import os
    import subprocess
    import time
    from tools.stdio import StdIO, MetaIO

    ligands_dir = Path("./tmp/inputs_dir")
    ligands_dir.mkdir(parents=True, exist_ok=True)

    content_stdio = StdIO(json_path=content_json)
    
    score_tag = "gnina_score"
    affinity_tag = "gnina_affinity"
    score_stdio = MetaIO(meta_json)
    score_stdio.init_key(score_tag, 0)
    score_stdio.init_key(affinity_tag, 0)

    sdf_files, batch_name_list = content_stdio.write_sdf_content_to_big_file(ligands_dir, sdf_batch)
    for i, sdf_file in enumerate(sdf_files):
        start_time = time.time()
        name_list = batch_name_list[i]
        cmd = f"gnina -r {str(receptor)} -l {sdf_file} --score_only"
        print(cmd)
        resp = subprocess.run(cmd, shell=True, encoding="utf-8", 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(resp.stdout)
        if resp.stderr:
            print(resp.stderr)

        lines = [line.strip() for line in resp.stdout.split('\n')]
        CNNscores = [line.split()[-1] for line in lines if "CNNscore" in line]
        CNNaffinitys = [line.split()[-1] for line in lines if "CNNaffinity" in line ]
        assert len(name_list) == len(CNNscores), "gnina result len not match with inputs"
        for j, name in enumerate(name_list):
            name, _, pose_ind = name.rpartition("_pose")
            pose_ind = int(pose_ind)
            score_stdio.update_pose_by_dict(name, pose_ind, 
                                            {score_tag: float(CNNscores[j]), 
                                             affinity_tag: float(CNNaffinitys[j])})

        print(f"One ligand calc time: {time.time() - start_time} seconds")

    meta_path = Path("meta.json")
    score_stdio.write_json(meta_path)

    return OPIO({"meta_json": meta_path})


@OP.function
def rescore_RTMScore_op(pocket:Artifact(Path), 
                        content_json:Artifact(Path), 
                        meta_json:Artifact(Path),
                        sdf_batch:Parameter(int, default=1800),
                        batch:Parameter(int, default=512)
) ->  {"meta_json":Artifact(Path)}:
    import os
    import subprocess
    import pandas as pd
    import time
    from tools.stdio import StdIO, MetaIO

    score_tag = "rtmscore"
    score_stdio = MetaIO(json_path=meta_json)
    score_stdio.init_key(score_tag, 0)

    ligands_dir = Path("inputs_dir")
    ligands_dir.mkdir(parents=True, exist_ok=True)

    content_stdio = StdIO(json_path=content_json)
    sdf_files, batch_name_list = content_stdio.write_sdf_content_to_big_file(ligands_dir, sdf_batch)

    for i, sdf_file in enumerate(sdf_files):
        start_time = time.time()
        name_list = batch_name_list[i]
        cmd = f"python /opt/RTMScore/RTMScore/main.py \
                -p {str(pocket)} -l {sdf_file} -o ligand-{i} -pl -b {batch}"
        print(cmd)
        resp = subprocess.run(cmd, shell=True, text=True, 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(resp.stdout)
        if resp.stderr:
            print(resp.stderr)
        
        score_table = pd.read_csv(f"ligand-{i}.csv", delimiter=",")
        print(score_table)
        assert len(name_list) == score_table.shape[0], "rtm score result len not match with inputs"
        for ind, row in score_table.iterrows():
            name = name_list[ind]
            name, _, pose_ind = name.rpartition("_pose")
            pose_ind = int(pose_ind)
            score = float(row["score"])
            score_stdio.update_pose_by_key_value(name, pose_ind, score_tag, score)
        print(f"One ligand calc time: {time.time() - start_time} seconds")
    
    meta_path = Path("meta.json")
    score_stdio.write_json(meta_path)
            
    return OPIO({"meta_json": meta_path})


@OP.function
def merge_score_op(meta_json_list:Artifact(List[Path], optional=True)
)-> {"meta_json":Artifact(Path)}:
    import os
    import uuid
    from tools.stdio import MetaIO

    print(meta_json_list)
    meta_io = None
    for meta_json_path in meta_json_list:
        if os.path.exists(meta_json_path):
            if not meta_io:
                meta_io = MetaIO(json_path=meta_json_path)
            else:
                meta_io.update_by_meta_json(meta_json_path)
    meta_path = Path(f"meta_{uuid.uuid4().hex}.json")
    meta_io.write_json(meta_path)
    return OPIO({"meta_json": meta_path})