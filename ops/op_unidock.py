from typing import Tuple, List
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
def gen_ad4_map_op(receptor:Artifact(Path), 
                   content_json:Artifact(Path), 
                   docking_params:Parameter(dict)
) -> {"map_dir":Artifact(Path)}:
    import os
    import math
    import shutil
    import json
    import subprocess
    from tools.stdio import StdIO

    center_x, center_y, center_z = docking_params["center_x"], docking_params["center_y"], docking_params["center_z"]
    size_x, size_y, size_z = docking_params["size_x"], docking_params["size_y"], docking_params["size_z"]

    map_dir = "mapdir"
    os.makedirs(map_dir, exist_ok=True)
    spacing = 0.375

    protein_name = os.path.splitext(os.path.basename(receptor))[0]
    shutil.copyfile(receptor, os.path.join(map_dir, os.path.basename(receptor)))
    
    atom_types = set()
    stdio = StdIO(content_json)
    for content in stdio.iter_best_content_list():
        print(content)
        tag = False
        for line in content.split("\n"):
            if line.strip():
                if line.startswith(">  <atomInfo>"):
                    tag = True
                elif tag and (line.startswith(">  <") or line.startswith("> <") or line.startswith("$$$$")):
                    tag = False
                elif tag:
                    atom_types.add(line[13:].strip())

    atom_types = list(atom_types)
    npts = [math.ceil(s / spacing) for s in [size_x, size_y, size_z]]

    data_path = "/opt/data/unidock/AD4.1_bound.dat"
    mgltools_python_path = shutil.which("pythonsh")
    if not mgltools_python_path:
        raise KeyError("No mgltools env")
    mgltools_python_path = str(mgltools_python_path)
    prepare_gpf4_script_path = os.path.join(os.path.dirname(os.path.dirname(mgltools_python_path)), 
        "MGLToolsPckgs", "AutoDockTools", "Utilities24", "prepare_gpf4.py")
    cmd = f'{mgltools_python_path} {prepare_gpf4_script_path} -r {os.path.basename(receptor)} \
        -p gridcenter="{center_x},{center_y},{center_z}" -p npts="{npts[0]},{npts[1]},{npts[2]}" \
        -p spacing={spacing} -p ligand_types="{",".join(atom_types)}" -o {protein_name}.gpf && \
        sed -i "1i parameter_file {data_path}" {protein_name}.gpf && \
        autogrid4 -p {protein_name}.gpf -l {protein_name}.glg'

    print(cmd)
    resp = subprocess.run(cmd, shell=True, 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
            encoding="utf-8", cwd=map_dir)
    print(resp.stdout)
    if resp.stderr:
        print(resp.stderr)
    return {"map_dir": Path(map_dir)}


@OP.function
def run_unidock_op(receptor:Artifact(Path), 
        ligand_content_json:Artifact(Path), 
        meta_json:Artifact(Path, optional=True),
        bias_file:Artifact(Path, optional=True), 
        bias_content_json:Artifact(Path, optional=True), 
        docking_params:Parameter(dict), 
        batch_size:Parameter(int, default=1200),
        score_key:Parameter(str, default="origin_score"),
        constrain_type:Parameter(str, default=""),
) -> {"meta_json":Artifact(Path), "content_json": Artifact(Path)}:
    import os
    import shutil
    import math
    import uuid
    import glob
    import subprocess
    import logging
    from pprint import pprint
    from functools import partial
    from multiprocessing import Pool
    from tools.stdio import StdIO, MetaIO
    from tools.helper import read_score_list_from_file

    logging.getLogger().setLevel(logging.INFO)
    logging.info(os.getcwd())

    scoring_func = docking_params["scoring"]
    receptor_path_str = receptor.as_posix()
    if scoring_func == "ad4":
        mapdir = "mapdir"
        map_name = os.path.basename(os.path.splitext(glob.glob(os.path.join(receptor_path_str, "*.glg"))[0])[0])
        shutil.copytree(receptor, mapdir)
        receptor_path_str = os.path.join(mapdir, map_name)

    output_dir = Path("./tmp_results_dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    ligands_dir = Path("./tmp_inputs_dir")
    ligands_dir.mkdir(parents=True, exist_ok=True)

    ligand_stdio = StdIO(json_path=ligand_content_json)
    ligand_files = ligand_stdio.write_raw_content_to_file(ligands_dir, fmt="sdf")

    if bias_file:
        docking_params["bias"] = str(bias_file)

    if bias_content_json:
        bias_stdio = StdIO(json_path=bias_content_json)
        bias_stdio.write_raw_content_to_file(ligands_dir, fmt="bpf")
        docking_params["multi_bias"] = True

    real_batch_size = math.ceil(len(ligand_files)/math.ceil(len(ligand_files)/batch_size))
    batch_ligand_files = [ligand_files[i:i+real_batch_size] for i in range(0, len(ligand_files), real_batch_size)]
    for batch_ind, sub_ligand_files in enumerate(batch_ligand_files):
        docking_file_list = output_dir.joinpath(f"file_list_{batch_ind}")
        with open(docking_file_list, "w") as f:
            for ligand_file in sub_ligand_files:
                f.write(ligand_file + "\n")
        if scoring_func == "ad4":
            cmd = "unidock --maps {} --ligand_index {} --dir {}".format(receptor_path_str, 
                        docking_file_list.as_posix(), output_dir.as_posix())
        else:
            cmd = "unidock --receptor {} --ligand_index {} --dir {}".format(receptor_path_str, 
                        docking_file_list.as_posix(), output_dir.as_posix())

        for k, v in docking_params.items():
            if isinstance(v, bool) and v:
                cmd += f' --{k}'
            else:
                cmd += f' --{k} {v}'

        cmd += " --verbosity 2"
        cmd += " --refine_step 3"
        cmd += " --keep_nonpolar_H"
        print(cmd)

        res = subprocess.run(args=cmd, shell=True, 
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    encoding="utf-8")
        print(res.stdout)
        if res.stderr:
            print(res.stderr)

    result_ligands = glob.glob(os.path.join(output_dir, "*.sdf"))
    results_stdio = StdIO()
    results_stdio.save_content_by_input_files(result_ligands, split_pose=True)
    content_path = Path(f"content_{uuid.uuid4().hex}.json")
    results_stdio.write_content_json(content_path)

    with Pool(processes=max(1, min(len(result_ligands), os.cpu_count())), maxtasksperchild=10) as pool:
        meta_name_list_tuple_list = pool.map(partial(read_score_list_from_file, 
                                                     score_key=score_key,
                                                     constrain_type=constrain_type), result_ligands)

    pprint(meta_name_list_tuple_list)
    score_stdio = MetaIO()
    for name, meta_list in meta_name_list_tuple_list:
        if len(meta_list) > 0:
            score_stdio.add_meta_list(name, meta_list)
    if meta_json:
        input_meta_io = MetaIO(json_path=meta_json)
        for name, meta_list in input_meta_io.to_dict().items():
            if score_stdio.has_key(name):
                score_stdio.update_ligand_by_key(name, meta_list[0])
    meta_path = Path(f"meta_{uuid.uuid4().hex}.json")
    score_stdio.write_json(meta_path)

    return OPIO({"content_json": content_path, 
                 "meta_json": meta_path})


def run_unidock_rescore_single(content_items:List[tuple], cmd:str, workdir:str):
    import os
    import shutil
    import math
    import datetime
    import uuid
    import subprocess
    import traceback
    from tqdm import tqdm

    rescore_list = []
    print(f"One process start time: {datetime.datetime.now()}")
    try:
        tmp_dir = Path(os.path.join(workdir, f"tmp_{uuid.uuid4().hex}"))
        tmp_dir.mkdir(parents=True, exist_ok=True)

        batch_size = 200
        real_batch_size = math.ceil(len(content_items)/math.ceil(len(content_items)/batch_size))
        batched_items = [content_items[i:i+real_batch_size] for i in range(0, len(content_items), real_batch_size)]
        for batch_ind, sub_items in enumerate(tqdm(batched_items)):
            sub_dir = tmp_dir.joinpath(f"subdir_{batch_ind}")
            sub_dir.mkdir(parents=True, exist_ok=True)
            ligand_files = []
            for name, content_dict in sub_items:
                content_list = content_dict["contents"]
                for i, content in enumerate(content_list):
                    ligand_file = sub_dir.joinpath(f"{name}_pose{i}.sdf")
                    with open(ligand_file, "w") as f:
                        f.write(content)
                    ligand_files.append(ligand_file.as_posix())

            file_list_path = tmp_dir.joinpath(f"docking_file_list_{batch_ind}")
            with open(file_list_path, "w") as f:
                for ligand_file in ligand_files:
                    f.write(ligand_file + "\n")
            print(len(ligand_files))
            batch_cmd = cmd + f" --ligand_index {file_list_path.as_posix()} --dir {tmp_dir.as_posix()}"
            print(batch_cmd)
            with open(tmp_dir.joinpath(f"{batch_ind}_log"), "w") as f:
                res = subprocess.run(args=batch_cmd, shell=True, 
                    stdout=f, stderr=subprocess.STDOUT, text=True)

            with open(tmp_dir.joinpath("scores.txt"), "r") as f:
                for line in f.readlines():
                    if line.startswith("REMARK"):
                        line_list = line.strip("\n").split(" ")
                        name = os.path.splitext(os.path.basename(line_list[1]))[0]
                        name, _, conf_id = name.rpartition("_pose")
                        conf_id = int(conf_id)
                        score = float(line_list[2])
                        rescore_list.append((name, conf_id, score))
            os.rename(tmp_dir.joinpath("scores.txt"), tmp_dir.joinpath(f"scores_{batch_ind}.txt"))
            shutil.rmtree(sub_dir, ignore_errors=True)
    except:
        traceback.print_exc()
    return rescore_list

@OP.function
def run_unidock_score_only_op(receptor:Artifact(Path), 
                              content_json:Artifact(Path), 
                              meta_json:Artifact(Path),
                              docking_params:Parameter(dict),
                              score_tag:Parameter(str, default="")
) -> {"meta_json":Artifact(Path)}:
    import os
    import shutil
    import glob
    import uuid
    import math
    import datetime
    from functools import partial
    from multiprocessing import Pool
    from tools.stdio import StdIO
    from tools.stdio import MetaIO

    print(f"Start in op time: {datetime.datetime.now()}")
    scoring_func = docking_params["scoring"]
    center_x, center_y, center_z = docking_params["center_x"], docking_params["center_y"], docking_params["center_z"]
    size_x, size_y, size_z = docking_params["size_x"], docking_params["size_y"], docking_params["size_z"]
    size_x, size_y, size_z = min(30, size_x+8), min(30, size_y+8), min(30, size_z+8)
    receptor_path_str = receptor.as_posix()

    cmd = f"unidock --scoring {scoring_func} --exhaustiveness 1 --score_only \
        --center_x {center_x} --center_y {center_y} --center_z {center_z} \
        --size_x {size_x} --size_y {size_y} --size_z {size_z}"

    if scoring_func == "ad4":
        mapdir = "mapdir"
        map_name = os.path.basename(os.path.splitext(glob.glob(os.path.join(receptor_path_str, "*.glg"))[0])[0])
        shutil.copytree(receptor, mapdir)
        receptor_path_str = os.path.join(mapdir, map_name)
        cmd += f" --maps {receptor_path_str}"
    else:
        cmd += f" --receptor {receptor_path_str}"

    workdir = Path("./tmp/workdir")
    workdir.mkdir(parents=True, exist_ok=True)

    content_io = StdIO(json_path=content_json)
    content_items = list(content_io.to_dict().items())
    n_process = max(1, min(os.cpu_count(), len(content_items)))
    batch_size = math.ceil(len(content_items)/n_process)
    batch_content_items = [content_items[i:i+batch_size] for i in range(0, len(content_items), batch_size)]
    #batch_ligand_files = [ligand_files[i:i+batch_size] for i in range(0, len(ligand_files), batch_size)]

    print(f"Start multiprocessing time: {datetime.datetime.now()}")
    with Pool(processes=n_process) as pool:
        rescore_list = pool.map(partial(run_unidock_rescore_single, cmd=cmd, workdir=workdir), 
                                    batch_content_items)
        rescore_list = sum(rescore_list, [])

    score_stdio = MetaIO(json_path=meta_json)
    if not score_tag:
        score_tag = f"{scoring_func}_score"
    score_stdio.init_key(score_tag, 0)

    print(f"Start update score time: {datetime.datetime.now()}")
    for rescore_item in rescore_list:
        name, conf_id, score = rescore_item
        score_stdio.update_pose_by_key_value(name, conf_id, score_tag, score)

    meta_path = Path(f"meta_{uuid.uuid4().hex}.json")
    score_stdio.write_json(meta_path)

    print(f"End time: {datetime.datetime.now()}")
    return OPIO({"meta_json": meta_path})


@OP.function
def run_unidock_score_only_gpu_op(receptor:Artifact(Path), 
                              content_json:Artifact(Path), 
                              meta_json:Artifact(Path),
                              docking_params:Parameter(dict),
                              batch_size:Parameter(int, default=6000)
        ) -> {"meta_json":Artifact(Path, optional=True)}:
    import os
    import shutil
    import glob
    import uuid
    import copy
    import math
    import subprocess
    from multiprocessing import Pool
    from functools import partial
    from tools.stdio import StdIO
    from tools.stdio import MetaIO
    from tools.helper import read_score_list_from_file

    ligands_dir = Path("./tmp/ligands_dir")
    ligands_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path("./tmp/output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    scoring_func = docking_params["scoring"]
    center_x, center_y, center_z = docking_params["center_x"], docking_params["center_y"], docking_params["center_z"]
    size_x, size_y, size_z = docking_params["size_x"], docking_params["size_y"], docking_params["size_z"]
    receptor_path_str = receptor.as_posix()

    cmd = f"unidock --verbosity 2 --scoring {scoring_func} --num_modes 1 \
        --exhaustiveness 1 --max_step 10 --local_only \
        --center_x {center_x} --center_y {center_y} --center_z {center_z} \
        --size_x {size_x} --size_y {size_y} --size_z {size_z} \
        --dir {output_dir.as_posix()}"
    if scoring_func == "ad4":
        mapdir = "mapdir"
        map_name = os.path.basename(os.path.splitext(glob.glob(os.path.join(receptor_path_str, "*.glg"))[0])[0])
        shutil.copytree(receptor, mapdir)
        receptor_path_str = os.path.join(mapdir, map_name)
        cmd += f" --maps {receptor_path_str}"
    else:
        cmd += f" --receptor {receptor_path_str}"

    content_io = StdIO(json_path=content_json)

    ligand_files = content_io.write_pose_content_to_file(ligands_dir, fmt="sdf")

    real_batch_size = math.ceil(len(ligand_files)/math.ceil(len(ligand_files)/batch_size))
    batch_ligand_files = [ligand_files[i:i+real_batch_size] for i in range(0, len(ligand_files), real_batch_size)]
    print(f"Total batch: {len(batch_ligand_files)}")
    for batch_ind, sub_ligand_files in enumerate(batch_ligand_files):
        docking_file_list = output_dir.joinpath(f"file_list_{batch_ind}")
        with open(docking_file_list, "w") as f:
            for ligand_file in sub_ligand_files:
                f.write(ligand_file + "\n")
        sub_cmd = copy.copy(cmd)
        sub_cmd += f" --ligand_index {docking_file_list.as_posix()}"
        print(sub_cmd)
        res = subprocess.run(args=sub_cmd, shell=True, 
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    encoding="utf-8")
        print(res.stdout)
        if res.stderr:
            print(res.stderr)
        print(f"Batch {batch_ind} finished")

    result_ligands = glob.glob(os.path.join(output_dir, "*.sdf"))

    score_key = f"{scoring_func}_score"
    with Pool(processes=max(1, min(len(result_ligands), os.cpu_count())), maxtasksperchild=10) as pool:
        meta_name_list_tuple_list = pool.map(partial(read_score_list_from_file, 
                                                     score_key=score_key), result_ligands)

    score_stdio = MetaIO(json_path=meta_json)
    score_tag = f"{scoring_func}_score"
    score_stdio.init_key(score_tag, 0)

    for name, rescore_list in meta_name_list_tuple_list:
        conf_id = 0
        _, _, score = rescore_list[0]
        name, _, conf_id = name.rpartition("_pose")
        conf_id = int(conf_id)
        score_stdio.update_pose_by_dict(name, conf_id, {score_tag: score})

    meta_path = Path(f"meta_{uuid.uuid4().hex}.json")
    score_stdio.write_json(meta_path)

    return OPIO({"meta_json": meta_path})