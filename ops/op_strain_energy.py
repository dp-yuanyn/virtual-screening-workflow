from pathlib import Path
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
def strain_energy_op(
        content_json:Artifact(Path), 
        meta_json:Artifact(Path),
        batch:Parameter(int, optional=True, default=500)
) -> {"meta_json":Artifact(Path), "tag_name": Parameter(str)}:
    import os
    import json
    import uuid
    import time
    import subprocess
    from tools.stdio import StdIO, MetaIO

    script_path = "~/mayachemtools/bin/RDKitFilterTorsionStrainEnergyAlerts.py"
    library_data_path = "~/mayachemtools/lib/data/TorsionStrainEnergyLibrary.xml"

    ligands_dir = Path("ligands_dir")
    ligands_dir.mkdir(parents=True, exist_ok=True)
    result_dir = Path(f"tmp_{uuid.uuid4().hex}")
    result_dir.mkdir(parents=True, exist_ok=True)

    content_stdio = StdIO(content_json)
    content_dict = content_stdio.to_dict()

    score_stdio = MetaIO(meta_json)
    key_name = "strain_energy_valid"

    start_time = time.time()

    names = list(content_dict.keys())
    batch_names = [names[i:i+batch] for i in range(0 ,len(names), batch)]
    print(f"Batch Num: {len(batch_names)}")
    
    score_stdio.init_key(key_name, 0)
    print(f"Init time: {time.time() - start_time} seconds")
    
    for i, sub_names in enumerate(batch_names):
        start_time = time.time()
        output_name = f"ligand-{i}"
        ligand_path = os.path.join(ligands_dir, f"{output_name}.sdf")

        basename_pose_map_tmp = []
        contents = ""
        for name in sub_names:
            content_list = content_dict[name][list(content_dict[name].keys())[0]]
            for j, content in enumerate(content_list):
                basename_pose_map_tmp.append([name, j])
            contents += "".join(content_list)

        with open(ligand_path, '+a') as f:
            f.write(contents)
        
        print(basename_pose_map_tmp)
        print(f"One ligand write time: {time.time() - start_time} seconds")
        start_time = time.time()


        resp = subprocess.run(f'python {script_path} \
                              --torsionEnergyLibraryFile {library_data_path} \
                                -i {os.path.abspath(ligand_path)} -o {output_name}.sdf --mp yes --overwrite', 
                                shell=True, text=True, cwd=result_dir,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(resp.stdout)
        if resp.stderr:
            print(resp.stderr)
        result_json_path = os.path.join(result_dir, f"{output_name}.json")

        print(f"One ligand calc time: {time.time() - start_time} seconds")
        start_time = time.time()

        if os.path.exists(result_json_path):
            with open(result_json_path, "r") as f:
                strain_energy_dict = json.load(f)
                for idx, ligand_id_str in enumerate(list(strain_energy_dict.keys())):
                    valid = -1
                    if strain_energy_dict[ligand_id_str]:
                        ligand_name = basename_pose_map_tmp[idx][0]
                        conf_id = basename_pose_map_tmp[idx][1]
                        valid = int(strain_energy_dict[ligand_id_str].get("AnglesNotObservedCount")==0)
                        score_stdio.update_pose_by_dict(ligand_name, int(conf_id), {key_name: valid})

        print(f"One ligand add info time: {time.time() - start_time} seconds")

    result_meta_path = Path("meta_strain_energy.json")
    score_stdio.write_json(result_meta_path)

    return OPIO({"meta_json": result_meta_path,
                 "tag_name": key_name})