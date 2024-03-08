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
def ifp_v2_op(receptor_pdb:Artifact(Path), 
            content_json:Artifact(Path),
            meta_json:Artifact(Path),
) -> {"meta_json":Artifact(Path), "content_json":Artifact(Path)}:
    import os
    import json
    import glob
    import uuid
    import pandas as pd
    from tools.stdio import StdIO, MetaIO
    from tools.cal_ifp import get_ifp_df, convert_df


    content_io = StdIO(json_path=content_json)
    protein_path = str(receptor_pdb)

    ligands_dir = "./"
    sdf_files, names_list = content_io.write_sdf_content_to_big_file(ligands_dir, batch_size=len(content_io))
    assert len(sdf_files) == 1, "only write one sdf file"
    sdf_file = sdf_files[0]
    names = names_list[0]
    print(f"Num ligands: {len(names)}")

    df = get_ifp_df(protein_path, sdf_file)
    df = convert_df(df)
    df.to_csv("df.csv")

    score_stdio = MetaIO(json_path=meta_json)
    for index, row in df.iterrows():
        ligname, _, conf_id = names[index].rpartition("_pose")
        conf_id = int(conf_id)
        ifp_info_list =[]
        for name, valid in row.items():
            if valid:
                ifp_info_list.append(name)

        score_stdio.update_pose_by_dict(ligname, conf_id, {"ifp": ",".join(ifp_info_list)})

    meta_path = Path(f"meta_{uuid.uuid4().hex}.json")
    score_stdio.write_json(meta_path)

    return OPIO({"meta_json": meta_path, "content_json": content_json})