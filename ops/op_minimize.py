from pathlib import Path
from dflow.python import (
    OP, 
    OPIO,
    OPIOSign, 
    Artifact,
    upload_packages,
)

if "__file__" in locals():
    upload_packages.append(__file__)


@OP.function
def minimize_op(
    content_json:Artifact(Path),
    receptor_pdb:Artifact(Path),
) -> {"content_json":Artifact(Path)}:
    import os
    import math
    import uuid
    from tools.stdio import StdIO
    from tools.minimize import minimize_batch

    content_io = StdIO(json_path=content_json)

    batch_size = 2500
    real_batch_size = math.ceil(len(content_io)/math.ceil(len(content_io)/batch_size))
    content_items = list(content_io.to_dict().items())
    for i in range(0, len(content_items), real_batch_size):
        sub_content_items = content_items[i:min(len(content_items), i+real_batch_size)]
        print(f"i: {i}; sub len: {len(sub_content_items)}")
        sub_name_content_list = [(f"{name}_pose{ind}", content) for name, one_content_dict in sub_content_items for ind, content in enumerate(one_content_dict["contents"])]
        sub_name_list = [item[0] for item in sub_name_content_list]
        sub_content_list = [item[1] for item in sub_name_content_list]
        refined_content_list = minimize_batch(str(receptor_pdb), sub_content_list)
        assert len(sub_name_list) == len(refined_content_list), "name and refined content length not match"
        for j, refined_content in enumerate(refined_content_list):
            name_pose = sub_name_list[j]
            name, _, pose_id = name_pose.rpartition("_pose")
            pose_id = int(pose_id)
            content_io.update_by_name_and_pose(name, pose_id, refined_content)

    result_path = Path(f"content_{uuid.uuid4().hex}.json")
    content_io.write_content_json(result_path)

    return OPIO({"content_json": result_path})
