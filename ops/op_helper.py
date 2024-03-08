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
def collect_parallel_meta_op(
        meta_json_list:Artifact(List[Path]),
) -> {"meta_json":Artifact(Path)}:
    from tools.stdio import StdIO, MetaIO

    meta_io = MetaIO()
    for i in range(len(meta_json_list)):
        meta_io.merge_meta_json(meta_json_list[i])
    result_meta_path = Path("/tmp/meta.json")
    meta_io.write_json(result_meta_path)
    
    return OPIO({"meta_json": result_meta_path})

@OP.function
def resplit_op(
        meta_json:Artifact(Path),
        content_json_list:Artifact(List[Path]),
        batch_size:Parameter(int),
        only_best:Parameter(bool, default=False),
) -> {"meta_json_list":Artifact(List[Path], archive=None), 
      "content_json_list":Artifact(List[Path], archive=None)}:
    import os
    import math
    from tools.logger import get_logger
    from tools.stdio import MetaIO, StdIO

    logger = get_logger()

    meta_io = MetaIO(json_path=meta_json)
    real_batch_size = math.ceil(len(meta_io)/math.ceil(len(meta_io)/batch_size))

    result_content_json_list = []
    result_meta_json_list = []
    curr_num = 0
    i = 0
    curr_result_meta_io = MetaIO()
    curr_result_content_io = StdIO()
    for content_json in content_json_list:
        content_io = StdIO(content_json)
        for name, one_ligand_content_dict in content_io.to_dict().items():
            if meta_io.has_key(name):
                curr_result_meta_io.insert_records({name: meta_io.get(name)})
                curr_result_content_io.insert_records({name: one_ligand_content_dict})
                curr_num += 1
                if curr_num == real_batch_size:
                    result_meta_json_path = Path(f"/tmp/meta_{i}.json")
                    curr_result_meta_io.write_json(result_meta_json_path)
                    result_meta_json_list.append(result_meta_json_path)
                    result_content_json_path = Path(f"/tmp/content_{i}.json")
                    curr_result_content_io.write_content_json(result_content_json_path)
                    result_content_json_list.append(result_content_json_path)
                    curr_result_meta_io = MetaIO()
                    curr_result_content_io = StdIO()
                    curr_num = 0
                    i += 1
    if curr_num > 0:
        result_meta_json_path = Path(f"/tmp/meta_{i}.json")
        curr_result_meta_io.write_json(result_meta_json_path)
        result_meta_json_list.append(result_meta_json_path)
        result_content_json_path = Path(f"/tmp/content_{i}.json")
        curr_result_content_io.write_content_json(result_content_json_path)
        result_content_json_list.append(result_content_json_path)

    logger.info(os.listdir("/tmp"))
    return OPIO({"meta_json_list": result_meta_json_list,
                 "content_json_list": result_content_json_list})


@OP.function
def filter_ligands_op(
        content_json:Artifact(Path), 
        filtered_meta_json_list:Artifact(List[Path]),
        filter_key_list:Parameter(List[str])
) -> {"content_json":Artifact(Path), "meta_json":Artifact(Path), "filter_info":Parameter(dict)}:
    import uuid
    from tools.stdio import StdIO, MetaIO

    meta_io = None
    for i, meta_json_path in enumerate(filtered_meta_json_list):
        if not meta_io:
            meta_io = MetaIO(json_path=meta_json_path)
        else:
            meta_io.update_by_meta_json(meta_json_path)

    filter_info = {"before_ligand_num": len(meta_io)}
    for filter_key in filter_key_list:
        before_num = len(meta_io)
        meta_io.filter_items_by_key(filter_key)
        after_num = len(meta_io)
        record_filter_name = filter_key.rstrip("_valid")
        filter_info[f"{record_filter_name}_filter_ligand_num"] = before_num - after_num
    filter_info["after_ligand_num"] = len(meta_io)

    key_pose_list_map = dict()
    for name, meta_list in meta_io.to_dict().items():
        for meta_dict in meta_list:
            pose_id = meta_dict["conf_id"]
            if name not in key_pose_list_map:
                key_pose_list_map[name] = []
            key_pose_list_map[name].append(pose_id)
    content_io = StdIO(content_json)
    content_io.keep_by_key_and_pose(key_pose_list_map)

    uid = uuid.uuid4().hex
    meta_path = Path(f"/tmp/meta_{uid}.json")
    meta_io.write_json(meta_path)
    content_path = Path(f"/tmp/content_{uid}.json")
    content_io.write_content_json(content_path)
    
    return OPIO({"content_json": content_path,
                 "meta_json": meta_path, 
                 "filter_info": filter_info})


@OP.function
def keep_best_top_ligands_op(
        meta_json:Artifact(Path),
        top_num:Parameter(int, default=None),
        top_ratio:Parameter(float, default=None),
        sort_key:Parameter(str),
        reverse:Parameter(bool, default=False),
) -> {"meta_json":Artifact(Path),
      "keep_num":Parameter(dict)}:
    import os
    from tools.logger import get_logger
    from tools.stdio import StdIO, MetaIO

    logger = get_logger()
    logger.info(f"CWD: {os.getcwd()}")

    keep_dict = dict()
    meta_io = MetaIO(meta_json)
    keep_dict["before_num"] = len(meta_io)
    if top_num:
        meta_io.keep_top_num(top_num, sort_key, reverse_flag=reverse)
    elif top_ratio:
        meta_io.keep_top_percent(top_ratio, sort_key, reverse_flag=reverse)
    keep_dict["after_num"] = len(meta_io)

    meta_path = Path("/tmp/meta.json")
    meta_io.write_json(meta_path)

    logger.info(os.listdir("/tmp"))

    return OPIO({"meta_json": meta_path,
                 "keep_num":keep_dict})


@OP.function
def keep_content_by_meta_op(
        content_json_list:Artifact(List[Path], archive=None), 
        meta_json:Artifact(Path),
) -> {"content_json_list":Artifact(List[Path])}:
    import uuid
    from tools.stdio import StdIO, MetaIO

    meta_io = MetaIO(meta_json)
    meta_dict = meta_io.to_dict()

    res_content_json_list = []
    for i, content_json in enumerate(content_json_list):
        content_io = StdIO(content_json)
        content_io.keep_by_key(list(meta_dict.keys()))
        content_json_path = Path(f"content_{i}.json")
        content_io.write_content_json(content_json_path)
        res_content_json_list.append(content_json_path)
    
    return OPIO({"content_json_list": res_content_json_list})


@OP.function
def save_retrieval_meta_op(
    meta_json: Artifact(Path),
    score_key_list: list[str],
) -> {"retrieval_json": Artifact(Path, global_name="retrieval_json")}:
    import json
    from tools.stdio import MetaIO

    score_key = score_key_list[0]
    meta_io = MetaIO(json_path=meta_json)
    retrieval_list = [{"ligand_id": int(name.rpartition("_")[0]), 
                       "source": int(name.rpartition("_")[2]), 
                       "score": meta_list[0][score_key]} for name, meta_list in meta_io.to_dict().items()]
    result_json_path = Path("/tmp/retrieval_results.json")
    with open(result_json_path, "w") as f:
        json.dump(retrieval_list, f)
    
    return OPIO({"retrieval_json": result_json_path})