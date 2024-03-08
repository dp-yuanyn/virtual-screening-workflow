from typing import Tuple
import os


def read_score_list_from_file(ligand_file:str, score_key:str="origin_score", constrain_type:str="") -> Tuple[str, list]:
    name = os.path.splitext(os.path.basename(ligand_file))[0].rstrip("_out")
    with open(ligand_file, "r") as f:
        content = f.read()
    content_list = []
    curr_conf_content = ""
    for line in content.split("\n"):
        curr_conf_content += line + "\n"
        if line.startswith("$$$$"):
            content_list.append(curr_conf_content)
            curr_conf_content = ""
    meta_list = []
    for i, content in enumerate(content_list):
        score_line_ind = None
        score = None
        content_line = content.split("\n")
        for j, line in enumerate(content_line):
            if line.startswith("> <Uni-Dock RESULT>"):
                score_line_ind = j + 1
                break
        if score_line_ind:
            score_line = content_line[score_line_ind]
            score = float(score_line.partition("LOWER_BOUND=")[0][len("ENERGY="):])
        meta_item = {"name": name, "conf_id": i, score_key: score}
        if constrain_type:
            meta_item["constrain_type"] = constrain_type
        meta_list.append(meta_item)
    return (name, meta_list)