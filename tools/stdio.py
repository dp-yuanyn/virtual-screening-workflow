from typing import Optional, List, Dict
from os import PathLike
from pathlib import Path
import os
import json
import math


class StdIO:
    # including ligprep results and bias results
    def __init__(self, json_path:Optional[Path]=None):
        self.content_dict = dict()
        if json_path:
            with open(json_path, "r") as f:
                self.content_dict = json.load(f)

    def __len__(self) -> int:
        return len(self.content_dict)

    def to_dict(self) -> dict:
        return self.content_dict

    def get_keys(self) -> List[str]:
        return list(self.content_dict.keys())

    def get_content_list(self, key:str, content_tag:str="contents") -> list:
        return self.content_dict[key][content_tag]

    def _split_sdf_content(self, sdf_content:str) -> List[str]:
        content_list = []
        curr_conf_content = ""
        for line in sdf_content.split("\n"):
            curr_conf_content += line + "\n"
            if line.startswith("$$$$"):
                content_list.append(curr_conf_content)
                curr_conf_content = ""
        return content_list

    def insert_records(self, insert_dict:dict):
        self.content_dict.update(insert_dict)

    def save_content_by_input_files(self, content_files:List[str], 
                                    content_tag:str="", split_pose:bool=False, 
                                    removed_suffix:str="_out"):
        for content_file in content_files:
            name = os.path.splitext(os.path.basename(content_file))[0]
            if removed_suffix and name.endswith(removed_suffix):
                name = name[:-len(removed_suffix)]
            with open(content_file, "r") as f:
                content = f.read()
            if content:
                if split_pose:
                    content_list = self._split_sdf_content(content)
                else:
                    content_list = [content]
                if not self.content_dict.get(name):
                    self.content_dict[name] = dict()
                if not content_tag:
                    content_tag = "contents"
                self.content_dict[name][content_tag] = content_list

    def append_content(self, name:str, content:str, content_tag:str="contents"):
        if not self.content_dict.get(name):
            self.content_dict[name] = dict()
        if not self.content_dict[name].get(content_tag):
            self.content_dict[name][content_tag] = []
        self.content_dict[name][content_tag].append(content)

    def update_by_name_and_pose(self, name:str, pose_id:int, 
                                content:str, content_tag:str="contents"):
        self.content_dict[name][content_tag][pose_id] = content

    def write_content_json(self, saved_path:str):
        with open(saved_path, "w") as f:
            json.dump(self.content_dict, f)

    def write_raw_content_to_file(self, saved_dir:PathLike, 
                              content_tag:str="", 
                              fmt:str="sdf"):
        file_path_list = []
        for name, content_item in self.content_dict.items():
            if not content_tag:
                content_tag = list(content_item.keys())[0]
            content_list = content_item[content_tag]
            content = content_list[0]
            file_path = os.path.join(saved_dir, f"{name}.{fmt}")
            with open(file_path, "w") as f:
                f.write(content)
            file_path_list.append(file_path)
        return file_path_list

    def write_pose_content_to_file(self, saved_dir:PathLike, 
                              content_tag:str="", 
                              fmt:str="sdf",
                              split_pose:bool=True) -> List[str]:
        file_path_list = []
        for name, content_item in self.content_dict.items():
            if not content_tag:
                content_tag = list(content_item.keys())[0]
            content_list = content_item[content_tag]
            if split_pose:
                for i, content in enumerate(content_list):
                    file_path = os.path.join(saved_dir, f"{name}_pose{i}.{fmt}")
                    with open(file_path, "w") as f:
                        f.write(content)
                    file_path_list.append(file_path)
            else:
                file_path = os.path.join(saved_dir, f"{name}.{fmt}")
                with open(file_path, "w") as f:
                    f.write("".join(content_list))
                file_path_list.append(file_path)
        return file_path_list

    def write_sdf_content_to_big_file(self, saved_dir:PathLike, batch_size:int=6000, overwrite_head_name:bool=False):
        os.makedirs(saved_dir, exist_ok=True)
        real_batch_size = math.ceil(len(self.content_dict)/math.ceil(len(self.content_dict)/batch_size))
        batch_items = [list(self.content_dict.items())[i:i+real_batch_size] for i in range(0, len(self.content_dict), real_batch_size)]
        sdf_files = []
        batch_names = []
        for batch_ind, sub_items in enumerate(batch_items):
            sub_names = []
            sdf_path = os.path.join(saved_dir, f"sub-{batch_ind}.sdf")
            batch_content = ""
            for name, content_dict in sub_items:
                content_tag = list(content_dict.keys())[0]
                content_list = content_dict[content_tag]
                for pose_id, content in enumerate(content_list):
                    if overwrite_head_name:
                        content = "\n".join([f"{name}_pose{pose_id}"] + content.split("\n")[1:])
                    batch_content += content
                    sub_names.append(f"{name}_pose{pose_id}")
            with open(sdf_path, "w") as f:
                f.write(batch_content)
            sdf_files.append(sdf_path)
            batch_names.append(sub_names)
        return sdf_files, batch_names

    def iter_best_content_list(self) -> str:
        for content_item in self.content_dict.values():
            content_list = content_item[list(content_item.keys())[0]]
            best_content = content_list[0]
            yield best_content

    def keep_by_key(self, key_list:List[str]):
        self.content_dict = dict([(k, self.content_dict[k]) for k in key_list if self.content_dict.get(k)])

    def keep_by_key_and_pose(self, key_pose_list_map:Dict[str, List[int]]):
        content_dict = dict([(k, self.content_dict[k]) for k in list(key_pose_list_map.keys()) if self.content_dict.get(k)])
        for name in list(content_dict.keys()):
            one_ligand_content_dict = content_dict[name]
            content_tag = list(one_ligand_content_dict.keys())[0]
            content_list = one_ligand_content_dict[content_tag]
            remain_pose_id_list = key_pose_list_map[name]
            remain_content_list = []
            for i, content in enumerate(content_list):
                if i in remain_pose_id_list:
                    remain_content_list.append(content)
            content_dict[name] = {content_tag: remain_content_list}
        self.content_dict = content_dict
#### rongfeng.zou ####    
    def get_content_and_namelist_from_multipose(self):
        content_str = ""
        name_list = []
        for name, ligand_content_dict in self.to_dict().items():
            content_list = list(ligand_content_dict.values())[0]
            for i, content in enumerate(content_list):
                name_list.append(f"{name}-{i}.sdf")
                content_str += content
        return content_str, name_list
####


class MetaIO:
    def __init__(self, json_path:Optional[Path]=None, meta_dict:Optional[dict]=None):
        self.meta_dict = dict()
        if json_path:
            with open(json_path, "r") as f:
                self.meta_dict = json.load(f)
        elif meta_dict:
            self.meta_dict = meta_dict

    def __len__(self) -> int:
        return len(self.meta_dict)

    def to_dict(self) -> dict:
        return self.meta_dict

    def has_key(self, key:str) -> bool:
        return self.meta_dict.get(key) is not None

    def get(self, key:str, default:object=None):
        return self.meta_dict.get(key, default)

    # for empty meta dict to add records
    def add_meta(self, key:str, meta_item:dict):
        if key not in self.meta_dict:
            self.meta_dict[key] = []
        self.meta_dict[key].append(meta_item)

    # for empty meta dict to add records
    def add_meta_list(self, key:str, meta_list:list):
        if key not in self.meta_dict:
            self.meta_dict[key] = []
        self.meta_dict[key].extend(meta_list)

    # for parallel meta dict to merge
    def merge_meta_json(self, meta_file:PathLike):
        with open(meta_file, "r") as f:
            meta_dict = json.load(f)
        self.meta_dict.update(meta_dict)

    # for parallel meta dict to merge
    def insert_records(self, meta_dict:dict):
        self.meta_dict.update(meta_dict)

    def write_json(self, saved_path:PathLike):
        with open(saved_path, "w") as f:
            json.dump(self.meta_dict, f)

    def init_key(self, key:str, default_value:object):
        for name in self.meta_dict.keys():
            for i in range(len(self.meta_dict[name])):
                self.meta_dict[name][i].update({key: default_value})

    # for meta dict to update single(no-pose) meta info given name
    def update_ligand_by_key(self, key:str, item:dict):
        for i in range(len(self.meta_dict[key])):
            self.meta_dict[key][i].update(item)

    def update_pose_by_dict(self, key:str, pose_id:int, update_dict:dict):
        self.meta_dict[key][pose_id].update(update_dict)

    def update_pose_by_key_value(self, item_name:str, item_pose:int, key:str, value:object):
        self.meta_dict[item_name][item_pose][key] = value

    # for meta dict to update list(pose) meta info dict
    def update_by_meta_dict(self, meta_dict:dict):
        for name, one_meta_list in meta_dict.items():
            if self.meta_dict.get(name):
                for i in range(len(self.meta_dict[name])):
                    update_ind = i
                    if i >= len(one_meta_list):
                        update_ind = -1
                    self.meta_dict[name][i].update(one_meta_list[update_ind])

    # for meta dict to update list(pose) meta info file
    def update_by_meta_json(self, meta_file:PathLike):
        with open(meta_file, "r") as f:
            meta_dict = json.load(f)
        self.update_by_meta_dict(meta_dict)

    def to_tuple_list_by_key_list(self, key_list:list) -> List[tuple]:
        return [tuple(item.get(key) for key in key_list) for ligand_list in self.meta_dict.values() for item in ligand_list]

    def get_best_tuple_list_by_key(self, key_list:list) -> List[tuple]:
        return [tuple(ligand_list[0].get(key) for key in key_list) for ligand_list in self.meta_dict.values()]

    def get_sorted_list(self, sort_key:str, reverse:bool=False):
        item_list = [(k, v) for k, v in self.meta_dict.items()]
        return sorted(item_list, key=lambda x:sorted([item.get(sort_key, 0) for item in x[1]], reverse=reverse)[0], reverse=reverse)

    def add_rank_by_sorted_key(self, sort_key:str, reverse:bool=False, rank_prefix:str="ranknum_"):
        item_list = [(name, ind, ligand_meta_list[ind]) for name, ligand_meta_list in self.meta_dict.items() \
                     for ind in range(len(ligand_meta_list))]
        sorted_item_list = sorted(item_list, key=lambda x:x[2][sort_key], reverse=reverse)
        rank_key = rank_prefix + sort_key
        for i, tpl in enumerate(sorted_item_list):
            name, pose_id, _ = tpl
            self.meta_dict[name][pose_id][rank_key] = i + 1

    def keep_top_num(self, top_num:int, sort_key:str, reverse_flag:bool=False):
        sorted_list = self.get_sorted_list(sort_key, reverse = reverse_flag)[:min(top_num, len(self.meta_dict))]
        self.meta_dict = dict(sorted_list)

    def keep_top_percent(self, top_percent:float, sort_key:str, reverse_flag:bool=False):
        sorted_list = self.get_sorted_list(sort_key, reverse = reverse_flag)[:int(max(1, len(self.meta_dict)*top_percent))]
        self.meta_dict = dict(sorted_list)

    def filter_by_key_only_best(self, filter_key:str):
        removed_name_list = []
        for name, ligand_meta_list in self.meta_dict.items():
            ligand_meta_dict = ligand_meta_list[0]
            if filter_key in ligand_meta_dict and not ligand_meta_dict[filter_key]:
                removed_name_list.append(name)
        for removed_name in removed_name_list:
            self.meta_dict.pop(removed_name)

    def filter_items_by_key(self, filter_key:str):
        keep_name_conf_tuple_list = []
        for name, ligand_meta_list in self.meta_dict.items():
            for i, ligand_meta_dict in enumerate(ligand_meta_list):
                if filter_key in ligand_meta_dict and not ligand_meta_dict[filter_key]:
                    continue
                keep_name_conf_tuple_list.append((name, i))
        meta_dict = dict()
        for name, ind in keep_name_conf_tuple_list:
            if name not in meta_dict:
                meta_dict[name] = []
            meta_dict[name].append(self.meta_dict[name][ind])
        self.meta_dict = meta_dict
