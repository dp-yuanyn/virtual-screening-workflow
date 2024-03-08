from typing import List
import copy
from dflow import (
    Task,
    DAG,
    Step,
    Steps,
    InputArtifact,
    InputParameter,
    OutputArtifact,
    OutputParameter,
)
from dflow.io import ArgoVar
from dflow.python import (
    PythonOPTemplate, 
)
from dflow.utils import randstr

from ops.op_unidock import gen_ad4_map_op, run_unidock_op, run_unidock_score_only_op
from ops.op_rescore import rescore_gnina_op, rescore_RTMScore_op, merge_score_op
from ops.op_karmadock import karmadock_op
from ops.op_pocket import get_pocket_op
from superops.constrain import constrain_superop
from superops.filter import postdocking_filter_superop


def docking_superop(docking_config:dict, image_dict:dict, 
                    executor_dict:dict, volumes_dict:dict, constrain_type:str=""
) -> DAG:
    mgltools_image = image_dict.get("mgltools_image", "")
    unidock_tools_image = image_dict.get("unidock_tools_image", "")
    unidock_params = docking_config.get("unidock_params", dict())

    docking_superop = DAG(name=f"docking-superop-{randstr(5)}")
    docking_superop.inputs.artifacts = {
        "input_receptor": InputArtifact(),
        "prepared_receptor": InputArtifact(),
        "ligand_content_json": InputArtifact(),
        "meta_json": InputArtifact(optional=True),
        "bias_file": InputArtifact(optional=True),
        "bias_content_json": InputArtifact(optional=True)
    }
    receptor_pdbqt = docking_superop.inputs.artifacts["prepared_receptor"]
    ligand_content_json = docking_superop.inputs.artifacts["ligand_content_json"]
    meta_json = docking_superop.inputs.artifacts["meta_json"]
    bias_file = docking_superop.inputs.artifacts["bias_file"]
    bias_content_json = docking_superop.inputs.artifacts["bias_content_json"]

    if unidock_params["scoring"] == "ad4":
        gen_ad4_map_step = Task(
            name="gen-ad4-map",
            artifacts={
                "receptor": receptor_pdbqt,
                "content_json": ligand_content_json,
            },
            parameters={
                "docking_params":unidock_params
            },
            template=PythonOPTemplate(
                gen_ad4_map_op,
                image=mgltools_image,
                image_pull_policy="IfNotPresent",
                volumes=volumes_dict.get("volumes"),
                mounts=volumes_dict.get("mounts"),
            ),
            executor=executor_dict["local"],
        )
        docking_superop.add(gen_ad4_map_step)
        receptor_pdbqt = gen_ad4_map_step.outputs.artifacts["map_dir"]

    unidock_score_key = "origin_score"
    rescore_key = ""
    if constrain_type:
        unidock_score_key = "constrained_score"
        rescore_key = "origin_score"
        if not docking_config.get("rescore"):
            docking_config["rescore"] = unidock_params["scoring"]

    unidock_step = Task(
        name="unidock",
        artifacts={
            "receptor": receptor_pdbqt, 
            "ligand_content_json": ligand_content_json,
            "meta_json": meta_json,
            "bias_file": bias_file,
            "bias_content_json": bias_content_json, 
        },
        parameters={
            "docking_params": unidock_params,
            "score_key": unidock_score_key,
            "constrain_type": constrain_type,
        },
        template=PythonOPTemplate(
            run_unidock_op,
            image=unidock_tools_image,
            image_pull_policy="IfNotPresent",
            volumes=volumes_dict.get("volumes"),
            mounts=volumes_dict.get("mounts"),
        ),
        executor=executor_dict["gpu_unidock"]
    )
    docking_superop.add(unidock_step)

    content_json = unidock_step.outputs.artifacts["content_json"]
    meta_json = unidock_step.outputs.artifacts["meta_json"]

    if docking_config.get("rescore"):
        docking_config["rescore_unidock_params"] = unidock_params
        task_list = get_rescore_step(rescore_config=docking_config, 
                                     image_dict=image_dict, 
                                     executor_dict=executor_dict,
                                     volumes_dict=volumes_dict,
                                     receptor_pdb=docking_superop.inputs.artifacts["input_receptor"],
                                     receptor_pdbqt=receptor_pdbqt, 
                                     content_json=content_json, 
                                     meta_json=meta_json,
                                     score_tag=rescore_key)
        task = task_list[0]
        docking_superop.add(task)
        meta_json = task.outputs.artifacts["meta_json"]

    docking_superop.outputs.artifacts["content_json"] = OutputArtifact(_from=content_json)
    docking_superop.outputs.artifacts["meta_json"] = OutputArtifact(_from=meta_json)

    return docking_superop


def docking_rescore_superop(params:dict, image_dict:dict, 
                            executor_dict:dict, volumes_dict:dict) -> DAG:
    mgltools_image = image_dict.get("mgltools_image", "")
    rescore_config = params.get("rescore_config", dict())

    docking_rescore_superop = DAG(name=f"docking-rescore-superop-{randstr(5)}")
    docking_rescore_superop.inputs.artifacts = {
        "receptor_pdb": InputArtifact(),
        "receptor_pdbqt": InputArtifact(),
        "content_json": InputArtifact(),
        "meta_json": InputArtifact(optional=True),
    }

    content_json = docking_rescore_superop.inputs.artifacts["content_json"]
    meta_json = docking_rescore_superop.inputs.artifacts["meta_json"]
    receptor_pdb = docking_rescore_superop.inputs.artifacts["receptor_pdb"]
    receptor_pdbqt = docking_rescore_superop.inputs.artifacts["receptor_pdbqt"]

    if len(rescore_config.get("rescore_method", [])) == 0:
        return docking_rescore_superop
    # rescore_method_list = docking_config["rescore_config"]["rescore_method"]
    rescore_config["unidock_params"] = rescore_config["rescore_unidock_params"]
    meta_json_list = []
    step_name_list = []
    for rescore_method in rescore_config["rescore_method"]:
        rescore_config["rescore"] = rescore_method
        if rescore_method == "ad4":
            gen_ad4_map_step = Task(
                name="gen-ad4-map",
                artifacts={
                    "receptor": receptor_pdbqt,
                    "content_json": content_json,
                },
                parameters={
                    "docking_params": rescore_config["rescore_unidock_params"]
                },
                template=PythonOPTemplate(
                    gen_ad4_map_op,
                    image=mgltools_image,
                    image_pull_policy="IfNotPresent",
                ),
                executor=executor_dict["local"],
            )
            docking_rescore_superop.add(gen_ad4_map_step)
            receptor_pdbqt = gen_ad4_map_step.outputs.artifacts["map_dir"]
        rescore_step_list = get_rescore_step(rescore_config=rescore_config, 
                                             image_dict=image_dict, 
                                             executor_dict=executor_dict, 
                                             volumes_dict=volumes_dict,
                                             receptor_pdb=receptor_pdb,
                                             receptor_pdbqt=receptor_pdbqt, 
                                             content_json=content_json, 
                                             meta_json=meta_json)
        for rescore_step in rescore_step_list:
            docking_rescore_superop.add(rescore_step)
        meta_json_list.append(rescore_step.outputs.artifacts["meta_json"])
        step_name_list.append(rescore_step.name)

    merge_step = Task(
        name="mergescore-step",
        artifacts={
            "meta_json_list": meta_json_list
        },
        template=PythonOPTemplate(
            merge_score_op,
            image="python:3.10",
            image_pull_policy="IfNotPresent",
            limits={"memory": "8Gi"},
            volumes=volumes_dict.get("volumes"),
            mounts=volumes_dict.get("mounts"),
        ),
        executor=executor_dict["local"],
    )
    docking_rescore_superop.add(merge_step)
    meta_json = merge_step.outputs.artifacts["meta_json"]
    docking_rescore_superop.outputs.artifacts["meta_json"] = OutputArtifact(_from=meta_json)
    return docking_rescore_superop


def get_rescore_step(rescore_config:dict, image_dict:dict, 
                     executor_dict:dict, volumes_dict:dict,
                     receptor_pdb:ArgoVar, receptor_pdbqt:ArgoVar, 
                     content_json:ArgoVar, meta_json:ArgoVar=None,
                     score_tag:str="",
) -> List[Task]:
    unidock_tools_image = image_dict.get("unidock_tools_image", "")
    gnina_image = image_dict.get("gnina_image", "")
    pymol_image = image_dict.get("pymol_image", "")
    rtmscore_image = image_dict.get("rtmscore_image", "")
    karma_image = image_dict.get("karma_dock_image", "")
    if not karma_image:
        karma_image = image_dict.get("karmadock_image", "")

    rescore_method = rescore_config["rescore"]

    unidock_params = rescore_config.get("rescore_unidock_params", dict())

    task_list = []
    if rescore_method in ["vina", "vinardo", "ad4"]:
        unidock_params["scoring"] = rescore_method
        rescore_step = Task(
            name=f"{rescore_method}-rescore-step-{randstr(5)}",
            artifacts={
                "receptor": receptor_pdbqt, 
                "content_json": content_json,
                "meta_json": meta_json,
            },
            parameters={
                "docking_params": copy.deepcopy(unidock_params),
                "score_tag": score_tag,
            },
            template=PythonOPTemplate(
                run_unidock_score_only_op,
                image=unidock_tools_image,
                image_pull_policy="IfNotPresent",
                volumes=volumes_dict.get("volumes"),
                mounts=volumes_dict.get("mounts"),
            ),
            executor=executor_dict["cpu"]
        )
    elif rescore_method == "gnina": 
        rescore_step = Task(
            name="gnina-rescore-step",
            artifacts={
                "receptor": receptor_pdbqt, 
                "content_json": content_json,
                "meta_json": meta_json,
            },
            template=PythonOPTemplate(
                rescore_gnina_op,
                image=gnina_image,
                image_pull_policy="IfNotPresent",
                volumes=volumes_dict.get("volumes"),
                mounts=volumes_dict.get("mounts"),
            ),
            executor=executor_dict["gpu"]
        )
    elif rescore_method == "rtmscore":
        rtmscore_params = rescore_config.get("rtmscore_params", dict())
        get_pocket_step = Task(
            name = "get-pocket-step",
            artifacts={
                "receptor": receptor_pdb
            },
            parameters = {
                "docking_params": unidock_params
            },
            template=PythonOPTemplate(
                get_pocket_op,
                image=pymol_image,
                image_pull_policy="IfNotPresent",
            ),
            executor=executor_dict["local"]    
        )
        pocket_file = get_pocket_step.outputs.artifacts["pocket"]
        task_list.append(get_pocket_step)
        rescore_step = Task(
            name = "RTMscore-step",
            artifacts={
                "pocket": pocket_file, 
                "content_json": content_json,
                "meta_json": meta_json,
            },
            parameters = {
                "sdf_batch": rtmscore_params.get("sdf_batch", 1800),
                "batch": rtmscore_params.get("batch_size", 384)
            },
            template=PythonOPTemplate(
                rescore_RTMScore_op,
                image=rtmscore_image,
                image_pull_policy="IfNotPresent",
                volumes=volumes_dict.get("volumes"),
                mounts=volumes_dict.get("mounts"),
            ),
            executor=executor_dict["gpu"]
        )
    elif rescore_method == "karmadock":
        rescore_step = Task(
            name="karma-rescore-step",
            artifacts={
                "receptor": receptor_pdb, 
                "content_json": content_json,
                "meta_json": meta_json,
            },
            template=PythonOPTemplate(
                karmadock_op,
                image=karma_image,
                image_pull_policy="IfNotPresent",
                volumes=volumes_dict.get("volumes"),
                mounts=volumes_dict.get("mounts"),
            ),
            executor=executor_dict["gpu"]
        )
    else:
        raise KeyError("Invalid score method")
    task_list.append(rescore_step)
    return task_list


def docking_pipeline_superop(docking_config:dict, image_dict:dict, 
                             executor_dict:dict, volumes_dict:dict
) -> DAG:
    unidock_pipeline = DAG(name=f"unidock-pipeline-superop-{randstr(5)}")
    unidock_pipeline.inputs.artifacts = {
        "input_receptor": InputArtifact(),
        "prepared_receptor": InputArtifact(),
        "ligands_json": InputArtifact(),
        "ref_sdf_file": InputArtifact(optional=True),
    }
    unidock_pipeline.inputs.parameters = {
        "ifp_filter_list": InputParameter()
    }
    ligands_json = unidock_pipeline.inputs.artifacts["ligands_json"]
    input_receptor = unidock_pipeline.inputs.artifacts["input_receptor"]
    prepared_receptor = unidock_pipeline.inputs.artifacts["prepared_receptor"]
    ref_sdf_file = unidock_pipeline.inputs.artifacts["ref_sdf_file"]

    constrain_type = ""
    bias_content_json = None
    bias_file = None
    meta_json = None
    if docking_config.get("constained_params") and docking_config["constained_params"].get("do_constrain", True):
        constrained_params = docking_config["constained_params"]
        constrained_params["scoring"] = docking_config["unidock_params"].get("scoring", "")
        constrain_step = Task(
            name="constrain-step",
            artifacts={
                "receptor_pdb": input_receptor,
                # "meta_json": None,
                "content_json": ligands_json,
                "ref_sdf_file": ref_sdf_file,
            },
            template=constrain_superop(constrained_params, image_dict, executor_dict, volumes_dict)
        )
        unidock_pipeline.add(constrain_step)
        bias_file = constrain_step.outputs.artifacts["bias_file"]
        bias_content_json = constrain_step.outputs.artifacts["bias_content_json"]
        meta_json = constrain_step.outputs.artifacts["meta_json"]
        constrain_type = docking_config["constained_params"].get("constrained_type", "")

    docking_step = Task(
        name="docking-step",
        artifacts={
            "input_receptor": input_receptor,
            "prepared_receptor": prepared_receptor,
            "ligand_content_json": ligands_json,
            "meta_json": meta_json,
            "bias_file": bias_file,
            "bias_content_json": bias_content_json,
        },
        template=docking_superop(docking_config, image_dict, executor_dict, volumes_dict, constrain_type)
    )
    unidock_pipeline.add(docking_step)

    content_json = docking_step.outputs.artifacts["content_json"]
    meta_json = docking_step.outputs.artifacts["meta_json"]

    if docking_config.get("filter_config"):
        filter_config = docking_config["filter_config"]
        filter_step = Task(
            name="filter-step",
            artifacts={
                "input_receptor": input_receptor,
                "content_json": content_json, 
                "meta_json": meta_json,
            },
            parameters={
                "ifp_filter_list": unidock_pipeline.inputs.parameters["ifp_filter_list"]
            },
            template=postdocking_filter_superop(filter_config, image_dict, executor_dict, volumes_dict)
        )
        unidock_pipeline.add(filter_step)
        content_json = filter_step.outputs.artifacts["content_json"]
        meta_json = filter_step.outputs.artifacts["meta_json"]

    unidock_pipeline.outputs.artifacts["content_json"] = OutputArtifact(_from=content_json)
    unidock_pipeline.outputs.artifacts["meta_json"] = OutputArtifact(_from=meta_json)
    return unidock_pipeline