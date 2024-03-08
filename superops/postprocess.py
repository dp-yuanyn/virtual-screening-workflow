from typing import List
import copy
from argo.workflows.client import (
    V1Volume, 
    V1VolumeMount, 
    V1PersistentVolumeClaimVolumeSource
)
from dflow import (
    Executor,
    Step,
    Task,
    Steps,
    DAG,
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

from ops.op_ifp import ifp_v2_op
from ops.op_minimize import minimize_op


def postprocess_superop(params:dict, image_dict:dict, executor_dict:dict, volume_dict:dict) -> Steps:
    pose_refine_image = image_dict.get("minimize_image", "")
    ifp_image = image_dict.get("minimize_image", "")

    postprocess_superop = DAG(name=f"postprocess-superop-{randstr(5)}")
    postprocess_superop.inputs.artifacts = {
        "receptor_pdb": InputArtifact(),
        "content_json": InputArtifact(),
        "meta_json": InputArtifact(),
    }
    receptor_pdb = postprocess_superop.inputs.artifacts["receptor_pdb"]
    content_json = postprocess_superop.inputs.artifacts["content_json"]
    meta_json = postprocess_superop.inputs.artifacts["meta_json"]

    methods = params.get("methods", [])
    for method in methods:
        if method == "pose_refine":
            pose_refine_step = Task(
                name="pose-refine-step",
                artifacts={
                    "receptor_pdb": receptor_pdb,
                    "content_json": content_json,
                },
                template=PythonOPTemplate(
                    minimize_op,
                    image=pose_refine_image,
                    image_pull_policy="IfNotPresent",
                    volumes=volume_dict.get("volumes"),
                    mounts=volume_dict.get("mounts"),
                ),
                executor=executor_dict["cpu"],
            )
            postprocess_superop.add(pose_refine_step)
            content_json = pose_refine_step.outputs.artifacts["content_json"]

        if method == "ifp":
            ifp_step = Task(
                name="ifp-step",
                artifacts={
                    "receptor_pdb": receptor_pdb,
                    "content_json": content_json,
                    "meta_json": meta_json,
                },
                template=PythonOPTemplate(
                    ifp_v2_op,
                    image=ifp_image,
                    image_pull_policy="IfNotPresent",
                    volumes=volume_dict.get("volumes"),
                    mounts=volume_dict.get("mounts"),
                ),
                executor=executor_dict["cpu"],
            )
            postprocess_superop.add(ifp_step)
            meta_json = ifp_step.outputs.artifacts["meta_json"]
            content_json = ifp_step.outputs.artifacts["content_json"]

    postprocess_superop.outputs.artifacts["content_json"] = OutputArtifact(_from=content_json)
    postprocess_superop.outputs.artifacts["meta_json"] = OutputArtifact(_from=meta_json)

    return postprocess_superop