from typing import List
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
from dflow.python import (
    PythonOPTemplate, 
)
from dflow.utils import randstr

from ops.op_strain_energy import strain_energy_op
from ops.op_helper import filter_ligands_op


def postdocking_filter_superop(filter_config, image_dict, executor_dict, volumes_dict):
    filter_pipeline = DAG(name=f"postdocking-filter-superop-{randstr(5)}")
    filter_pipeline.inputs.artifacts = {
        "input_receptor": InputArtifact(),
        "content_json": InputArtifact(),
        "meta_json": InputArtifact(),
    }
    filter_pipeline.inputs.parameters = {
        "ifp_filter_list": InputParameter(),
    }
    input_receptor = filter_pipeline.inputs.artifacts["input_receptor"]
    content_json = filter_pipeline.inputs.artifacts["content_json"]
    meta_json = filter_pipeline.inputs.artifacts["meta_json"]
    ifp_filter_list = filter_pipeline.inputs.parameters["ifp_filter_list"]

    filter_methods = filter_config["filter_methods"]
    filter_meta_art_list = []
    filter_tag_name_list = []
    for filter_method in filter_methods:
        if filter_method == "strain_energy":
            strain_energy_image = image_dict.get("gnina_image", "")
            strain_energy_step = Task(
                name="strain-energy-step",
                artifacts={
                    "content_json": content_json, 
                    "meta_json": meta_json,
                },
                template=PythonOPTemplate(
                    strain_energy_op,
                    image=strain_energy_image,
                    image_pull_policy="IfNotPresent",
                    volumes=volumes_dict.get("volumes"),
                    mounts=volumes_dict.get("mounts"),
                ),
                executor=executor_dict["cpu"],
            )
            filter_meta_art_list.append(strain_energy_step.outputs.artifacts["meta_json"])
            filter_pipeline.add(strain_energy_step)

    if filter_methods:
        filter_step = Task(
            name="filter-step",
            artifacts={
                "content_json": content_json, 
                "filtered_meta_json_list": filter_meta_art_list,
            },
            parameters={
                "filter_key_list": filter_tag_name_list,
            },
            template=PythonOPTemplate(
                filter_ligands_op,
                image="python:3.10",
                image_pull_policy="IfNotPresent",
                limits={"memory": "4Gi"},
                volumes=volumes_dict.get("volumes"),
                mounts=volumes_dict.get("mounts"),
                retry_on_failure_and_error=3,
            ),
            executor=executor_dict["local"],
        )
        content_json = filter_step.outputs.artifacts["content_json"]
        meta_json = filter_step.outputs.artifacts["meta_json"]
        filter_pipeline.add(filter_step)

    filter_pipeline.outputs.artifacts = {
        "content_json": OutputArtifact(_from=content_json), 
        "meta_json": OutputArtifact(_from=meta_json),
    }
    return filter_pipeline