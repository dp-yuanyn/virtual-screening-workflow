from typing import List
import copy
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
from dflow.python import (
    Slices,
    PythonOPTemplate, 
)
from ops.op_read import read_ligands_op


def read_and_prepare_ligands_superop(config:dict, 
                                     image_dict:dict, 
                                     executor_dict:dict,
                                     volumes_dict:dict,
) -> DAG:
    ligprep_image = image_dict.get("ligprep_image", "")

    read_ligands_superop = DAG(name=f"read-ligands-superop")
    read_ligands_superop.inputs.artifacts = {
        "ligands_dir": InputArtifact(),
        "label_table": InputArtifact(optional=True),
    }

    read_ligands_step = Task(
        name="read-ligands-step",
        artifacts={
            "ligands_dir": read_ligands_superop.inputs.artifacts["ligands_dir"],
            "label_table": read_ligands_superop.inputs.artifacts["label_table"],
        },
        parameters={
            "batch_size": config.get("parallel_batch_size", 18000)
        },
        template=PythonOPTemplate(
            read_ligands_op,
            image=ligprep_image,
            image_pull_policy="IfNotPresent",
            requests={"memory": "1Gi"},
            limits={"memory": "16Gi"},
            volumes=volumes_dict.get("volumes"),
            mounts=volumes_dict.get("mounts"),
        ),
        executor=executor_dict["cpu"] if config.get("bohrium_dataset_key") else executor_dict["local"] 
    )
    read_ligands_superop.add(read_ligands_step)
    ligands_json_list = read_ligands_step.outputs.artifacts["ligands_json_list"]
    label_meta_json = read_ligands_step.outputs.artifacts["label_json"]
    
    read_ligands_superop.outputs.artifacts = {
        "ligands_json_list": OutputArtifact(_from=ligands_json_list, archive=None),
        "label_json": OutputArtifact(_from=label_meta_json, optional=True),
    }
    return read_ligands_superop