from dflow import (
    Executor,
    Task,
    DAG,
    Step,
    InputArtifact,
    InputParameter,
    OutputArtifact,
)
from dflow.python import (
    PythonOPTemplate, 
    Slices,
)
from dflow.utils import randstr

from ops.op_pbgbsa import run_pbgbsa_op
from ops.op_helper import collect_parallel_meta_op, save_retrieval_meta_op
from superops.docking import docking_pipeline_superop, docking_rescore_superop
from superops.postprocess import postprocess_superop


def get_merge_parallel_step(meta_json_list:list[InputArtifact], 
                            local_executor:Executor,
                            volumes_dict:dict) -> Task:
    merge_meta_step = Task(
        name=f"merge-parallel-meta-step-{randstr()}",
        artifacts={
            "meta_json_list": meta_json_list,
        },
        template=PythonOPTemplate(
            collect_parallel_meta_op,
            image="python:3.10",
            image_pull_policy="IfNotPresent",
            requests={"memory": "1Gi"},
            limits={"memory": "64Gi"},
            volumes=volumes_dict.get("volumes"),
            mounts=volumes_dict.get("mounts"),
            retry_on_failure_and_error=3,
        ),
        executor=local_executor
    )
    return merge_meta_step


def parallel_docking_pipeline_superop(docking_config:dict, image_dict:dict, 
                                      executor_dict:dict, volumes_dict:dict) -> DAG:
    parallel_unidock_pipeline = DAG(name=f"parallel-unidock-pipeline-superop-{randstr()}")
    parallel_unidock_pipeline.inputs.artifacts = {
        "input_receptor": InputArtifact(),
        "prepared_receptor": InputArtifact(),
        "ligands_json_list": InputArtifact(archive=None),
        "ref_sdf_file": InputArtifact(optional=True),
    }
    parallel_unidock_pipeline.inputs.parameters = {
        "ifp_filter_list": InputParameter()
    }
    parallel_unidock_pipeline_step = Task(
        name="unidock-step",
        slices=Slices(sub_path=True, input_artifact=["ligands_json"], output_artifact=["content_json", "meta_json"]),
        artifacts={
            "input_receptor": parallel_unidock_pipeline.inputs.artifacts["input_receptor"],
            "prepared_receptor": parallel_unidock_pipeline.inputs.artifacts["prepared_receptor"],
            "ligands_json": parallel_unidock_pipeline.inputs.artifacts["ligands_json_list"],
            "ref_sdf_file": parallel_unidock_pipeline.inputs.artifacts["ref_sdf_file"],
        },
        parameters={
            "ifp_filter_list": parallel_unidock_pipeline.inputs.parameters["ifp_filter_list"],
        },
        template=docking_pipeline_superop(docking_config, image_dict, executor_dict, volumes_dict),
    )

    parallel_unidock_pipeline.add(parallel_unidock_pipeline_step)
    merge_parallel_step = get_merge_parallel_step(parallel_unidock_pipeline_step.outputs.artifacts["meta_json"], 
                                                  executor_dict["local"], volumes_dict)
    parallel_unidock_pipeline.add(merge_parallel_step)

    parallel_unidock_pipeline.outputs.artifacts["content_json_list"] = OutputArtifact(_from=parallel_unidock_pipeline_step.outputs.artifacts["content_json"])
    parallel_unidock_pipeline.outputs.artifacts["meta_json"] = OutputArtifact(_from=merge_parallel_step.outputs.artifacts["meta_json"])
    return parallel_unidock_pipeline


def parallel_rescore_superop(params:dict, image_dict:dict, 
                             executor_dict:dict, volumes_dict:dict
) -> DAG:
    parallel_rescore_pipeline = DAG(name=f"parallel-ranking-pipeline-superop-{randstr()}")
    parallel_rescore_pipeline.inputs.artifacts = {
        "receptor_pdb": InputArtifact(),
        "receptor_pdbqt": InputArtifact(),
        "content_json_list": InputArtifact(archive=None),
        "meta_json_list": InputArtifact(archive=None, optional=True),
    }
    parallel_rescore_step = Task(
        name="rescore-step",
        slices=Slices(sub_path=True, input_artifact=["content_json", "meta_json"], output_artifact=["meta_json"]),
        artifacts={
            "receptor_pdb": parallel_rescore_pipeline.inputs.artifacts["receptor_pdb"],
            "receptor_pdbqt": parallel_rescore_pipeline.inputs.artifacts["receptor_pdbqt"],
            "content_json": parallel_rescore_pipeline.inputs.artifacts["content_json_list"],
            "meta_json": parallel_rescore_pipeline.inputs.artifacts["meta_json_list"],
        },
        template=docking_rescore_superop(params, image_dict, executor_dict, volumes_dict),
    )
    parallel_rescore_pipeline.add(parallel_rescore_step)

    merge_parallel_step = get_merge_parallel_step(parallel_rescore_step.outputs.artifacts["meta_json"], 
                                                  executor_dict["local"], volumes_dict)
    parallel_rescore_pipeline.add(merge_parallel_step)

    parallel_rescore_pipeline.outputs.artifacts["meta_json"] = OutputArtifact(_from=merge_parallel_step.outputs.artifacts["meta_json"])
    return parallel_rescore_pipeline


def parallel_post_process_superop(params:dict, image_dict:dict, 
                                  executor_dict:dict, volumes_dict:dict) -> DAG:
    parallel_postprocess_pipeline = DAG(name=f"parallel-postprocess-superop-{randstr()}")
    parallel_postprocess_pipeline.inputs.artifacts = {
        "receptor_pdb": InputArtifact(),
        "content_json_list": InputArtifact(archive=None),
        "meta_json_list": InputArtifact(archive=None)
    }
    parallel_postprocess_step = Task(
        name="parallel-postprocess-pipeline-step",
        slices=Slices(sub_path=True, input_artifact=["content_json", "meta_json"], output_artifact=["content_json", "meta_json"]),
        artifacts={
            "receptor_pdb": parallel_postprocess_pipeline.inputs.artifacts["receptor_pdb"],
            "content_json": parallel_postprocess_pipeline.inputs.artifacts["content_json_list"],
            "meta_json": parallel_postprocess_pipeline.inputs.artifacts["meta_json_list"],
        },
        template=postprocess_superop(params, image_dict, executor_dict, volumes_dict)
    )
    parallel_postprocess_pipeline.add(parallel_postprocess_step)

    merge_parallel_step = get_merge_parallel_step(parallel_postprocess_step.outputs.artifacts["meta_json"], 
                                                  executor_dict["local"], volumes_dict)
    parallel_postprocess_pipeline.add(merge_parallel_step)

    parallel_postprocess_pipeline.outputs.artifacts["content_json_list"] = OutputArtifact(_from=parallel_postprocess_step.outputs.artifacts["content_json"])
    parallel_postprocess_pipeline.outputs.artifacts["meta_json"] = OutputArtifact(_from=merge_parallel_step.outputs.artifacts["meta_json"])
    return parallel_postprocess_pipeline


def parallel_pbgbsa_superop(params:dict, image_dict:dict, executor_dict:dict, volume_dict:dict) -> DAG:
    parallel_pbgbsa_superop = DAG(name=f"parallel-pbgbsa-superop-{randstr()}")
    parallel_pbgbsa_superop.inputs.artifacts = {
        "receptor_pdb": InputArtifact(),
        "content_json_list": InputArtifact(archive=None),
        "meta_json_list": InputArtifact(archive=None)
    }
    parallel_pbgbsa_step = Task(
        name="parallel-pbgbsa-pipeline-step",
        slices=Slices(sub_path=True, input_artifact=["content_json", "meta_json"], output_artifact=["meta_json"]),
        artifacts={
            "receptor_file": parallel_pbgbsa_superop.inputs.artifacts["receptor_pdb"],
            "content_json": parallel_pbgbsa_superop.inputs.artifacts["content_json_list"],
            "meta_json": parallel_pbgbsa_superop.inputs.artifacts["meta_json_list"],
        },
        parameters={
            "pbgbsa_params": params.get("pbgbsa_params", dict())
        },
        template=PythonOPTemplate(
            run_pbgbsa_op,
            image=image_dict.get("pbgbsa_image"),
            image_pull_policy="IfNotPresent",
            volumes=volume_dict.get("volumes"),
            mounts=volume_dict.get("mounts"),
        ),
        executor=executor_dict["cpu"],
    )
    parallel_pbgbsa_superop.add(parallel_pbgbsa_step)

    merge_parallel_step = get_merge_parallel_step(parallel_pbgbsa_step.outputs.artifacts["meta_json"], 
                                                  executor_dict["local"], volume_dict)
    parallel_pbgbsa_superop.add(merge_parallel_step)

    parallel_pbgbsa_superop.outputs.artifacts["meta_json"] = OutputArtifact(_from=merge_parallel_step.outputs.artifacts["meta_json"])
    return parallel_pbgbsa_superop