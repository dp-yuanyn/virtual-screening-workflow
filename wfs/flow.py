from pathlib import Path
import os
import sys
import copy
import json
import re
import argparse

from dflow import (
    Workflow, 
    DAG, 
    Task,
    S3Artifact,
    upload_artifact,
)
from dflow.python import (
    PythonOPTemplate, 
    Slices,
    upload_packages,
)
from dflow.plugins.bohrium import BohriumDatasetsArtifact

sys.path = [os.path.dirname(os.path.dirname(__file__))] + sys.path
from wfs.utils import (
    setup,
    generate_executor_dict,
    generate_volumes_dict
)

upload_packages.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "ops"))
from ops.op_preprocess import convert_receptor_op, preprocess_ligands_op
from ops.op_helper import resplit_op, keep_best_top_ligands_op
from superops.read import read_and_prepare_ligands_superop
from superops.parallel import parallel_docking_pipeline_superop, parallel_rescore_superop, parallel_post_process_superop, parallel_pbgbsa_superop
upload_packages.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools"))


score_method_tag_map = {"vina": "vina_score", "vinardo": "vinardo_score", "ad4": "ad4_score",
                        "gnina": "gnina_affinity", "rtmscore": "rtmscore", "karmadock": "karmadock_score"}


def run_nash_workflow(config:dict) -> Workflow:
    job_name = config.get("job_name", "vs_workflow")
    setup(config)

    remote_config = config.get("dispatcher_config", dict())
    scass_type_dict = config.get("scass_type_dict", dict())
    executor_dict = generate_executor_dict(remote_config, scass_type_dict, False)

    step_params_list = config.get("steps", [])
    image_dict = config.get("image_dict", dict())
    metric_params = config.get("metric", dict())
    volumn_config = config.get("volume_config", dict())
    volumes_dict = generate_volumes_dict(volumn_config)

    receptor_pdb = upload_artifact(config["receptor_file"])
    ref_ligand_artifact = None
    if config.get("ref_ligand_path"):
        ref_ligand_artifact = upload_artifact(config["ref_ligand_path"])

    label_metric_art = None
    if metric_params.get("label_metric_path"):
        label_metric_art = upload_artifact(metric_params["label_metric_path"])

    mgltools_image = image_dict.get("mgltools_image", "")
    ligprep_image = image_dict.get("ligprep_image", "")

    dag = DAG()
    convert_receptor_step = Task(
        name="convert-receptor",
        artifacts={
            "input_receptor": receptor_pdb, 
        },
        template=PythonOPTemplate(
            convert_receptor_op,
            image=mgltools_image,
            image_pull_policy="IfNotPresent",
        ),
        executor=executor_dict["local"]
    )
    dag.add(convert_receptor_step)
    receptor_pdbqt = convert_receptor_step.outputs.artifacts["output_receptor"]

    if config.get("bohrium_dataset_key"):
        ligands_dir = BohriumDatasetsArtifact(config["bohrium_dataset_key"])
    elif config.get("ligands_key"):
        ligands_dir = S3Artifact(key=config["ligands_key"])
    else:
        ligands_dir = upload_artifact(config["ligands_dir"])
    read_ligands_step = Task(
        name="read-ligands-pipeline-step",
        artifacts={
            "ligands_dir": ligands_dir,
            "label_table": label_metric_art,
        },
        template=read_and_prepare_ligands_superop(config, image_dict, executor_dict, volumes_dict),
    )
    dag.add(read_ligands_step)
    content_json_list = read_ligands_step.outputs.artifacts["ligands_json_list"]
    label_meta_json = read_ligands_step.outputs.artifacts["label_json"]

    parallel_ligand_preprocess_step = Task(
        name="parallel-ligand-preprocess-step",
        slices=Slices(sub_path=True, input_artifact=["ligands_json"], output_artifact=["content_json"]),
        artifacts={
            "ligands_json": content_json_list, 
        },
        template=PythonOPTemplate(
            preprocess_ligands_op,
            image=ligprep_image,
            image_pull_policy="IfNotPresent",
            limits={"cpu": "4096m", "memory": "8Gi"},
            volumes=volumes_dict.get("volumes"),
            mounts=volumes_dict.get("mounts"),
            retry_on_failure_and_error=3,
        ),
        executor=executor_dict["local"]
    )
    dag.add(parallel_ligand_preprocess_step)
    content_json_list = parallel_ligand_preprocess_step.outputs.artifacts["content_json"]

    rescore_name = None
    for step_ind, step_params in enumerate(step_params_list):
        name = step_params["name"]
        params = step_params["params"]
        scass_type = step_params.get("scass_type", "")
        scass_type_cpu = step_params.get("scass_type_cpu", "c16_m32_cpu")
        step_remote_config = copy.deepcopy(remote_config)
        if step_params.get("remote_extra"):
            step_remote_config["machine_dict"]["remote_profile"]["input_data"].update(step_params["remote_extra"])
        if not scass_type_dict:
            scass_type_dict = {"gpu": scass_type, "gpu_unidock": scass_type, "cpu": scass_type_cpu}
        executor_dict = generate_executor_dict(step_remote_config, scass_type_dict, False)

        if name == "score":
            unidock_pipeline_step = Task(
                name=f"parallel-unidock-pipeline-step-{step_ind}",
                artifacts={
                    "input_receptor": receptor_pdb,
                    "prepared_receptor": receptor_pdbqt,
                    "ligands_json_list": content_json_list,
                    "ref_sdf_file": ref_ligand_artifact,
                },
                parameters={
                    "ifp_filter_list": []
                },
                template=parallel_docking_pipeline_superop(params, image_dict, executor_dict, volumes_dict),
            )
            meta_json = unidock_pipeline_step.outputs.artifacts["meta_json"]
            content_json_list = unidock_pipeline_step.outputs.artifacts["content_json_list"]

            rescore_name = score_method_tag_map.get(params.get("rescore"), "origin_score")
            score_tag_list = [rescore_name]
            dag.add(unidock_pipeline_step)

        elif name == "rescore":
            score_tag_list = params['rescore_config']['rescore_method']
            parallel_rescore_step = Task(
                name=f"parallel-rescore-step-{step_ind}",
                artifacts={
                    "receptor_pdb": receptor_pdb,
                    "receptor_pdbqt": receptor_pdbqt,
                    "content_json_list": content_json_list,
                    "meta_json_list": meta_json_list,
                },
                template=parallel_rescore_superop(params, image_dict, executor_dict, volumes_dict),
            )
            meta_json = parallel_rescore_step.outputs.artifacts["meta_json"]
            dag.add(parallel_rescore_step)
            rescore_name = score_method_tag_map.get(score_tag_list[0])

        elif name == "postprocess":
            parallel_postprocess_step = Task(
                name=f"parallel-postprocess-step-{step_ind}",
                artifacts={
                    "receptor_pdb": receptor_pdb,
                    "content_json_list": content_json_list,
                    "meta_json_list": meta_json_list,
                },
                template=parallel_post_process_superop(params, image_dict, executor_dict, volumes_dict),
            )
            dag.add(parallel_postprocess_step)
            content_json_list = parallel_postprocess_step.outputs.artifacts["content_json_list"]
            meta_json = parallel_postprocess_step.outputs.artifacts["meta_json"]

        elif name == "pbgbsa":
            parallel_pbgbsa_step = Task(
                name=f"parallel-pbgbsa-step-{step_ind}",
                artifacts={
                    "receptor_pdb": receptor_pdb,
                    "content_json_list": content_json_list,
                    "meta_json_list": meta_json_list,
                },
                template=parallel_pbgbsa_superop(params, image_dict, executor_dict, volumes_dict)
            )
            dag.add(parallel_pbgbsa_step)
            meta_json = parallel_pbgbsa_step.outputs.artifacts["meta_json"]

        if params.get("top_config"):
            top_config = params["top_config"]
            assert top_config.get("top_num") or top_config.get("top_ratio"), "top_num or top_ratio must be set"
            keep_top_step = Task(
                name=f"keep-top-step-{step_ind}",
                artifacts={
                    "meta_json": meta_json,
                },
                parameters={
                    "top_num": top_config.get("top_num"),
                    "top_ratio": top_config.get("top_ratio"),
                    "sort_key": top_config["sort_key"],
                    "reverse": top_config["reverse"],
                },
                template=PythonOPTemplate(
                    keep_best_top_ligands_op,
                    image="python:3.10",
                    image_pull_policy="IfNotPresent",
                    requests={"memory": "1Gi"},
                    limits={"memory": "64Gi"},
                    volumes=volumes_dict.get("volumes"),
                    mounts=volumes_dict.get("mounts"),
                    retry_on_failure_and_error=3,
                ),
                executor=executor_dict["local"]
            )
            meta_json = keep_top_step.outputs.artifacts["meta_json"]
            dag.add(keep_top_step)

        resplit_step = Task(
            name=f"resplit-step-{step_ind}",
            artifacts={
                "meta_json": meta_json,
                "content_json_list": content_json_list, 
            },
            parameters={
                "batch_size": config.get("parallel_batch_size", 18000),
                "only_best": True if name in ["retrieval"] else False,
            },
            template=PythonOPTemplate(
                resplit_op,
                image="python:3.10",
                image_pull_policy="IfNotPresent",
                requests={"memory": "1Gi"},
                limits={"cpu": "1024m", "memory": "64Gi"},
                volumes=volumes_dict.get("volumes"),
                mounts=volumes_dict.get("mounts"),
                retry_on_failure_and_error=3,
            ),
            executor=executor_dict["local"]
        )
        dag.add(resplit_step)
        meta_json_list = resplit_step.outputs.artifacts["meta_json_list"]
        content_json_list = resplit_step.outputs.artifacts["content_json_list"]

    wf = Workflow(name=re.sub(r'_', '-', job_name), dag=dag, parallelism=config.get("wf_max_parallel", 100))
    return wf


def main(config:dict, output_yaml:str=None):
    wf = run_nash_workflow(config)
    if output_yaml:
        with open(output_yaml, "w") as f:
            f.write(wf.to_yaml())
    else:
        wf.submit()


def main_cli():
    parser = argparse.ArgumentParser(description="workflow:unimol-bias-unidock-docking")
    parser.add_argument("-c", "--config_file", type=str, required=True, 
                        help="config file")
    parser.add_argument("-j", "--job_name", type=str, default=None, 
        help="job name")
    parser.add_argument("-o", "--output_yaml", type=str, default=None,
        help="if set, workflow will generate a yaml file instead of submitting")
    args = parser.parse_args()
    with open(args.config_file, "r") as f:
        config = json.load(f)
    if args.job_name:
        config["job_name"] = args.job_name
    main(config, args.output_yaml)

if __name__=="__main__":
    main_cli()
