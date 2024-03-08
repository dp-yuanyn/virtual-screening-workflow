from enum import Enum

from dflow import (
    Executor,
    Workflow, 
    Task,
    DAG,
    InputArtifact,
    InputParameter,
    OutputArtifact,
    randstr,
)
from dflow.python import (
    PythonOPTemplate, 
    Slices,
)

from ops.op_bias import (
    gen_hbond_bpf_op, 
    merge_bpf_op, 
    gen_substructure_bpf_op, 
    get_substructure_ind_op, 
    gen_mcs_bpf_op,
    gen_shape_bpf_op, 
)


class ConstrainedDockingType(Enum):
    HBond = "hbond"
    Substructure = "substructure"
    MCS = "mcs"
    Shape = "shape"
    MetalBond = "metal_bond"


def constrain_superop(constrained_params:dict, image_dict:dict, 
                      executor_dict:dict, volumes_dict:dict) -> DAG:
    local_executor = executor_dict["local"]
    image = image_dict.get("minimize_image", "")

    constrain_superop = DAG(name=f"constrain-superop-{randstr(5)}")
    constrain_superop.inputs.artifacts = {
        "receptor_pdb": InputArtifact(),
        # "meta_json": InputArtifact(),
        "content_json": InputArtifact(),
        "ref_sdf_file": InputArtifact(optional=True),
    }
    receptor_pdb = constrain_superop.inputs.artifacts["receptor_pdb"]
    # meta_json = constrain_superop.inputs.artifacts["meta_json"]
    content_json = constrain_superop.inputs.artifacts["content_json"]
    ref_sdf_file = constrain_superop.inputs.artifacts["ref_sdf_file"]

    meta_json = None
    bias_file = None
    bias_content_json = None

    constrain_type = constrained_params["constrained_type"]
    if constrain_type == ConstrainedDockingType.HBond.value:
        hbond_bias_step = Task(
            name="hbond-bias-step",
            artifacts={
                "receptor_path": receptor_pdb,
            },
            parameters={
                "hbond_sites": constrained_params.get("h_bond_sites", ""),
                "scoring_func": constrained_params.get("scoring", ""),
            },
            template=PythonOPTemplate(
                gen_hbond_bpf_op,
                image=image,
                image_pull_policy="IfNotPresent",
                limits={"memory": "4Gi"},
            ),
            executor=local_executor,
        )
        constrain_superop.add(hbond_bias_step)
        bias_file = hbond_bias_step.outputs.artifacts["bias_file"]

    elif constrain_type == ConstrainedDockingType.Substructure.value:
        substructure_bias_step = Task(
            name="substructure-bias-step",
            artifacts={
                "ref_sdf_file": ref_sdf_file,
            },
            parameters={
                "ind_list": constrained_params.get("indices_list", []),
            },
            template=PythonOPTemplate(
                gen_substructure_bpf_op,
                image=image,
                image_pull_policy="IfNotPresent",
                limits={"memory": "4Gi"},
            ),
            executor=local_executor,
        )
        constrain_superop.add(substructure_bias_step)
        bias_file = substructure_bias_step.outputs.artifacts["bias_file"]

        get_substructure_ind_step = Task(
            name="get-substructure-ind-step",
            artifacts={
                "ref_sdf_file": ref_sdf_file,
                "ligand_content_json": content_json,
            },
            parameters={
                "ind_list": constrained_params.get("indices_list", []),
            },
            template=PythonOPTemplate(
                get_substructure_ind_op,
                image=image,
                image_pull_policy="IfNotPresent",
                limits={"memory": "4Gi"},
                volumes=volumes_dict.get("volumes"),
                mounts=volumes_dict.get("mounts"),
            ),
            executor=local_executor,
        )
        constrain_superop.add(get_substructure_ind_step)
        meta_json = get_substructure_ind_step.outputs.artifacts["meta_json"]

    elif constrain_type == ConstrainedDockingType.MCS.value:
        gen_mcs_bpf_step = Task(
            name="gen-mcs-bpf-step",
            artifacts={
                "ref_sdf_file": ref_sdf_file,
                # "meta_json": meta_json,
                "ligand_content_json": content_json,
            },
            template=PythonOPTemplate(
                gen_mcs_bpf_op,
                image=image,
                image_pull_policy="IfNotPresent",
                requests={"cpu": 2, "memory": "4Gi"},
                limits={"cpu": 4, "memory": "16Gi"},
                volumes=volumes_dict.get("volumes"),
                mounts=volumes_dict.get("mounts"),
            ),
            executor=local_executor,
        )
        constrain_superop.add(gen_mcs_bpf_step)
        bias_content_json = gen_mcs_bpf_step.outputs.artifacts["bias_content_json"]
        meta_json = gen_mcs_bpf_step.outputs.artifacts["meta_json"]

    elif constrain_type == ConstrainedDockingType.Shape.value:
        shape_bias_step = Task(
            name="shape-bias-step",
            artifacts={
                "ref_sdf_file": ref_sdf_file,
            },
            parameters={
                "shape_scale": constrained_params.get("shape_scale", 1),
            },
            template=PythonOPTemplate(
                gen_shape_bpf_op,
                image=image,
                image_pull_policy="IfNotPresent",
                limits={"memory": "4Gi"},
            ),
            executor=local_executor,
        )
        constrain_superop.add(shape_bias_step)
        bias_file = shape_bias_step.outputs.artifacts["bias_file"]

    constrain_superop.outputs.artifacts = {
        "bias_file": OutputArtifact(_from=bias_file, optional=True),
        "bias_content_json": OutputArtifact(_from=bias_content_json, optional=True),
        "meta_json": OutputArtifact(_from=meta_json, optional=True),
    }
    return constrain_superop