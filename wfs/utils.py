from typing import Union, Optional
import copy
from argo.workflows.client import (
    V1Volume, 
    V1VolumeMount, 
    V1PersistentVolumeClaimVolumeSource
)
import dflow
from dflow.plugins import bohrium
from dflow.executor import ContainerExecutor
from dflow.plugins.dispatcher import DispatcherExecutor


def setup(config:dict):
    if "bohrium_config" in config.keys():
        bohrium_config = config.pop("bohrium_config")
        if "username" in bohrium_config:
            bohrium.config["username"] = bohrium_config.pop("username")
        if "password" in bohrium_config:
            bohrium.config["password"] = bohrium_config.pop("password")
        if "ticket" in bohrium_config:
            bohrium.config["ticket"] = bohrium_config.pop("ticket")
        for k, v in bohrium_config.items():
            bohrium.config[k] = v

    if "dflow_config" in config.keys():
        dflow_config = config.pop("dflow_config")
        for k, v in dflow_config.items():
            dflow.config[k] = v

    if "dflow_s3_config" in config.keys():
        dflow_s3_config = config.pop("dflow_s3_config")
        for k, v in dflow_s3_config.items():
            dflow.s3_config[k] = v
    if dflow.s3_config["repo_key"] == "oss-bohrium":
        from dflow.plugins.bohrium import TiefblueClient
        dflow.s3_config["storage_client"] = TiefblueClient()


def get_local_executor(local_mode:bool=False, docker_executable:str="docker") -> Union[None, ContainerExecutor]:
    local_executor = None
    if local_mode:
        local_executor = ContainerExecutor(docker=docker_executable)
    return local_executor


def get_dispatcher_executor(dispatcher_config:dict) -> DispatcherExecutor:
    image = dispatcher_config.get("image")
    clean = dispatcher_config.get("clean", True)
    machine_dict = dispatcher_config.get("machine_dict", dict())
    resources_dict = dispatcher_config.get("resources_dict")
    docker_executable = dispatcher_config.get("docker_executable")
    singularity_executable = dispatcher_config.get("singularity_executable")
    container_args = dispatcher_config.get("container_args", "")
    remote_executor = DispatcherExecutor(
        image=image,
        clean=clean,
        machine_dict=machine_dict,
        resources_dict=resources_dict,
        docker_executable=docker_executable,
        singularity_executable=singularity_executable,
        container_args=container_args,
    )
    return remote_executor


def get_remote_executor(executor_config:dict, scass_type:str) -> DispatcherExecutor:
    executor_config = copy.deepcopy(executor_config)
    try:
        executor_config["machine_dict"]["remote_profile"]["input_data"]["scass_type"] = scass_type
    except:
        pass
    remote_executor = get_dispatcher_executor(executor_config)
    return remote_executor


def generate_executor_dict(remote_config:dict, scass_type_dict:dict, use_lbg:bool=False) -> dict:
    executor_dict = {
        "gpu_unidock": get_remote_executor(executor_config=remote_config, 
                                        scass_type=scass_type_dict["gpu_unidock"]) if scass_type_dict.get("gpu_unidock") else None,
        "gpu": get_remote_executor(executor_config=remote_config, 
                                scass_type=scass_type_dict["gpu"]) if scass_type_dict.get("gpu") else None,
        "cpu": get_remote_executor(executor_config=remote_config,
                                scass_type=scass_type_dict["cpu"]) if scass_type_dict.get("cpu") else None,
        "local": get_local_executor()
    }
    return executor_dict


def generate_volumes_dict(volume_config:dict) -> dict:
    volumes, mounts = None, None
    if volume_config:
        volumes=[V1Volume(
            name=volume_config["volume_name"],
            persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(
                claim_name=volume_config["claim_name"]))]
        mounts=[V1VolumeMount(
            name=volume_config["volume_name"],
            mount_path=volume_config["mount_path"],
            sub_path="{{pod.name}}-{{workflow.duration}}")]
    return {"volumes": volumes, "mounts": mounts}