from typing import List
from pathlib import Path
import os
import shutil
import subprocess

from dflow.python import (
    OP, 
    OPIO, 
    Artifact,
    upload_packages,
)

if "__file__" in locals():
    upload_packages.append(__file__)


@OP.function
def convert_receptor_op(
        input_receptor:Artifact(Path),
) -> {"output_receptor": Artifact(Path)}:

    protein_basename = os.path.basename(input_receptor.as_posix())
    output_receptor = os.path.splitext(protein_basename)[0] + '.pdbqt'

    mgltools_python_path = shutil.which("pythonsh")
    if not mgltools_python_path:
        raise KeyError("mgltools env not found, please install first")
    mgltools_python_path = str(mgltools_python_path)
    prepare_receptor_script_path = os.path.join(os.path.dirname(os.path.dirname(mgltools_python_path)), 
            "MGLToolsPckgs", "AutoDockTools", "Utilities24", "prepare_receptor4.py")
    cmd = f"{mgltools_python_path} {prepare_receptor_script_path} \
        -r {input_receptor.as_posix()} -o {output_receptor} -U nphs_lps_nonstdres"
    resp = subprocess.run(cmd, shell=True, 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            encoding="utf-8")
    print(resp.stdout)
    if resp.stderr:
        print(resp.stderr)
    return OPIO({"output_receptor":Path(output_receptor)})


@OP.function
def preprocess_ligands_op(
        ligands_json: Artifact(Path),
) -> {"content_json": Artifact(Path)}:
    import os
    import glob
    import uuid
    from multiprocessing import Pool
    from functools import partial
    from tools.logger import get_logger
    from tools.stdio import StdIO
    from tools.unidock_topo_tree import calc_torsion_tree_info

    logger = get_logger()

    ligands_dir = Path(f"/tmp/ligprepocess_dir_{uuid.uuid4().hex}")
    ligands_dir.mkdir(parents=True, exist_ok=True)
    content_io = StdIO(json_path=ligands_json)
    logger.info("Start write input ligands time")
    input_ligands = content_io.write_raw_content_to_file(ligands_dir)
    logger.info("End write input ligands")
    results_dir = Path(f"results_dir")
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Start calc torsion tree")
    with Pool(os.cpu_count()) as pool:
        pool.map(partial(calc_torsion_tree_info, results_dir=results_dir), input_ligands)
    logger.info("End calc torsion tree")
    result_ligands = glob.glob(os.path.join(results_dir, "*.sdf"))
    results_io = StdIO()
    results_io.save_content_by_input_files(result_ligands)
    result_path = Path(f"ligands_{uuid.uuid4().hex}.json")
    results_io.write_content_json(result_path)

    logger.info(os.listdir("/tmp"))
    return OPIO({"content_json": result_path})