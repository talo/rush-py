import os
import tarfile

import rush


async def setup_workspace(workspace_dir, clean_workspace):
    if not workspace_dir.exists():
        os.makedirs(workspace_dir)
    if clean_workspace:
        client = rush.Provider(workspace=workspace_dir)
        await client.nuke()


async def check_status_and_report_failures(client):
    """
    This will show the status of all of your runs
    """
    status = await client.status(group_by="path")
    print(f"{'Module':<20} | {'Status':<20} | Count")
    print("-" * 50)
    for instance_id, (status, path, count) in status.items():
        print(f"{path:<20} | {status:<20} | {count}")
        if status.value == "FAILED":
            try:
                async for log_page in client.logs(instance_id, "stderr"):
                    for log in log_page:
                        print(log)
            except Exception:
                pass


def get_resources(machine_name, gpus):
    machine_data = {}
    machine_data["target"] = machine_name
    if machine_name == "GADI":
        if gpus != 4:
            print("WARNING: Gadi runs should use 4 GPUs; setting gpus to 4")
        # Gadi needs CPUs and walltime
        machine_data["resources"] = {
            "gpus": 4,
            "storage": 10,
            "storage_units": "GB",
            "cpus": 48,
            "walltime": 60,
        }
    elif machine_name.startswith("NIX_SSH"):
        # All the Nix machines care about
        machine_data["resources"] = {
            "gpus": gpus,
            "storage": 10,
            "storage_units": "GB",
        }


def extract_gmx_dry_frames(client, dry_frames_path, rcsb_id):
    # Extract the "dry" (i.e. non-solvated) pdb frames we asked for
    with tarfile.open(dry_frames_path, "r") as tf:
        selected_frame_pdbs = [
            (member.name, tf.extractfile(member).read())
            for member in tf
            if member.name.startswith("outputs_md.dry.pdb/md") and member.name.endswith("pdb")
        ]
        for name, frame in selected_frame_pdbs:
            print(type(name))
            index = name.split("/")[1].split(".")[2][3]
            with open(client.workspace / f"02_{rcsb_id}_gmx_frame{index}.pdb", "w") as pf:
                pf.write(frame.decode("utf-8"))
