import os

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
