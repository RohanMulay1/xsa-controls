"""Launch Pythia-1.4b and Pythia-2.8b reliability runs on RunPod GPU and retrieve results."""

import io
import os
import pathlib
import sys
import tarfile
import time

import paramiko
import runpod

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True, errors='replace')
sys.stderr.reconfigure(encoding='utf-8', line_buffering=True, errors='replace')

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
if not RUNPOD_API_KEY:
    raise RuntimeError("Please set RUNPOD_API_KEY environment variable.")
runpod.api_key = RUNPOD_API_KEY

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
SSH_KEY_PATH = pathlib.Path.home() / ".ssh" / "id_ed25519"
EXTENDED_RESULTS_DIR = REPO / "results" / "extended"


def create_archive_bytes() -> bytes:
    """Pack xsac, scripts, and requirements into a tar.gz in memory."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in ["xsac", "scripts", "requirements.txt"]:
            src = REPO / name
            if src.is_dir():
                tar.add(src, arcname=name, filter=lambda ti: None if "__pycache__" in ti.name or ti.name.endswith(".pyc") else ti)
            elif src.is_file():
                tar.add(src, arcname=name)
    buf.seek(0)
    return buf.read()


def find_gpu_and_create_pod():
    candidate_gpus = [
        ("NVIDIA RTX 6000 Ada Generation", "COMMUNITY"),
        ("NVIDIA RTX 6000 Ada Generation", "SECURE"),
        ("NVIDIA RTX A6000", "COMMUNITY"),
        ("NVIDIA RTX A6000", "SECURE"),
        ("NVIDIA A40", "COMMUNITY"),
        ("NVIDIA GeForce RTX 4090", "COMMUNITY"),
    ]
    pod = None
    for gpu_id, cloud_type in candidate_gpus:
        print(f"Attempting to create pod with {gpu_id} ({cloud_type})...", flush=True)
        try:
            pod = runpod.create_pod(
                name="xsa-pythia-reliability",
                image_name="runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04",
                gpu_type_id=gpu_id,
                cloud_type=cloud_type,
                support_public_ip=True,
                start_ssh=True,
                ports="22/tcp",
                container_disk_in_gb=40,
                volume_in_gb=0,
            )
            if pod and "id" in pod:
                print(f"Successfully created pod {pod['id']} with {gpu_id} ({cloud_type})", flush=True)
                return pod
        except Exception as exc:
            print(f"  Failed: {exc}", flush=True)
    raise RuntimeError("Could not allocate any candidate GPU pod on RunPod.")


def wait_for_pod_ready(pod_id, timeout_sec=300):
    print(f"Waiting for pod {pod_id} to be RUNNING and expose SSH...", flush=True)
    start = time.time()
    while time.time() - start < timeout_sec:
        info = runpod.get_pod(pod_id)
        status = info.get("desiredStatus")
        runtime = info.get("runtime")
        if runtime and runtime.get("ports"):
            ports = runtime["ports"]
            ssh_port_info = next((p for p in ports if p.get("privatePort") == 22), None)
            if ssh_port_info and ssh_port_info.get("ip") and ssh_port_info.get("publicPort"):
                ip = ssh_port_info["ip"]
                port = ssh_port_info["publicPort"]
                print(f"Pod is ready: IP={ip}, SSH Port={port}", flush=True)
                return ip, port
        time.sleep(5)
    raise TimeoutError(f"Pod {pod_id} did not become ready within {timeout_sec}s")


def run_command(ssh, cmd):
    print(f"\n[remote] $ {cmd}", flush=True)
    stdin, stdout, stderr = ssh.exec_command(cmd, get_pty=True)
    for line in iter(stdout.readline, ""):
        safe_line = line.encode("ascii", errors="replace").decode("ascii")
        print(safe_line, end="", flush=True)
    status = stdout.channel.recv_exit_status()
    if status != 0:
        err = stderr.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Command failed with status {status}: {err}")


def main():
    pod = None
    pod_id = None
    try:
        pod = find_gpu_and_create_pod()
        pod_id = pod["id"]
        ip, port = wait_for_pod_ready(pod_id)

        print("Waiting 10s for sshd service to settle...", flush=True)
        time.sleep(10)

        pkey = paramiko.Ed25519Key.from_private_key_file(str(SSH_KEY_PATH))
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Retry SSH connection for up to 60 seconds
        for attempt in range(12):
            try:
                print(f"Connecting to root@{ip}:{port} (attempt {attempt + 1})...", flush=True)
                ssh.connect(ip, port=port, username="root", pkey=pkey, timeout=10)
                break
            except Exception as e:
                print(f"  SSH connect failed: {e}. Retrying in 5s...", flush=True)
                time.sleep(5)
        else:
            raise RuntimeError("Could not establish SSH connection to pod.")

        print("Connected via SSH! Uploading codebase tarball...", flush=True)
        sftp = ssh.open_sftp()
        tar_bytes = create_archive_bytes()
        with sftp.file("/root/xsa-controls.tar.gz", "wb") as f:
            f.write(tar_bytes)

        run_command(ssh, "mkdir -p /root/xsa-controls && tar -xzf /root/xsa-controls.tar.gz -C /root/xsa-controls")
        run_command(ssh, "pip install --progress-bar off --no-cache-dir 'numpy<2.0' scipy pandas tqdm 'transformers==4.44.2' accelerate datasets")

        # Execute Check 0 on Pythia-1.4B and Pythia-2.8B

        run_command(
            ssh,
            "cd /root/xsa-controls && python scripts/run_reliability.py "
            "--models EleutherAI/pythia-1.4b EleutherAI/pythia-2.8b "
            "--n-docs 32 --block 512 --device cuda --dtype bfloat16 "
            "--results-dir results_extended"
        )

        print("\nRetrieving extended results...")
        EXTENDED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        for fname in ["reliability.csv", "a2_correlations.csv", "reliability_budget_sweep.csv", "a2_per_head.csv", "reliability.json"]:
            remote_path = f"/root/xsa-controls/results_extended/{fname}"
            local_path = EXTENDED_RESULTS_DIR / fname
            try:
                sftp.get(remote_path, str(local_path))
                print(f"  Downloaded: {local_path} ({local_path.stat().st_size} bytes)")
            except Exception as e:
                print(f"  Could not download {fname}: {e}")

        sftp.close()
        ssh.close()
        print("\nSUCCESS: All inferences complete and retrieved!")

    finally:
        if pod_id:
            print(f"\nTERMINATING POD {pod_id} TO PREVENT ANY CREDIT LEAK...")
            try:
                res = runpod.terminate_pod(pod_id)
                print(f"Pod termination response: {res}")
            except Exception as e:
                print(f"ERROR terminating pod {pod_id}: {e}")


if __name__ == "__main__":
    main()
