#!/usr/bin/env python3
"""
Run all PHC imitation models for N episodes each, saving logs and video.

Usage:
    python scripts/eval_all_models.py /path/to/output_dir [--episodes 3] [--no-video] [--timeout 180]
"""

import argparse
import os
import subprocess
import time
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SMPLX_PKL = "data/iiith_cooking_109_4_slahmr.pkl"
SMPL_PKL  = "data/iiith_cooking_109_4_slahmr_smpl.pkl"

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
MODELS = [
    dict(
        name="phc_x_pnn",
        pkl=SMPLX_PKL,
        args=[
            "learning=im_pnn_big", "exp_name=phc_x_pnn",
            "env=env_im_x_pnn", "robot=smplx_humanoid",
            "env.training_prim=0", "epoch=-1", "test=True",
            "env.num_envs=1",
        ],
    ),
    dict(
        name="phc_comp_kp_2",
        pkl=SMPL_PKL,
        args=[
            "learning=im_mcp_big", "learning.params.network.ending_act=False",
            "exp_name=phc_comp_kp_2", "env.obs_v=7",
            "env=env_im_getup_mcp", "robot=smpl_humanoid",
            "robot.real_weight_porpotion_boxes=False",
            "env.models=['output/HumanoidIm/phc_kp_2/Humanoid.pth']",
            "env.num_prim=3", "epoch=-1", "test=True", "env.num_envs=1",
        ],
    ),
    dict(
        name="phc_comp_3",
        pkl=SMPL_PKL,
        args=[
            "learning=im_mcp_big", "exp_name=phc_comp_3",
            "env=env_im_getup_mcp", "robot=smpl_humanoid",
            "env.zero_out_far=False", "robot.real_weight_porpotion_boxes=False",
            "env.models=['output/HumanoidIm/phc_3/Humanoid.pth']",
            "env.num_prim=3", "epoch=-1", "test=True", "env.num_envs=1",
        ],
    ),
    dict(
        name="phc_shape_pnn_iccv",
        pkl=SMPL_PKL,
        args=[
            "learning=im_pnn", "exp_name=phc_shape_pnn_iccv",
            "epoch=-1", "test=True", "env=env_im_pnn",
            "robot=smpl_humanoid_shape", "robot.freeze_hand=True",
            "robot.box_body=False", "env.num_prim=4", "env.num_envs=1",
        ],
    ),
    dict(
        name="phc_kp_pnn_iccv",
        pkl=SMPL_PKL,
        args=[
            "learning=im_pnn", "exp_name=phc_kp_pnn_iccv",
            "epoch=-1", "test=True", "env=env_im_pnn",
            "robot.freeze_hand=True", "robot.box_body=False",
            "env.obs_v=7", "env.num_prim=4", "env.num_envs=1",
        ],
    ),
    dict(
        name="phc_kp_mcp_iccv",
        pkl=SMPL_PKL,
        args=[
            "learning=im_mcp", "exp_name=phc_kp_mcp_iccv",
            "env=env_im_getup_mcp", "robot=smpl_humanoid",
            "robot.freeze_hand=True", "robot.box_body=False",
            "env.z_activation=relu",
            "env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth']",
            "env.obs_v=7", "epoch=-1", "test=True", "env.num_envs=1",
        ],
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def free_gpu():
    """Kill any lingering Isaac Gym processes before starting a new run."""
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True, text=True
    )
    for pid in result.stdout.strip().splitlines():
        pid = pid.strip()
        if pid:
            try:
                subprocess.run(["kill", pid], check=False)
                print(f"  Freed GPU: killed PID {pid}")
            except Exception:
                pass
    time.sleep(2)


def start_ffmpeg(video_path: Path, display: str):
    """Start ffmpeg screen capture. Returns None if ffmpeg unavailable."""
    if not display:
        return None
    cmd = [
        "ffmpeg", "-y",
        "-f", "x11grab", "-r", "30", "-s", "1920x1080",
        "-i", display,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        str(video_path),
    ]
    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(1)  # let ffmpeg initialise
        return proc
    except FileNotFoundError:
        print("  ffmpeg not found — skipping video capture")
        return None


def stop_ffmpeg(proc):
    if proc is None:
        return
    try:
        proc.communicate(input=b"q", timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def run_model(model: dict, output_dir: Path, n_episodes: int,
              timeout: int, record_video: bool,
              no_early_term: bool = False) -> dict:
    name = model["name"]
    log_path   = output_dir / f"{name}.log"
    video_path = output_dir / f"{name}.mp4"

    display = os.environ.get("DISPLAY", "") if record_video else ""
    headless = "False" if record_video and display else "True"

    cmd = (
        ["python", "phc/run_hydra.py"]
        + model["args"]
        + [
            f"env.motion_file={model['pkl']}",
            f"learning.params.config.player.games_num={n_episodes}",
            f"headless={headless}",
        ]
    )
    if no_early_term:
        cmd.append("env.enableEarlyTermination=False")
    if record_video and display:
        cmd.append("no_virtual_display=True")

    print(f"\n{'='*60}")
    print(f"  Model : {name}")
    print(f"  Log   : {log_path}")
    if record_video and display:
        print(f"  Video : {video_path}")
    print(f"  CMD   : {' '.join(cmd)}")
    print(f"{'='*60}")

    free_gpu()

    ffmpeg = start_ffmpeg(video_path, display) if (record_video and display) else None

    # Ensure the conda env's lib dir is in LD_LIBRARY_PATH so isaacgym
    # can find libpython3.8.so.1.0 in the subprocess.
    env = os.environ.copy()
    conda_lib = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "lib")
    existing_ldpath = env.get("LD_LIBRARY_PATH", "")
    if conda_lib not in existing_ldpath:
        env["LD_LIBRARY_PATH"] = f"{conda_lib}:{existing_ldpath}".rstrip(":")

    returncode = None
    with open(log_path, "w") as lf:
        lf.write(f"CMD: {' '.join(cmd)}\n\n")
        lf.flush()
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        try:
            proc.wait(timeout=timeout)
            returncode = proc.returncode
            print(f"  Done (exit {returncode})")
        except subprocess.TimeoutExpired:
            print(f"  Timeout after {timeout}s — killing")
            proc.kill()
            proc.wait()
            returncode = -1

    stop_ffmpeg(ffmpeg)

    # Parse last reward/steps from log
    reward, steps = None, None
    try:
        lines = log_path.read_text().splitlines()
        for line in reversed(lines):
            if line.startswith("reward:") and "steps:" in line:
                parts = line.split()
                reward = float(parts[1])
                steps  = float(parts[3])
                break
    except Exception:
        pass

    return {"name": name, "returncode": returncode,
            "last_reward": reward, "last_steps": steps}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", help="Directory to save logs and videos")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes per model (default: 3)")
    parser.add_argument("--timeout", type=int, default=180,
                        help="Hard kill timeout per model in seconds (default: 180)")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip video recording (headless=True, faster)")
    parser.add_argument("--models", nargs="+",
                        help="Run only specific models by name (default: all)")
    parser.add_argument("--no-early-term", action="store_true",
                        help="Disable early termination (run full episode_length=300 steps)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models_to_run = MODELS
    if args.models:
        models_to_run = [m for m in MODELS if m["name"] in args.models]
        if not models_to_run:
            print(f"No models matched. Available: {[m['name'] for m in MODELS]}")
            sys.exit(1)

    record_video = not args.no_video
    if record_video and not os.environ.get("DISPLAY"):
        print("Warning: DISPLAY not set — video recording disabled")
        record_video = False

    print(f"Output dir  : {output_dir}")
    print(f"Episodes    : {args.episodes}")
    print(f"Timeout     : {args.timeout}s")
    print(f"Record video: {record_video}")
    print(f"No early term: {args.no_early_term}")
    print(f"Models      : {[m['name'] for m in models_to_run]}")

    results = []
    for model in models_to_run:
        result = run_model(
            model, output_dir,
            n_episodes=args.episodes,
            timeout=args.timeout,
            record_video=record_video,
            no_early_term=args.no_early_term,
        )
        results.append(result)

    # Summary
    summary_path = output_dir / "summary.txt"
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    lines = []
    header = f"{'Model':<25} {'Exit':>5} {'Last Reward':>12} {'Last Steps':>11}"
    print(header)
    lines.append(header)
    sep = "-" * len(header)
    print(sep)
    lines.append(sep)
    for r in results:
        reward_s = f"{r['last_reward']:.3f}" if r["last_reward"] is not None else "N/A"
        steps_s  = f"{r['last_steps']:.1f}"  if r["last_steps"]  is not None else "N/A"
        row = f"{r['name']:<25} {r['returncode']:>5} {reward_s:>12} {steps_s:>11}"
        print(row)
        lines.append(row)

    summary_path.write_text("\n".join(lines) + "\n")
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
