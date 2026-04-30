import torch
import joblib
import numpy as np
from scipy.spatial.transform import Rotation as sRot
import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "poselib"))

from smpl_sim.smpllib.smpl_parser import SMPLH_BONE_ORDER_NAMES as joint_names
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState

robot_cfg = {
    "mesh": False,
    "model": "smplx",
    "upright_start": False,  # must match smplx_humanoid.yaml has_upright_start: False
    "body_params": {},
    "joint_params": {},
    "geom_params": {},
    "actuator_params": {},
}
print(robot_cfg)

smpl_local_robot = LocalRobot(
    robot_cfg,
    data_dir="data/smpl",
)

# ----------- slahmr data loading -----------
# smooth_fit npz has pose_body already decoded from VPoser latents
SLAHMR_NPZ = (
    "/home/abhiram03/motion_imitation/iiith_cooking_109_4/logs/video-val/"
    "2026-04-07/points_triangulated-all-shot-0-0-1117/smooth_fit/"
    "points_triangulated_000060_world_results.npz"
)
OUTPUT_PATH = (
    "/home/abhiram03/motion_imitation/PHC/data/iiith_cooking_109_4_slahmr.pkl"
)

data = np.load(SLAHMR_NPZ, allow_pickle=True)
# shapes: [1, T, *]
root_orient = data["root_orient"][0]   # [T, 3]
pose_body   = data["pose_body"][0]     # [T, 63]  (21 body joints * 3)
trans       = data["trans"][0]         # [T, 3]
betas_raw   = data["betas"][0]         # [16]

T = root_orient.shape[0]

# Build full SMPLH/SMPLX pose_aa:
#   root(3) + body(63) + left_hand_zeros(45) + right_hand_zeros(45) = 156
pose_aa = np.concatenate(
    [root_orient, pose_body, np.zeros((T, 45)), np.zeros((T, 45))], axis=-1
)  # [T, 156]

# PHC uses 10 betas for smplx
beta = betas_raw[:10].astype(np.float32)

gender        = "neutral"
gender_number = [0]
fps           = 30.0
key_name      = "iiith_cooking_109_4"

# -------------------------------------------

# SMPLH has 52 joints; mujoco_joint_names lists them in the order they appear
# in the generated smplx humanoid XML (same as SMPLH_BONE_ORDER_NAMES)
mujoco_joint_names = [
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Toe", "Torso",
    "Spine", "Chest", "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist",
    "L_Index1", "L_Index2", "L_Index3",
    "L_Middle1", "L_Middle2", "L_Middle3",
    "L_Pinky1", "L_Pinky2", "L_Pinky3",
    "L_Ring1", "L_Ring2", "L_Ring3",
    "L_Thumb1", "L_Thumb2", "L_Thumb3",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist",
    "R_Index1", "R_Index2", "R_Index3",
    "R_Middle1", "R_Middle2", "R_Middle3",
    "R_Pinky1", "R_Pinky2", "R_Pinky3",
    "R_Ring1", "R_Ring2", "R_Ring3",
    "R_Thumb1", "R_Thumb2", "R_Thumb3",
]

smpl_2_mujoco = [joint_names.index(q) for q in mujoco_joint_names if q in joint_names]

batch_size = pose_aa.shape[0]
# Reorder from SMPLH canonical order to mujoco order
pose_aa_mj = pose_aa.reshape(-1, 52, 3)[..., smpl_2_mujoco, :].copy()

pose_quat = (
    sRot.from_rotvec(pose_aa_mj.reshape(-1, 3))
    .as_quat()
    .reshape(batch_size, 52, 4)
)

beta[:] = 0
gender_number = [0]
gender = "neutral"
print("using neutral model")

smpl_local_robot.load_from_skeleton(
    betas=torch.from_numpy(beta[None,]), gender=gender_number, objs_info=None
)
smpl_local_robot.write_xml("phc/data/assets/mjcf/smpl_humanoid_1.xml")
skeleton_tree = SkeletonTree.from_mjcf("phc/data/assets/mjcf/smpl_humanoid_1.xml")

root_trans_offset = torch.from_numpy(trans) + skeleton_tree.local_translation[0]

new_sk_state = SkeletonState.from_rotation_and_root_translation(
    skeleton_tree,
    torch.from_numpy(pose_quat),
    root_trans_offset,
    is_local=True,
)

# SLAHMR already outputs in Z-up world (root_orient ~90° X rotation for standing).
# The AMASS [0.5,0.5,0.5,0.5].inv() correction is NOT applied here — it would
# over-rotate the already-correct orientation and make the body lay down.
pose_quat_global = new_sk_state.global_rotation.reshape(-1, 4).numpy().reshape(batch_size, -1, 4)

new_sk_state = SkeletonState.from_rotation_and_root_translation(
    skeleton_tree,
    torch.from_numpy(pose_quat_global),
    root_trans_offset,
    is_local=False,
)
pose_quat = new_sk_state.local_rotation.numpy()

new_motion_out = {
    "pose_quat_global":   pose_quat_global,
    "pose_quat":          pose_quat,
    "trans_orig":         trans,
    "root_trans_offset":  root_trans_offset,
    "beta":               beta,
    "gender":             gender,
    "pose_aa":            pose_aa,
    "fps":                fps,
}

full_motion_dict = {key_name: new_motion_out}
joblib.dump(full_motion_dict, OUTPUT_PATH)
print(f"Saved to {OUTPUT_PATH}")
print(f"  frames: {batch_size}, fps: {fps}")
print(f"  pose_aa shape: {pose_aa.shape}")
print(f"  pose_quat_global: {pose_quat_global.shape}")
print(f"  pose_quat:        {pose_quat.shape}")
