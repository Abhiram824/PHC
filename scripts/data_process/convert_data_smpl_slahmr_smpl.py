import torch
import joblib
import numpy as np
from scipy.spatial.transform import Rotation as sRot
import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "poselib"))

from smpl_sim.smpllib.smpl_mujoco_new import SMPL_BONE_ORDER_NAMES as joint_names
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState

robot_cfg = {
    "mesh": False,
    "model": "smpl",
    "upright_start": False,  # matches smpl_humanoid.yaml has_upright_start: True
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
SLAHMR_NPZ = (
    "/home/abhiram03/motion_imitation/iiith_cooking_109_4/logs/video-val/"
    "2026-04-07/points_triangulated-all-shot-0-0-1117/smooth_fit/"
    "points_triangulated_000060_world_results.npz"
)
OUTPUT_PATH = (
    "/home/abhiram03/motion_imitation/PHC/data/iiith_cooking_109_4_slahmr_smpl.pkl"
)

data = np.load(SLAHMR_NPZ, allow_pickle=True)
root_orient = data["root_orient"][0]   # [T, 3]
pose_body   = data["pose_body"][0]     # [T, 63]  (21 body joints * 3)
trans       = data["trans"][0]         # [T, 3]
betas_raw   = data["betas"][0]         # [16]

T = root_orient.shape[0]

# Build SMPL pose_aa: root(3) + body(63) + dummy_hands(6) = 72
# SMPL has 24 joints: 22 body + L_Hand + R_Hand (last two are wrist terminal joints)
pose_aa = np.concatenate(
    [root_orient, pose_body, np.zeros((T, 6))], axis=-1
)  # [T, 72]

beta = betas_raw[:10].astype(np.float32)

gender        = "neutral"
gender_number = [0]
fps           = 30.0
key_name      = "iiith_cooking_109_4"

# -------------------------------------------

# SMPL mujoco joint order (24 joints, as they appear in smpl_humanoid.xml)
mujoco_joint_names = [
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Toe", "Torso",
    "Spine", "Chest", "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand",
]

smpl_2_mujoco = [joint_names.index(q) for q in mujoco_joint_names if q in joint_names]

batch_size = pose_aa.shape[0]
pose_aa_mj = pose_aa.reshape(-1, 24, 3)[..., smpl_2_mujoco, :].copy()

pose_quat = (
    sRot.from_rotvec(pose_aa_mj.reshape(-1, 3))
    .as_quat()
    .reshape(batch_size, 24, 4)
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

# SLAHMR is already in Z-up world — no [0.5,0.5,0.5,0.5].inv() correction needed.
# (That correction is for AMASS Y-up convention and would make the person lay down.)
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
