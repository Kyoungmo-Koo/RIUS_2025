
"""simulation program for the raus system integration testing 
"""

# from vortex_get_vol_data import OCTEngine 

import pyrealsense2 as rs
import numpy as np
import cv2 
import igmr_robotics_toolkit
import open3d
import os
from random import seed, uniform
from threading import Thread
from random import seed, uniform
import scipy 
from sklearn.neighbors import NearestNeighbors
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib

import yaml

# igmr-robotics-toolkits related work
from typing import List, Optional
import igmr_robotics_toolkit.util.default_logging
from igmr_robotics_toolkit.viewer.core import create_simple_viewer
from igmr_robotics_toolkit.collision import Collider
# from igmr_robotics_toolkit.robot.loader import load_robot
from igmr_robotics_toolkit.viewer.widget import PathWidget, PointCloudWidget, TransformWidget, RobotWidget, TransformListWidget, LineWidget
from igmr_robotics_toolkit.control.simulator import Simulator
# from igmr_robotics_toolkit.robot.loader import load_robot
from igmr_robotics_toolkit.motion.trajectory import TrajectoryGenerator
from igmr_robotics_toolkit.viewer.motion import show_trajectory
from igmr_robotics_toolkit.robot.loader import load_robot
from igmr_robotics_toolkit.motion.path import ik_path, check_joint_path
from igmr_robotics_toolkit.math import hinv
from scipy.spatial.transform import Rotation, Slerp
from igmr_robotics_toolkit.viewer.core import _T
from panda3d.core import NodePath
from igmr_robotics_toolkit.viewer.core import create_simple_viewer
from igmr_robotics_toolkit.viewer.widget import ControlledRobotWidget, PathWidget, TransformWidget
from igmr_robotics_toolkit.control.simple import PointToPoint
from igmr_robotics_toolkit.math import average_pose
from igmr_robotics_toolkit.control.action import ActionProgram
from igmr_robotics_toolkit.control.action.position_trajectory import BufferedJointTrajectoryAction, BufferedCartesianTrajectoryAction

# third-party
import socket

# klampt math module 
from klampt.math import so3 

# ultrasound module
# us system 
from ultrasound_ge_class import Ultrasound, FORMAT_CFM, FORMAT_NAMES

import shutil
import time 

LINEAR = 1
ARC = 2
INTEGRATED = 3

# ========== Settings ==========
SERIAL_NUMBER = "234422060685"
RESOLUTION = (1280, 720)
SAVE_DIR = "20251021_3"

# RealSense / point cloud controls
SUBSAMPLE = 4
Z_CLIP = 1.5
VIS = False

# 2D projection image controls
GRID_SIZE = 640
XY_RANGE = 0.30
FLIP_Y = True
GAUSSIAN_BLUR = 3
SAVE_OVERLAY = True

# MediaPipe controls
MAX_NUM_HANDS = 2
STATIC_IMAGE_MODE = True
MODEL_COMPLEXITY = 1
MIN_DET_CONF = 0.3
MIN_TRACK_CONF = 0.3

seed(1337)
np.random.seed(0)

# UR3 2017332424 RUSS
delta_theta = [ 4.61804064568337139e-05, -0.0596917792801850075, 0.187982530479293558, -0.128421417771249463, 1.86920644791972534e-05, -3.70876954857309694e-05]
delta_a = [ 3.74645469127644245e-05, 0.000432751255883601083, 0.0015154637341757704, 3.54739391322167398e-05, 5.12409078229780638e-05, 0]
delta_d = [ 0.000223224169787122895, -6.29131988413453058, 8.56941731705468257, -2.27804644471935935, -7.31385696645381334e-05, 0.000410720409206782877]
delta_alpha = [ 0.000118200678426161332, 0.0023051124802664618, 0.0120016019438704737, 0.000466335407550255709, -0.000233982646400399119, 0]

from PyUniversalRobot import kinematics
table = kinematics.UR3.table

for (dh, dt, dr, dd, da) in zip(table, delta_theta, delta_a, delta_d, delta_alpha):
    dh.t += dt
    dh.r += dr
    dh.d += dd
    dh.a += da

ur3 = kinematics.UR3
generic_ik = kinematics.Kinematics('UR3 generic', table, ur3)
generic_ik.set_max_deviation(10/180*np.pi)
ptol = 1e-5
atol = 1e-4
etol = 1e-6
generic_ik.solver.set_error_tolerances(etol, np.sqrt(etol / ptol**2), 180 / np.pi * np.sqrt(etol / atol**2))
generic_ik.solver.set_time_limit(1)
generic_ik.solver.set_random_seed(0)

def robust_ik_path(q_start: np.ndarray, ee_path: List[np.ndarray], kin) -> List[np.ndarray]:
    # check that initial joint config matches starting pose
    if not np.allclose(ee_path[0], kin.forward(q_start), atol=1e-5):
        raise RuntimeError(f'initial joint configuration does not match starting pose: {q_start}')

    q_path = [q_start]
    perturbations = [] 

    for (i, pose) in enumerate(ee_path[1:]):
        a = 0
        e = None
        while True:
            if a > 0.1:
                raise RuntimeError(f"Failed to find nearby IK solution at waypoint {i+1} after high perturbation (a = {a:.3f})")
            q_ref = q_path[-1] + [uniform(-a, a) for _ in range(len(q_path[-1]))]
            try:
                q_new = kin.inverse_nearest(pose, q_ref)
                diff = np.mean(np.abs(q_new - q_path[-1]) ** 2)
                if diff < 0.5:
                    q_path.append(q_new)
                    break  # Accept and exit loop
            except kinematics.InverseException as exc:
                e = exc  # Store error in case we eventually fail
            a += 0.001  # Increment perturbation regardless of success/failure

        perturbations.append(a)
        if e and len(q_path) == i + 1:  # If no valid solution was appended
            raise e
    print(f"Max perturbation used: {max(perturbations)}")
    return q_path

def rs_to_xyzrgb(color_frame, depth_frame):
    """Convert aligned RealSense frames to (x,y,z,r,g,b)."""
    pc = rs.pointcloud()
    pc.map_to(color_frame)
    points = pc.calculate(depth_frame)
    vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
    tex = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
    color_image = np.asanyarray(color_frame.get_data())
    H, W, _ = color_image.shape
    u = (tex[:, 0] * W).astype(np.int32)
    v = (tex[:, 1] * H).astype(np.int32)
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H) & np.isfinite(vtx[:, 2]) & (vtx[:, 2] > 0)
    if Z_CLIP is not None:
        valid &= (vtx[:, 2] <= Z_CLIP)
    if not np.any(valid):
        return np.empty((0, 6), dtype=np.float32)
    vtx = vtx[valid]
    u, v = u[valid], v[valid]
    rgb = color_image[v, u, :][:, ::-1].astype(np.float32) / 255.0
    xyzrgb = np.column_stack((vtx, rgb)).astype(np.float32)
    return xyzrgb


def pcs_to_xy_image(pcs, grid_size=640, xy_range=0.3, flip_y=True, subsample=None, gaussian_blur=0):
    """Project list of xyzrgb point clouds to a single 2D XY image."""
    if len(pcs) == 0:
        return None, None, None

    all_xyz = np.concatenate([pc[:, :3] for pc in pcs], axis=0)
    all_rgb = np.concatenate([pc[:, 3:] for pc in pcs], axis=0)
    if subsample and subsample > 1:
        all_xyz = all_xyz[::subsample]
        all_rgb = all_rgb[::subsample]

    x, y = all_xyz[:, 0], all_xyz[:, 1]
    m = (x >= -xy_range) & (x <= xy_range) & (y >= -xy_range) & (y <= xy_range)
    if not np.any(m):
        blank = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
        return blank, cv2.cvtColor(blank, cv2.COLOR_BGR2RGB), all_xyz

    x, y, rgb = x[m], y[m], all_rgb[m]
    xi = np.clip(((x + xy_range) / (2 * xy_range) * grid_size).astype(np.int32), 0, grid_size - 1)
    yi = np.clip(((y + xy_range) / (2 * xy_range) * grid_size).astype(np.int32), 0, grid_size - 1)
    if flip_y:
        yi = (grid_size - 1) - yi

    lin = yi * grid_size + xi
    count = np.bincount(lin, minlength=grid_size * grid_size).astype(np.float32)
    sum_r = np.bincount(lin, weights=rgb[:, 0], minlength=grid_size * grid_size).astype(np.float32)
    sum_g = np.bincount(lin, weights=rgb[:, 1], minlength=grid_size * grid_size).astype(np.float32)
    sum_b = np.bincount(lin, weights=rgb[:, 2], minlength=grid_size * grid_size).astype(np.float32)
    count[count == 0] = 1.0
    r = (sum_r / count).reshape(grid_size, grid_size)
    g = (sum_g / count).reshape(grid_size, grid_size)
    b = (sum_b / count).reshape(grid_size, grid_size)

    img_rgb = np.stack([r, g, b], axis=-1)
    img_rgb = np.clip(img_rgb * 255.0, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    if gaussian_blur and gaussian_blur % 2 == 1:
        img_bgr = cv2.GaussianBlur(img_bgr, (gaussian_blur, gaussian_blur), 0)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_bgr, img_rgb, all_xyz


def find_joint_xyz(all_xyz, xy_norm, grid_size, xy_range, flip_y=True, radius_mm=1.0):
    """Estimate (x,y,z) by averaging nearby 3D points within radius_mm."""
    x_norm, y_norm = xy_norm
    x = (x_norm - 0.5) * 2 * xy_range
    y = (0.5 - y_norm if flip_y else y_norm - 0.5) * 2 * xy_range
    radius = radius_mm / 1000.0
    dist2 = (all_xyz[:, 0] - x) ** 2 + (all_xyz[:, 1] - y) ** 2
    near = dist2 <= radius ** 2
    if np.any(near):
        pts = all_xyz[near]
        return pts[:, :3].mean(axis=0)
    else:
        return None


def visualize_and_detect(pcs, save_dir, grid_size=GRID_SIZE, xy_range=XY_RANGE,
                         flip_y=FLIP_Y, subsample=SUBSAMPLE, gaussian_blur=GAUSSIAN_BLUR):
    """Combine PCs → XY image, run Mediapipe Hands, draw MCP/PIP joints, print 3D positions."""
    img_bgr, img_rgb, all_xyz = pcs_to_xy_image(
        pcs, grid_size=grid_size, xy_range=xy_range, flip_y=flip_y,
        subsample=subsample, gaussian_blur=gaussian_blur
    )
    if img_bgr is None:
        print("No points to project; skipping visualization/detection.")
        return

    import mediapipe as mp
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=STATIC_IMAGE_MODE,
        max_num_hands=MAX_NUM_HANDS,
        model_complexity=MODEL_COMPLEXITY,
        min_detection_confidence=MIN_DET_CONF,
        min_tracking_confidence=MIN_TRACK_CONF
    )
    results = hands.process(img_rgb)
    hands.close()

    joint_xyz_dict = {}
    if results.multi_hand_landmarks:
        for hand_id, hand_landmarks in enumerate(results.multi_hand_landmarks):
            MCP_IDXS = [2, 5, 9, 13, 17]
            PIP_IDXS = [3, 6, 10, 14, 18]
            print(f"\n🖐 Hand {hand_id+1}: Estimated Joint Positions (mm)")
            print("Joint\t\tX(mm)\tY(mm)\tZ(mm)")
            print("-" * 40)
            for i, (mcp_idx, pip_idx) in enumerate(zip(MCP_IDXS, PIP_IDXS), start=1):
                for j_type, idx in zip(["MCP", "PIP"], [mcp_idx, pip_idx]):
                    lm = hand_landmarks.landmark[idx]
                    avg_xyz = find_joint_xyz(all_xyz, (lm.x, lm.y), grid_size, xy_range, flip_y)
                    if avg_xyz is not None:
                        x_mm, y_mm, z_mm = avg_xyz * 1000
                        print(f"{j_type}{i}\t\t{x_mm:7.2f}\t{y_mm:7.2f}\t{z_mm:7.2f}")
                        joint_xyz_dict[f"{j_type}{i}"] = avg_xyz
                        cx, cy = int(lm.x * img_bgr.shape[1]), int(lm.y * img_bgr.shape[0])
                        color = (255, 0, 0) if j_type == "MCP" else (0, 255, 0)
                        cv2.circle(img_bgr, (cx, cy), 6, color, -1)
                        cv2.putText(img_bgr, f"{j_type}{i}", (cx + 8, cy - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    else:
                        print(f"{j_type}{i}\t\t---\t---\t(No nearby points)")
    else:
        print("MediaPipe: no hands detected.")

    cv2.imshow("XY Projection + MCP/PIP Joints", img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if SAVE_OVERLAY:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, "xy_projection_with_MCP_PIP.png")
        cv2.imwrite(out_path, img_bgr)
        print(f"Saved overlay: {out_path}")

    return joint_xyz_dict


class raus_systsem_test():

    def __init__(self): 

        # UR3 robot
        self.path_ur3_config     = "./database/system/UR3/robot.yaml" 
        self.path_ur5e_config    = "./database/system/UR5e/robot.yaml" 

        # ip address
        self._host_ip_ur3        = '169.254.88.100'
        # self._host_ip_ur5e       = '10.162.76.24'

    def print_current_ee_cen_pts(self):
        """print the current ee center point"""

        tcp_in_world_tform    = self._ee_tform @ self._ee_to_tcp_tform
        # val_test = self._ee_tform @ self._ee_to_tcp_tform
        pts_tcp_in_world      = tcp_in_world_tform[0:3,3]
        rot_tcp_in_world      = tcp_in_world_tform[0:3,0:3]

        return pts_tcp_in_world, rot_tcp_in_world

    def print_current_q(self):

        global idx_global
        path_data_local = self._path_data_unique_folder

        # define the robot state 
        state = self._ptp_exp.state

        if not state:
            print('no robot state')
            return
        else: 
            print("the current ID = ", idx_global)
            print("the q_pose = ", state.actual_q)
            
            q_use = state.actual_q
            print("q_use = ", q_use)
            # np.save( path_data_local + str(idx_global) + ".npy", q_use )
            # idx_global += 1

            return state.actual_q 
        
    def control_exp(self):

        # set the mode 
        mode_use                        = "exp"
        # mode_use                        = "sim"
        self._is_ultrasound_mode        = True

        # robot configuration 
        # self._robot                 = load_robot( self.path_ur5e_config ) 
        self._robot                     = load_robot( self.path_ur3_config ) 

        # define the home configuration 
        qs_home                         = [-0.03559095, -1.30845672,  1.61374092, -1.85577661, -1.54902679, -1.61667949]
        self._qs_home                   = qs_home

        self._mode_ge_img               = '2D'

        # setup the ultrasound streaming 
        # unit-1: test the ge-ultrasound
        if self._is_ultrasound_mode: 
            self._display_size              = (500, 600)
            self._host_ge                   = '169.254.107.11'
            # self._mode_ge_img               = 'BMIAF'
            self._mode_ge_img               = '2D'
            # self._mode_ge_img               = 'CFM'
            self._obj_ultrasound            = Ultrasound()
            self._obj_ultrasound.connect( host = self._host_ge )
            self._obj_ultrasound.start( self._mode_ge_img )

        # folder definitions 
        idx_folder_unique_name             = "unit_test_1"
        self._path_data_unique_folder      = "./data_raus/" + idx_folder_unique_name +  "/"
        idx_exist_folder                   = os.path.isdir(self._path_data_unique_folder)
        if idx_exist_folder == False:
            print("The folder does not exist")
            os.mkdir(self._path_data_unique_folder)
        
        # global index
        global idx_global
        idx_global                  = 0 
        
        # exp connection 
        if mode_use == "exp":
            self._ctrl_exp          = self._robot.Controller( host  = self._host_ip_ur3, 
                                                              model = self._robot,
                                                              robot_info=('2017332424', '3.4.1.59')  )
            print("_ctrl_exp = ", self._ctrl_exp._robot_info)
        elif mode_use == "sim": 
            self._ctrl_exp          = Simulator( self._robot )
            self._ctrl_exp._q       = qs_home

        # create the sample viewers 
        (self._window, self._root)  = create_simple_viewer()

        # connect the ptp agent
        self._ptp_exp               = PointToPoint(self._robot, self._ctrl_exp )
        self._ptp_exp.connect()

        # step-2: setup the viewer and window
        self._crwa                  = ControlledRobotWidget(model = self._robot, 
                                                            controller = self._ptp_exp, 
                                                            parent = self._root, 
                                                            frames=[0, self._robot.dof ], 
                                                            show_mode='actual_q')
        
        path                        = PathWidget(parent=self._root)

        # basic settings
        self._robot.visual.parent   = self._root
        self._tform_ee_to_us_opt_res =  np.array([[ 1, 0,  0, 0], [0, 1, 0, 0], [0, 0, 1, 0.146], [ 0, 0, 0, 1]])
        self._tcp_xform             = np.array( self._tform_ee_to_us_opt_res )
        self._tcp_xform2 = np.array([[ 1, 0,  0, 0], [0, 1, 0, 0], [0, 0, 1, 0.146], [ 0, 0, 0, 1]])

        # adjust the speed 
        if mode_use == "sim": 
            self._qd_limit_input     = np.pi / 2
        elif mode_use == "exp": 
            self._qd_limit_input     = np.pi / 60
        self.idx_global_oct_data = 0 

        # define the tform widget globally
        self._scan_tform_widget     = TransformListWidget( parent = self._root, scale = 0.5 )

        mode_scan_use = "data_collection_test"

        if mode_scan_use == "data_collection_test":
            self._is_collect_cfm = False
            Thread(target = self.main_data_platform, daemon=True).start()

        # run the windows
        self._window.run()

    # def main_data_platform(self): 
    #     print("patient data collection platform")
    #     time.sleep(2.0)

    #     self._is_collect_cfm = False

    #     # --- Get current state and end-effector pose ---
    #     initial_state = self._ptp_exp.state
    #     initial_q = initial_state.actual_q
    #     initial_ee = self._robot.kinematics.forward(initial_q)
    #     print("Initial end effector pose:\n", initial_ee)

    #     q_seed = initial_q

    #     # =========================
    #     # 1️⃣ MCP scanning (Z=0.210)
    #     # =========================
    #     MCP_indices = [0, 2, 4, 6, 8]
    #     print(f"\n=== Moving to {len(MCP_indices)} MCP joints (Z = 0.210 m) ===")

    #     for i in MCP_indices:
    #         if i >= len(joint_xyz_robot_np):
    #             print(f"⚠️ Skipping index {i}: out of range")
    #             continue

    #         target_ee = initial_ee.copy()
    #         target_ee[:2, 3] = joint_xyz_robot_np[i, :2]   # use detected XY
    #         target_ee[2, 3]  = 0.220                      # fixed Z height

    #         print(f"\n➡️ Move to MCP#{int(i/2)+1} → XY = {joint_xyz_robot_np[i, :2]*1000} mm, Z = 210 mm")

    #         try:
    #             target_q = self._robot.kinematics.inverse_nearest(target_ee, q_seed)
    #             self._ptp_exp.move_joint(target_q, qd_limits=self._qd_limit_input)
    #             q_seed = target_q
    #             time.sleep(0.5)
    #         except Exception as e:
    #             print(f"❌ IK failed at MCP index {i}: {e}")
    #             continue

    #     print("✅ Finished MCP scanning sequence.")

    #     # =========================
    #     # 2️⃣ PIP scanning (Z=0.200)
    #     # =========================
    #     PIP_indices = [1, 3, 5, 7, 9]
    #     print(f"\n=== Moving to {len(PIP_indices)} PIP joints (Z = 0.200 m) ===")

    #     for i in PIP_indices:
    #         if i >= len(joint_xyz_robot_np):
    #             print(f"⚠️ Skipping index {i}: out of range")
    #             continue

    #         target_ee = initial_ee.copy()
    #         target_ee[:2, 3] = joint_xyz_robot_np[i, :2]   # use detected XY
    #         target_ee[2, 3]  = 0.200                      # fixed Z height (lower)

    #         print(f"\n➡️ Move to PIP#{int(i/2)+1} → XY = {joint_xyz_robot_np[i, :2]*1000} mm, Z = 200 mm")

    #         try:
    #             target_q = self._robot.kinematics.inverse_nearest(target_ee, q_seed)
    #             self._ptp_exp.move_joint(target_q, qd_limits=self._qd_limit_input)
    #             q_seed = target_q
    #             time.sleep(0.5)
    #         except Exception as e:
    #             print(f"❌ IK failed at PIP index {i}: {e}")
    #             continue

    #     print("✅ Finished PIP scanning sequence.")

    def main_data_platform(self): 
        print("patient data collection platform")
        time.sleep(2.0)

        self._is_collect_cfm = False

        # --- Get current robot state and pose ---
        initial_state = self._ptp_exp.state
        initial_q = initial_state.actual_q
        initial_ee = self._robot.kinematics.forward(initial_q)
        print("Initial end effector pose:\n", initial_ee)

        q_seed = initial_q

        # =========================
        # 1️⃣ MCP scanning (Z=0.210)
        # =========================
        MCP_indices = [0, 2, 4, 6, 8]
        print(f"\n=== Moving to {len(MCP_indices)} MCP joints (Z = 0.210 m) ===")

        for i in MCP_indices:
            if i >= len(joint_xyz_robot_np):
                print(f"⚠️ Skipping index {i}: out of range")
                continue

            target_ee = initial_ee.copy()
            target_ee[:2, 3] = joint_xyz_robot_np[i, :2]
            target_ee[2, 3]  = 0.210

            print(f"\n➡️ Move to MCP#{int(i/2)+1} → XY = {joint_xyz_robot_np[i, :2]*1000} mm, Z = 210 mm")

            try:
                target_q = self._robot.kinematics.inverse_nearest(target_ee, q_seed)
                self._ptp_exp.move_joint(target_q, qd_limits=self._qd_limit_input)
                q_seed = target_q
                time.sleep(0.5)
            except Exception as e:
                print(f"❌ IK failed at MCP index {i}: {e}")
                continue

        print("✅ Finished MCP scanning sequence.")

        # ===========================================
        # 2️⃣ Move between MCP → first PIP transition
        # ===========================================
        bridge_idx = 1  # First PIP joint (index 1)
        if bridge_idx < len(joint_xyz_robot_np):
            bridge_ee = initial_ee.copy()
            bridge_ee[:2, 3] = joint_xyz_robot_np[bridge_idx, :2]  # XY of first PIP
            bridge_ee[2, 3]  = 0.210                              # same height as MCP
            print(f"\n↪️ Moving to transition point: same XY as PIP#1, Z=210 mm")

            try:
                bridge_q = self._robot.kinematics.inverse_nearest(bridge_ee, q_seed)
                self._ptp_exp.move_joint(bridge_q, qd_limits=self._qd_limit_input)
                q_seed = bridge_q
                time.sleep(0.5)
            except Exception as e:
                print(f"❌ IK failed at transition point: {e}")

        # =========================
        # 3️⃣ PIP scanning (Z=0.200)
        # =========================
        PIP_indices = [1, 3, 5, 7, 9]
        print(f"\n=== Moving to {len(PIP_indices)} PIP joints (Z = 0.200 m) ===")

        for i in PIP_indices:
            if i >= len(joint_xyz_robot_np):
                print(f"⚠️ Skipping index {i}: out of range")
                continue

            target_ee = initial_ee.copy()
            target_ee[:2, 3] = joint_xyz_robot_np[i, :2]
            target_ee[2, 3]  = 0.200

            print(f"\n➡️ Move to PIP#{int(i/2)+1} → XY = {joint_xyz_robot_np[i, :2]*1000} mm, Z = 200 mm")

            try:
                target_q = self._robot.kinematics.inverse_nearest(target_ee, q_seed)
                self._ptp_exp.move_joint(target_q, qd_limits=self._qd_limit_input)
                q_seed = target_q
                time.sleep(0.5)
            except Exception as e:
                print(f"❌ IK failed at PIP index {i}: {e}")
                continue

        print("✅ Finished PIP scanning sequence.")


if __name__ == "__main__":

    # ========= 1️⃣ RealSense initialization =========
    os.makedirs(SAVE_DIR, exist_ok=True)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(SERIAL_NUMBER)
    config.enable_stream(rs.stream.color, *RESOLUTION, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, *RESOLUTION, rs.format.z16, 30)
    align = rs.align(rs.stream.color)
    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    print(f"Depth scale: {depth_scale:.6f} m/unit")

    # ========= 2️⃣ Capture 5 point clouds =========
    print("\n📸 Capturing 5 pointclouds (1s apart)...")
    pcs = []
    for i in range(5):
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        c = aligned.get_color_frame()
        d = aligned.get_depth_frame()
        if not c or not d:
            print(f"⚠️ Missing frame at index {i}")
            time.sleep(1)
            continue
        pc = rs_to_xyzrgb(c, d)
        if pc.size > 0:
            pcs.append(pc)
            fname = os.path.join(SAVE_DIR, f"pc_{i}.npy")
            np.save(fname, pc)
            print(f"✅ Saved {fname} ({pc.shape[0]} pts)")
        time.sleep(1.0)

    pipeline.stop()
    print(f"✅ Finished capturing {len(pcs)} valid pointclouds.\n")

    # ========= 3️⃣ MediaPipe Hand Detection =========
    if len(pcs) == 0:
        print("❌ No valid pointclouds — aborting detection and robot move.")
    else:
        print("🖐 Running MediaPipe hand joint detection...")
        joint_xyz = visualize_and_detect(pcs, SAVE_DIR)
        print("✅ Detection complete. Overlay saved to:", SAVE_DIR)

        # ========= 4️⃣ Transform to robot base frame =========
        print("\n🔄 Transforming hand joint coordinates to robot base frame...")

        T_base_from_cam = np.array([
            [-0.0022, -0.9990,  0.0450, -0.3994],
            [-0.9991,  0.0040,  0.0413, -0.1605],
            [-0.0415, -0.0448, -0.9981,  0.6085],
            [0,        0,        0,       1.0000]
        ])

        joint_xyz_robot = {}
        for name, p_cam in joint_xyz.items():
            p_h = np.append(p_cam, 1.0)  # homogeneous
            p_robot = T_base_from_cam @ p_h
            joint_xyz_robot[name] = p_robot[:3]
            print(f"{name:6s} → Robot frame [x y z] = [{p_robot[0]*1000:8.2f}, {p_robot[1]*1000:8.2f}, {p_robot[2]*1000:8.2f}] mm")
        
        joint_xyz_robot_np = np.vstack([v for v in joint_xyz_robot.values()])


        # ========= 5️⃣ Initialize robot and rotate 90° =========
        print("\n🤖 Initializing UR3 connection...")
        test_class = raus_systsem_test()
        test_class.control_exp()  # start robot viewer/control
