# RIUS_2025
Robot Integrated Ultrasound System

# check_numpy.py
import 'linear-q-path.npy' file and store .npy file that contains robot configuration of each pose.

# each_plane_xyz_pos.m
Matlab file for calculating the position of origin of each plane and x, y, z vectors of each plane considering robot configuration of each pose.

# 20250101_CPU.py, 20250101_CPU.ipynb
Python files for generating 3D reconstruction result using image and origin, x_vector, y_vector, and z_vector of each plane. (80GB CPU)

# 20250101_GPU.py, 20250101_GPU.ipynb
Python files for generating 3D reconstruction result using image and origin, x_vector, y_vector, and z_vector of each plane. But use GPU. (80GB CPU, 40GB GPU)
