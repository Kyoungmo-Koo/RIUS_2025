%% 3D sensor fusion 
% linear-scanning patterns 

clc; 
clear all; 
close all;

% visualization flag 
flag_vis_plane              = "true"; 
color_vec                   = ["r", "g", "b", "w", "y", "c"]; 

% testing dataset
path_data_folder            = strcat( "/Users/kookyoungmo/Desktop/Bscans/" );  
var_base_img                = strcat( path_data_folder, "images/20241123_finger_final2/combine_linear_scan/" ); 
var_base_robot              = strcat( path_data_folder, "images/20241123_finger_final2/combine_linear_robot/" ); 
list_of_img_idx             = [ 15, 17, 14, 14, 19, 14 ];
list_of_fid_cen_pxl         = [ [460, 578]; [416, 540]; [389, 601]; [422, 572]; [521, 581]; [398, 496] ]; 

% robot configuration 
index_of_robot             = "universalUR3"; 
eng_ur3                    = loadrobot( index_of_robot );

% sensor calibration (sensor to ee) 
var_snesor_calib_input      = [ 0, 0, 0, 0.129, 0, 0];
% var_snesor_calib_input      = [ 0, 0, 0, 0.129, 0, 0];
theta_ee_to_sensor_use      = var_snesor_calib_input(1:3); 
trans_ee_to_sensor_use      = var_snesor_calib_input(4:6); 

% basic parameter settings
num_of_robot_pose           = 6; 
num_of_pose_each_scan       = 180; % Have to change
ratio_thres_binary_bscan    = 0.1;
vec_len_scale               = 0.3;
list_predicted_cen_3d       = [];
scale_global_factor         = 1; 
len_pxl_height              = 1083; 
len_mm_height               = 30.0 * scale_global_factor;
len_pxl_width               = 903;
len_mm_width                = 25.0 * scale_global_factor; 
ratio_of_height             = len_pxl_height / len_mm_height;             
ratio_of_width              = len_pxl_width  / len_mm_width;            
x_cen_ref_pxl               = len_pxl_width  / 2; 
ratio_mm_to_meter           = 0.001;
x_cube_len                  = 0.030;    % unit: meter
len_us_plane                = 1083;
wid_us_plane                = 903; 
y_cube_len                  = x_cube_len * 2 * ( len_us_plane ./ wid_us_plane ) ; 

%% develop the simulation program 
close all; 

idx_robot_q_config                                  = 1;   
idx_img_track                                       = list_of_img_idx( idx_robot_q_config );
pixel_cen_label                                     = list_of_fid_cen_pxl( idx_robot_q_config, : ); 
pts_global_vol                                      = []; 
col_global_vol                                      = []; 


% Initialize storage for 120 frames (X, Y, Z axes, and origins)
num_frames = num_of_pose_each_scan; % Adjust as per your dataset
data_matrix = zeros(num_frames, 12); % 12 columns: 3 for X, 3 for Y, 3 for Z, 3 for Origin

for idx_pose_each_scan = 1:num_frames
    % Load robot pose configuration
    path_get_robot_pose_tmp = strcat(var_base_robot, "robot_config_", num2str(idx_pose_each_scan - 1), ".npy");
    pose_robot_tmp = readNPY(path_get_robot_pose_tmp)';

    % Compute robot configuration transformation
    para_input = [];
    para_input.eng_ur3 = eng_ur3;
    para_input.robot_q_input = pose_robot_tmp;
    para_input.flag_idx = 1;
    [para_robot_output] = robot_ur3_config_api_raus(para_input);

    % Extract transformation matrix of the end-effector
    tform_ee_to_sensor_use = tform_3d_homogenous_raus( theta_ee_to_sensor_use, trans_ee_to_sensor_use ); 
    tform_ee_in_world_local = para_robot_output.ee_link_tform;
    tform_sensor_in_world_local = tform_ee_in_world_local * tform_ee_to_sensor_use; 

    % Extract origin and axes from transformation matrix
    origin = tform_sensor_in_world_local(1:3, 4)' % Origin as row vector
    x_axis = tform_sensor_in_world_local(1:3, 1)'; % X-axis as row vector
    y_axis = tform_sensor_in_world_local(1:3, 2)'; % Y-axis as row vector
    z_axis = tform_sensor_in_world_local(1:3, 3)'; % Z-axis as row vector

    % Save into matrix
    data_matrix(idx_pose_each_scan, :) = [x_axis, y_axis, z_axis, origin];
end

% Save to .mat file
save('frame_axes_data_20250102_linear.mat', 'data_matrix');