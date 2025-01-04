import numpy as np
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the name of the .npy file (replace 'example.npy' with your actual file name)
file_name = 'linear-q-path.npy'
print("hello")


# Create the full path to the .npy file
file_path = os.path.join(current_dir, file_name)

# Load the .npy file
data = np.load(file_path)

for i in range(data.shape[0]):
    # Extract the ith row
    row_data = data[i, :]

    # Define the output file name (e.g., robot_config_0.npy, robot_config_1.npy, ...)
    output_file_name = f'combine_linear_robot/robot_config_{i}.npy'

    # Create the full path to the output file
    output_file_path = os.path.join(current_dir, output_file_name)

    # Save the row data as a new .npy file
    np.save(output_file_path, row_data)