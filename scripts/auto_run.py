import os
import subprocess
import time

# Set the config dir
CONFIG_DIR = "./config/t2"

# Set the list of INTERVAL values
INTERVALS = ['second']

# Set the dir
OUTPUT_DIR = "./output/t2"

# Set the path to the main.py script
SCRIPT_PATH = "./main.py"

# Loop through each INTERVAL value
for encoder_type in os.listdir(CONFIG_DIR):
    if 'model' in encoder_type or 'test' in encoder_type or 'bkp' in encoder_type:
        print(f"Skipping {encoder_type} directory")
        continue
    
    for interval in INTERVALS:
        # Replace the ENCODER_TYPE value in the config file
        global_config_path = os.path.join(CONFIG_DIR, f"{encoder_type}/{interval}/global_config.yaml")
        data_config_path = os.path.join(CONFIG_DIR, f"{encoder_type}/{interval}/data_config.yaml")
        train_config_path = os.path.join(CONFIG_DIR, f"{encoder_type}/{interval}/train_config.yaml")
        draw_config_path = os.path.join(CONFIG_DIR, f"{encoder_type}/{interval}/draw_config.yaml")

        # Print paths
        print(global_config_path)
        print(data_config_path)
        print(train_config_path)
        print(draw_config_path)

        # Create output directory if it doesn't exist
        output_path = os.path.join(OUTPUT_DIR, f"{encoder_type}/{interval}")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print(f"{output_path} Created")
        else:
            print(f"{output_path} Exists")
            
        # Execute the main.py script with the updated config file
        main_py_command = ["mprof", "run", "-o", f"{output_path}/memory.dat", SCRIPT_PATH,
                        "--global_config", global_config_path, "--data_config", data_config_path,
                        "--train_config", train_config_path, "--draw_config", draw_config_path]
        subprocess.run(main_py_command, shell=True)

        # Plot the memory profile
        mprof_plot_command = ["mprof", "plot", "-t", f"{encoder_type}-{interval}-memory.png", "-o",
                        f"{output_path}/memory.png",
                        f"{output_path}/memory.dat"]
        subprocess.run(mprof_plot_command, shell=True)

        # Wait for a few seconds before the next iteration
        time.sleep(5)
