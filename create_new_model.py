import os
import shutil

# --- Configs ---
model_nr = 4

file_to_run = "train_model.py"
job_name = "KAN"
email = "cristianplacinta04@gmail.com"
GPU_model = "gpuv100"
nr_CPU_cores = 4
memory_per_core = 512  # in MB

main_folder = 'KAN_' + str(model_nr)
results_folder = os.path.join(main_folder, 'results')
error_folder = os.path.join(main_folder, 'errors')
output_folder = os.path.join(main_folder, 'output')
run_model_file = os.path.join(main_folder, 'run_model.lsf')
train_model_file = os.path.join(main_folder, 'train_model.py')
try_model_file = os.path.join(main_folder, 'try_model.py')


if not os.path.exists(main_folder):
    os.makedirs(main_folder)

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    if not os.path.exists(error_folder):
        os.makedirs(error_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    lsf_content = f"""
#!/bin/bash

#BSUB -J {job_name}
#BSUB -q {GPU_model}
#BSUB -n {nr_CPU_cores}

# -- Request 1 GPU in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

#BSUB -R "rusage[mem={memory_per_core}]"
#BSUB -W 04:00
#BSUB -e errors/kan_%J.err
#BSUB -u {email}
#BSUB -B
# -- BSUB -N --
module purge
module load python3/3.9.19
module load cuda/11.6
source ~/PNJ/venv/bin/activate

python -u ~/PNJ/KAN_{model_nr}/{file_to_run} > output/model_{model_nr}_$LSB_JOBID.out 2>&1
"""

    # Create and write the content to run_model.lsf file
    with open(run_model_file, 'w') as f:
        f.write(lsf_content)

    print(f"Created file: {run_model_file}")

    # Check for data.npy in the previous model folder and copy if it exists
    prev_folder = 'KAN_' + str(model_nr - 1)
    prev_data_file = os.path.join(prev_folder, 'data.npy')
    current_data_file = os.path.join(main_folder, 'data.npy')

    if os.path.exists(prev_data_file):
        shutil.copy(prev_data_file, current_data_file)
        print(f"Copied 'data.npy' from {prev_folder} to {main_folder}")
    else:
        print(f"\n---> No 'data.npy' found in {prev_folder}. <---\n")

    if not os.path.exists(train_model_file):
        with open(train_model_file, 'w') as f:
            f.write("# Empty train_model.py file\n")
        print(f"Created empty file: {train_model_file}")
    else:
        print(f"File already exists: {train_model_file}")


    if not os.path.exists(try_model_file):
        with open(try_model_file, 'w') as f:
            f.write("# Empty try_model.py file\n")
        print(f"Created empty file: {try_model_file}")
    else:
        print(f"File already exists: {try_model_file}")


else:
    print("Model setup already created.")
