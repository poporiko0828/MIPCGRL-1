#!/bin/bash

# Docker-based execution script (automatic allocation of empty GPUs and log saving)
# Usage: ./run_in_docker_with_gpu.sh python_file key1=value1 key2=value2 ...

# Receives the Python script file name as the first argument
python_file="$1"
shift  # remove first argument

# 나머지 인자들을 key=value 형태로 처리
ARGS=("$@")

# Process the remaining arguments in key=value format
param_str=""
for arg in "${ARGS[@]}"; do
    param_str+="$arg "
done

# Check CUDA version (use nvidia-smi)
cuda_version=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+")
# Select Docker image
if [ "$cuda_version" -eq 12 ]; then
    docker_image="your docker image for cuda 12"
else
    echo "Unsupported CUDA version: $cuda_version"
    exit 1
fi

# Log directory settings
mkdir -p output_logs error_logs
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="output_logs/output_${timestamp}.log"
error_log_file="error_logs/error_${timestamp}.log"

# GPU selection logic
echo "Searching for available GPU..."

# Check GPU memory usage with `nvidia-smi`
gpu_info=$(nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits)
available_gpu=$(echo "$gpu_info" | awk -F, '{if ($5 > 0) print $1 " " $5}' | sort -k2 -nr | head -n1 | cut -d' ' -f1)

# IF $CUDA_VISIBLE_DEVICES is set, use it
if [ -n "$GPU" ]; then
    available_gpu=$GPU
fi

if [ -z "$available_gpu" ]; then
    echo "No available GPU found!" | tee -a "$error_log_file"
    exit 1
fi

# Get selected GPU details
selected_gpu_info=$(echo "$gpu_info" | awk -v gpu_id="$available_gpu" -F, '{if ($1 == gpu_id) print}')
selected_gpu_name=$(echo "$selected_gpu_info" | cut -d, -f2)
selected_gpu_total_mem=$(echo "$selected_gpu_info" | cut -d, -f3)
selected_gpu_used_mem=$(echo "$selected_gpu_info" | cut -d, -f4)
selected_gpu_free_mem=$(echo "$selected_gpu_info" | cut -d, -f5)

# Selected GPU output
echo "Selected GPU: $available_gpu (GPU Number: $available_gpu)" | tee -a "$log_file"
echo "GPU Details:" | tee -a "$log_file"
echo "  GPU ID: $available_gpu" | tee -a "$log_file"
echo "  Model Name: $selected_gpu_name" | tee -a "$log_file"
echo "  Total Memory: ${selected_gpu_total_mem}MiB" | tee -a "$log_file"
echo "  Used Memory: ${selected_gpu_used_mem}MiB" | tee -a "$log_file"
echo "  Free Memory: ${selected_gpu_free_mem}MiB" | tee -a "$log_file"


# List of arguments to exclude
exclude_args=("wandb_project" "n_envs" "overwrite" "use_nlp" "wandb_key" "traj_path" "buffer_path")

# Initialize container name with a base name
container_name="your container name"

for arg in "$@"; do
    # Separating argument names and values with '='
    key=$(echo "$arg" | cut -d '=' -f 1)

    # Process only if the factor is not to be excluded
    exclude=false
    for exclude_arg in "${exclude_args[@]}"; do
        if [[ "$key" == "$exclude_arg" ]]; then
            exclude=true
            break
        fi
    done

    if [[ "$exclude" == false ]]; then
        # Replace '=' with '-', '.' with '_', and '/' with '-'.
        sanitized_arg=$(echo "$arg" | sed 's/=/-/g; s/\./_/g ; s/\//-/g')
        container_name="${container_name}_${sanitized_arg}"
    fi
done

echo "Container Name: $container_name"
echo "Output Log File: $log_file"


root_args=("traj_path")

for arg in "$@"; do
    # Separating argument names and values with '='
    key=$(echo "$arg" | cut -d '=' -f 1)

    # Process only if the factor is not to be excluded
    for root_arg in "${root_args[@]}"; do
        if [[ "$key" == "root_arg" ]]; then
            user_param="-u $(id -u):$(id -g)"
            break
        fi
    done
done


# Docker execution command
docker_command="docker run --rm -it
    -v $(pwd):/workspace
    -w /workspace
    --gpus all
    -e CUDA_VISIBLE_DEVICES=$available_gpu
    -e XLA_PYTHON_CLIENT_PREALLOCATE=false
    -e XLA_PYTHON_CLIENT_MEM_FRACTION=.95
    -v /mnt/nas:/mnt/nas
    -v /raid:/raid
    -v $(pwd)/.netrc:/.netrc
    --network=host
    -e HF_HOME=/workspace/cache/huggingface
    --name \"$container_name\"
    $user_param \
    $docker_image
    python $python_file $param_str"

echo "Executing Docker command:" | tee -a "$log_file"
echo "$docker_command" | tee -a "$log_file"

# Docker command execution and logging
{
    eval $docker_command
} 2>&1 | tee -a "$log_file"

# Check execution results
exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "Execution failed. Check logs for details." | tee -a "$error_log_file"
    echo "Docker logs (last 10 lines of $log_file):" | tee -a "$error_log_file"
    tail -n 10 "$log_file" | tee -a "$error_log_file"
    exit $exit_code
else
    echo "Execution completed successfully." | tee -a "$log_file"
fi

exit $exit_code
