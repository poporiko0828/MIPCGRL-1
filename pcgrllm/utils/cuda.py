import os
import subprocess

CUDA_VERSION = None

def get_cuda_version():
    global CUDA_VERSION  # To modify the global variable
    CUDA_VERSION = False  # Default value if no valid version is found
    try:
        # Run the command and suppress console output
        result = subprocess.run(
            ["nvcc", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        cuda_version = result.stdout
        for line in cuda_version.split("\n"):
            if "release" in line:
                version = line.split("release")[-1].strip().split(" ")[0]
                version = version.replace(",", "")
                CUDA_VERSION = float(version)
                break
    except Exception as e:
        # Handle the error gracefully
        CUDA_VERSION = False
    return CUDA_VERSION

if __name__ == "__main__":
    print(f'CUDA: {get_cuda_version()}')