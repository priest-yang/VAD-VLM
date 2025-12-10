#!/bin/bash

# Function to check and install CUDA 11.8 if not present
check_and_install_cuda() {
  # Check if CUDA 11.8 is in the /usr/local/ directory
  if [ ! -d "/usr/local/cuda-11.8" ]; then
    echo "CUDA 11.8 is not installed. Installing CUDA Toolkit 11.8..."
    sudo apt update
    sudo apt install -y cuda-toolkit-11-8

    # Set CUDA_HOME environment variable
    export CUDA_HOME=/usr/local/cuda-11.8
    echo 'export CUDA_HOME=/usr/local/cuda-11.8' >> ~/.bashrc
    echo "CUDA_HOME is set to /usr/local/cuda-11.8"
  else
    echo "CUDA 11.8 is already installed."
    export CUDA_HOME=/usr/local/cuda-11.8
    echo 'export CUDA_HOME=/usr/local/cuda-11.8' >> ~/.bashrc
  fi
}

# Function to check the CUDA version of installed PyTorch
check_cuda_version() {
  # Get the torch version and cuda version from Python
  cuda_version=$(python3.8 -c "import torch; print(torch.version.cuda)" 2>/dev/null)

  # If CUDA version retrieval fails or version is not 11.8
  if [ $? -ne 0 ] || [ "$cuda_version" != "11.8" ]; then
    echo "Re-installing PyTorch, Torchvision, and Torchaudio with CUDA 11.8..."
    python3.8 -m pip uninstall torch torchvision torchaudio
    python3.8 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  else
    echo "PyTorch is already using CUDA 11.8"
  fi
}

# Check if torch is installed

check_and_install_cuda
echo "export PATH=/usr/bin:$PATH" >> ~/.bashrc

if python3.8 -c "import torch" &> /dev/null; then
  check_cuda_version
else
  echo "PyTorch is not installed. Installing with CUDA 11.8..."
  python3.8 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

python3.8 -m pip install ninja packaging
export PATH=/usr/bin:$PATH
export CUDA_HOME=/usr/local/cuda-11.8

/usr/bash

# uninstall mmcv mmdet3d mmdet
python3.8 -m pip uninstall mmcv mmdet3d mmdet
rm -rf /home/lightwheel/.local/lib/python3.8/site-packages/mmcv/


#
sudo apt install python3-tk