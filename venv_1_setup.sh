#!/bin/bash

echo "venv_1 Pre requirement set up"

# Add Python 3.9 repository and update the system
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.9 python3.9-venv -y

# Create a virtual environment and activate it
python3.9 -m venv venv_1
source venv_1/bin/activate

# Upgrade pip and install setuptools
pip install --upgrade pip
pip install setuptools==69.5.1

# # Install gensim and bio-embeddings
pip install gensim==4.3.2
pip install git+https://github.com/sacdallago/bio_embeddings.git@develop

# Install system packages
sudo apt-get install -y brltty command-not-found python3-cups
sudo apt-get install -y libatlas-base-dev liblapack-dev libblas-dev
sudo apt-get install -y libsystemd-dev pkg-config python3.9-dev libcairo2-dev
sudo apt-get install -y libpq-dev libgirepository1.0-dev libdbus-1-dev libhdf5-dev build-essential libssl-dev libffi-dev python3-dev llvm
sudo apt-get install -y gfortran libopenblas-dev liblapack-dev libcups2-dev

# Install additional Python packages
pip install wheel lazr.uri

# Set torch path (optional, check if needed)
export PYTHONPATH=$(python -c 'import site; print(site.getsitepackages()[0])')

# Modify gensim matutils.y
sed -i 's/from scipy.linalg import get_blas_funcs, triu/from scipy.linalg import get_blas_funcs\nfrom numpy import triu/' ./venv_1/lib/python3.9/site-packages/gensim/matutils.py


# # Install NVIDIA driver
# echo "Installing NVIDIA driver..."
# sudo apt update
# sudo apt install -y nvidia-driver-460
# sudo update-initramfs -u

# # Install CUDA toolkit
# echo "Installing CUDA Toolkit 11.8..."
# wget http://archive.ubuntu.com/ubuntu/pool/universe/n/ncurses/libtinfo5_6.3-2ubuntu0.1_amd64.deb
# sudo dpkg -i libtinfo5_6.3-2ubuntu0.1_amd64.deb
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
# sudo dpkg -i cuda-keyring_1.1-1_all.deb
# sudo apt-get update
# sudo apt-get -y install cuda-toolkit-11-8

# # Install cuDNN 8.7 for CUDA 11.8
# echo "Installing cuDNN 8.7 for CUDA 11.8..."
# sudo dpkg -i ./cudnn-local-repo-ubuntu2204-8.7.0.84_1.0-1_arm64.deb
# sudo cp /var/cudnn-local-repo-ubuntu2204-8.7.0.84/cudnn-local-BF23AD8A-keyring.gpg /usr/share/keyrings/
# sudo apt-get update
# sudo apt-get install -y libcudnn8 libcudnn8-dev libcudnn8-samples

# # Set environment variables in the virtual environment
# echo "Setting CUDA environment variables in venv..."
# echo 'export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}' >> ./venv_1/bin/activate
# echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ./venv_1/bin/activate

# Deactivate and reactivate the virtual environment
deactivate
source venv_1/bin/activate


# Install Python requirements
echo "Installing Python requirements for venv_1, this may take a while..."
pip install --use-deprecated=legacy-resolver -r requirements.txt

# Install MMseqs2
sudo apt install -y mmseqs2

# Install Foldseek
echo "Installing Foldseek..."
wget https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz
tar xvzf foldseek-linux-avx2.tar.gz
rm foldseek-linux-avx2.tar.gz
export PATH=$(pwd)/foldseek/bin/:$PATH

# Final system reboot to apply changes
# echo "Rebooting the system to apply new NVIDIA driver..."
# sudo reboot
