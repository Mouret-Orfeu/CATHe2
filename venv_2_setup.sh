#!/bin/bash

echo "venv_2 Pre requirement set up"

# Add Python 3.9 repository and update the system
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.9 python3.9-venv -y

# Create a new virtual environment and activate it
python3.9 -m venv venv_2
source venv_2/bin/activate

# Upgrade pip and install setuptools
pip install --upgrade pip
pip install setuptools==69.5.1

# Install manually gensim, wheel and lazr.uri
pip install gensim==4.3.2
pip install wheel lazr.uri

sudo apt-get install -y brltty command-not-found python3-cups libatlas-base-dev liblapack-dev libblas-dev libsystemd-dev pkg-config python3.9-dev libcairo2-dev libpq-dev libgirepository1.0-dev libdbus-1-dev libhdf5-dev build-essential libssl-dev libffi-dev python3-dev llvm gfortran libopenblas-dev liblapack-dev libcups2-dev


# Install PyTorch and its dependencies with CUDA 11.8 support

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Set the PyTorch path
export PYTHONPATH=$(python -c 'import site; print(site.getsitepackages()[0])')

# Modify gensim matutils.y
sed -i 's/from scipy.linalg import get_blas_funcs, triu/from scipy.linalg import get_blas_funcs\nfrom numpy import triu/' ./venv_2/lib/python3.9/site-packages/gensim/matutils.py


# # Set environment variables for CUDA in the virtual environment
# echo 'export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}' >> ./venv_2/bin/activate
# echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ./venv_2/bin/activate

# Deactivate and reactivate the virtual environment
deactivate
source venv_2/bin/activate

# Install requirements from requirements_2.txt
echo "Installing Python requirements from requirements_2.txt, this may take a while..."
pip install --use-deprecated=legacy-resolver -r requirements_2.txt

# Install MMseqs2
sudo apt install -y mmseqs2

# Install foldseek
echo "Installing foldseek..."
wget https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz
tar xvzf foldseek-linux-avx2.tar.gz
rm foldseek-linux-avx2.tar.gz
