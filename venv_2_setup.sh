#!/bin/bash

# Step 1: Deactivate the current virtual environment if any is active
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Deactivating the current virtual environment..."
    deactivate
else
    echo "No virtual environment is currently active."
fi

# Step 2: Create a new virtual environment and activate it
echo "Creating and activating new virtual environment (venv_2)..."
python3.9 -m venv venv_2
source venv_2/bin/activate

# Step 3: Upgrade pip and install setuptools
echo "Upgrading pip and installing setuptools..."
pip install --upgrade pip
pip install setuptools==69.5.1

# Step 4: Install gensim, bio-embeddings, wheel, and other packages
echo "Installing gensim, wheel, lazr.uri..."
pip install gensim==4.3.2
pip install wheel lazr.uri

# Step 5: Install PyTorch and its dependencies with CUDA 11.8 support
echo "Installing PyTorch with CUDA 11.8 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Step 6: Set the PyTorch path
echo "Setting PYTHONPATH for PyTorch..."
export PYTHONPATH=$(python -c 'import site; print(site.getsitepackages()[0])')

# Step 7: Modify gensim's matutils.py to fix an import issue
GENSIM_FILE="venv_2/lib/python3.9/site-packages/gensim/matutils.py"
if [[ -f "$GENSIM_FILE" ]]; then
    echo "Modifying gensim's matutils.py..."
    sed -i 's/from scipy.linalg import get_blas_funcs, triu/from scipy.linalg import get_blas_funcs\nfrom numpy import triu/' "$GENSIM_FILE"
else
    echo "Could not find gensim's matutils.py at $GENSIM_FILE. Please verify the path."
fi

# Step 8: Set environment variables for CUDA in the virtual environment
echo "Setting CUDA environment variables in venv_2..."
echo 'export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}' >> ./venv_2/bin/activate
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ./venv_2/bin/activate

# Step 9: Deactivate and reactivate the virtual environment
echo "Deactivating and reactivating venv_2..."
deactivate
source venv_2/bin/activate

# Step 10: Install requirements from requirements_2.txt
echo "Installing Python requirements from requirements_2.txt..."
pip install --use-deprecated=legacy-resolver -r requirements_2.txt

# Step 11: Install MMseqs2
echo "Installing MMseqs2..."
sudo apt install -y mmseqs2

# Step 12: Install TM_Vec, Ankh, and FAISS-GPU
echo "Installing TM_Vec, Ankh, and FAISS-GPU..."
pip install tm_vec
pip install ankh
pip install faiss-gpu

# Step 13: Install foldseek
echo "Installing foldseek..."
wget https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz
tar xvzf foldseek-linux-avx2.tar.gz

echo "All tasks completed successfully."
