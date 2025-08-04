[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14534966.svg)](https://doi.org/10.5281/zenodo.14534966)


# Table of Contents

1. [Introduction](#introduction)
2. [Project Information](#project-information)
   - [Memory Space Requirements](#memory-space-requirements)
   - [GPU Management](#gpu-management)
3. [Setup](#setup)
   - [1.1 Prepare CATHe2 Models](#11-prepare-cathe2-models)
   - [1.2 Verify Root User](#12-verify-root-user)
   - [2.1 Set Up the Virtual Environment](#21-set-up-the-virtual-environment)
   - [2.2 Activate the Virtual Environment](#22-activate-the-virtual-environment)
   - [3.1 Prepare Primary Sequences for Inference](#31-prepare-primary-sequences-for-inference)
   - [3.2 Prepare PDB Files for Inference](#32-prepare-pdb-files-for-inference)
   - [4 Run the Inference](#4-run-the-inference)
   - [5 Look at the Result](#5-look-at-the-result)
4. [Data](#data)
5. [Pre-Print](#pre-print)





# Introduction
This project is based on the work of Vamsi Nallapareddy https://github.com/vam-sin/CATHe and the CATHe team [CATHe paper](https://pubmed.ncbi.nlm.nih.gov/36648327/)

CATHe (short for CATH embeddings) is a deep learning tool designed to detect remote homologues (up to 20% sequence similarity) for superfamilies in the CATH database. CATHe consists of an artificial neural network model which was trained on sequence embeddings from the ProtT5 protein Language Model (pLM). It was able to achieve an accuracy of 85.6% +- 0.4% (F1 score of 72%), and outperform the other baseline models derived from both, simple machine learning algorithms such as Logistic Regression, and homology-based inference using BLAST. 

CATHe2 is an improved version of CATHe with a different architecture, using embeddings from the ProstT5 pLM. CATHe2 is also able to take 3D structure information as input as well as protein primary sequences, via 3Di sequences derived from PDB files. This allows CATHe2 to reach an accuracy of 92.2% (F1 score of 82.3%).

To know more about CATHe2 and how it was built, see (put CATHe2 link when published) [CATHe2 paper](https://...)

# Project information
This project was tested on Ubuntu 22.04

## Memory space requirements:

To run inferences with the former version of CATHe, you need **20 GB** of free disk space.

To run inferences with the new version you need **24 GB** of free disk space.

To be able to test the training process and check the CATHe2 data you need **70 GB** of free disk space.

## GPU management

When running this project code, if you see a message indicating that the GPU is not used it is probably because the version of CUDA, cuDNN and/or the nvidia driver is not compatible with the tensorflow version used (TensorFlow 2.14.0). Note: For a small number of inferences, GPU usage is optional.

The versions recommended for this project are

CUDA 11.8  
cuDNN 8.7   
nvidia-driver-460

See the commented code in venv_1_setup.sh or venv_2_setup.sh to get some indications on how to install these.

You can check your CUDA version with ```nvcc --version``` and your GPU availability as well as driver version with ```nvidia-smi```.
# Setup

## 1.1 Prepare CATHe2 models
- Change the working directory to the project root
- Download CATHe2 models and place them in the right folder by running

```bash
chmod +x ./CATHe2_setup.sh
./CATHe2_setup.sh
```

## 1.2 Verify root user
- Verify you are the root user running

```bash
whoami
```

- If you are not the root user (if whoami does not return “root”) add your user to the sudo list

```bash
usermod -aG sudo your-username
```

## 2.1 Set Up the Virtual Environment
Setup the right venv based on the CATHe model you want to use, venv_1 for the former version (with ProtT5) and venv_2 for the new version (with ProstT5)

```bash
chmod +x ./venv_1_setup.sh
./venv_1_setup.sh

or

chmod +x ./venv_2_setup.sh
./venv_2_setup.sh
```
venv_1 takes ~6.3 GB
venv_2 takes ~10.3 GB

## 2.2 Activate the Virtual Environment
- And activate it

```bash
source venv_1/bin/activate

or

source venv_2/bin/activate
```

## 3.1 Prepare primary sequences for inference
Put the protein sequences for which you want to predict the CATH annotation into a `FASTA` file named `Sequences.fasta`, in the `./src/cathe-predict` folder.

## 3.2 Prepare PDB files for inference
If you want to use both 3Di sequences and Amino Acid (AA) sequences to predict the CATH annotation, ensure that the corresponding PDB files for the sequences in `Sequences.fasta` are placed in the `PDB_folder` folder located in `./src/cathe-predict`. Each PDB file must be prefixed by the index of its respective sequence in `Sequences.fasta`, followed by an underscore. For example:

If my `Sequences.fasta` has 3 sequences of protein domain in this order:  protein domain corresponding to the 3hhl.pdb file, protein domain corresponding to the 4jkm.pdb file and protein domain corresponding to the 3ddn.pdb file, then the corresponding PDB files in `./src/cathe-predict/PDB_folder` should be renamed:

```
0_3hhl.pdb
1_4jkm.pdb
2_3ddn.pdb
```

## 4 Run the inference
Then you can launch the desired version of CATHe to predict CATH annotation.

- To use the old version (with model ProtT5 and input type AA only), run

```bash
python ./src/cathe-predict/cathe_predictions.py 
```

(venv_1 has to be activated)

- To use the new version of CATHe, i.e CATHe2  (for input type AA only)

run

```bash
python ./src/cathe-predict/cathe_predictions.py --model ProstT5 --input_type AA
```
(venv_2 has to be activated)

- To use the new version of CATHe, i.e CATHe2  (with 3Di input as well as AA).

```bash
python ./src/cathe-predict/cathe_predictions.py --model ProstT5 --input_type AA+3Di
```
 (You need to fill `PDB_folder` accordingly to run this, see [paragraph 3.2](#32), make sure venv_2 is activated too)

 If you encounter an error during sequence embedding indicating that the number of embeddings does not match the number of sequences, it is likely that you need to lower the values of the following variables, at the beginning of CATHe2/src/model_building/models ProstT5_Ankh_TMVec_ESM2_ProtT5new/embed_all_new_models.py file, in order to fit each batch of sequences to embed in your GPU memory:
 
 ```
 max_res_per_batch = 4096 
 nb_seq_max_per_batch = 4096
```

 (These two variables control the maximum number of residues and sequences per batch, respectively. You can keep it simple by setting both to the same value, e.g 512 should work for an 8 GB GPU)


## 5 Look at the Result

The Results will be in `./src/cathe-predict/Results.csv`

# Data

The dataset used for training, optimizing, and testing CATHe2 was derived from the CATH database. The datasets, along with the weights for the CATHe2 artificial neural network as well as all the intermediary training files can be downloaded from Zenodo from this link: [Dataset](https://doi.org/10.5281/zenodo.14534966).
Or running the following code at the project root:

```bash
 wget https://zenodo.org/records/14970431/files/data.zip?download=1 -O data.zip
 unzip ./data.zip
 rm -f data.zip
 ```
 This is not necessary to run CATHe2 inferences.

 data.zip is 30 GB




# CATHe2 Code Location Guide

The main components extending the original CATHe project are located in the following files and directories:

- **New ANN Classifiers:**  
   `CATHe2/src/model_building/models/ann_all_new_models.py`  
   Contains code for creating and training the new artificial neural network classifiers.

- **Embedding with New Protein Language Models:**  
   `CATHe2/src/model_building/models/ProstT5_Ankh_TMVec_ESM2_ProtT5new/embed_all_new_models.py`  
   Implements embedding generation using updated protein language models.

- **3Di Sequence Processing:**  
   `CATHe2/src/model_building/get_3Di`  
   Handles the processing related to 3Di structural information.


# Pre-Print

If you found this work useful, please consider citing the following article:

```
@article {Mouret2025.06.22.660903,
	author = {Mouret, Orfe{\'u} and Abbass, Jad},
	title = {CATHe2: Enhanced CATH Superfamily Detection Using ProstT5 and Structural Alphabets},
	elocation-id = {2025.06.22.660903},
	year = {2025},
	doi = {10.1101/2025.06.22.660903},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/06/26/2025.06.22.660903},
	eprint = {https://www.biorxiv.org/content/early/2025/06/26/2025.06.22.660903.full.pdf},
	journal = {bioRxiv}
}
```
