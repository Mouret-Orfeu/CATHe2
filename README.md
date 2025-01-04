[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14534966.svg)](https://doi.org/10.5281/zenodo.14534966)



# Introduction
This project is based on the work of Vamsi Nallapareddy https://github.com/vam-sin/CATHe and the CATHe team [CATHe paper](https://watermark.silverchair.com/btad029.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAA3EwggNtBgkqhkiG9w0BBwagggNeMIIDWgIBADCCA1MGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQM-he7uvdTMzk1zDXOAgEQgIIDJGfpD_jUcvAY66ZNH4KdwDL1Fvj6SBj1iXcAvfKW-dgVyvwYBUK7CBP2chYewliixDX_ZqpjV41EDy9gziM5G4A23RFVZqbpTpCehzKi9s0KcNRvdnp2q44Buv9STjbzFzdFu-9fNQKOMnSZ9dR8Dz3Pi0oqEPyrVuO2VeEJwriSqVoHXoS9IrG-Kt1M0EwEe1pYe8rK_g1Di-U89JAfQOadpvN3TlzU0FJJUm39H2yjV7wzYXcVnaUDEIyylq4I0d3bRXe2pUj3eZ_kayClbd9uGCLMiL6d5eYAGikZluyr4Ih6SIWYrvI2ItLeR4s_GqQLVErq1Xe0CFkakeIvI4JdNWUFI0N6MuDKjQDe532qPndw9I6eTkYhMfJIUob6ABel1bxujvMJdhuNtf1M4RnBr_72KTeEvkHVZh20d-dSek9RqnPmiihqtKcHc_h0gh0aIuuU2zvrGxiEjQiyLdRut5R9T0Xk4_bKPEmvfD9czULAT3dQ-wOFA6JtjoKjM2mVhLcBmPDR4-sXYp9musfv1ISYsGZ4tOs7_ssjs0u-7wpG1xL5T_walXCQ_2GtM7vV2ZTrPTW2Jl3yAlRwYsXqQlhhiYWC3hMsCdf2La0b65u9Lqzl5pUQjGvAurYk07-Ykcn_4dH-KNJfCj7ZaVn5AVKOdDg6qZNZBGNtEB6IIxr3pCaN1_VuwIbMIITlyihoDBAUrvDL2J1V7p6VIHtjN2GgaBl07lJoI4iAMuw83M8qizVEBTZ0PvYfUOmJq3ZM7ZroFuvTyiyvhy5zJpqAE9p3F4sTdb1hi8Q_biLvYETS2hNjaremJ95aYIgmE3dev8z2jHTYRR-np8lfvi2LwfrDgf4h5l3zomU05GrSJUHztuTaFuBIN8aCqbNeRpi_X5rTc8P8btsqc1_dGgz_jNJXnQgQWEjnOQuhVBh9ij5lEQAw9_rtO6mLoKu-njqrYlnHjJ2Is4HqM6G9n5nk3xURVPp2wKTOvnEO8EbrL9-b8q-IvyJOudWlZTjB_sSnd1Cz5UX_qC2krgUMrYSJirv5iW7fslZRUaj_PD4cVALoVQ)

CATHe (short for CATH embeddings) is a deep learning tool designed to detect remote homologues (up to 20% sequence similarity) for superfamilies in the CATH database. CATHe consists of an artificial neural network model which was trained on sequence embeddings from the ProtT5 protein Language Model (pLM). It was able to achieve an accuracy of 85.6% +- 0.4% (F1 score of 72%), and outperform the other baseline models derived from both, simple machine learning algorithms such as Logistic Regression, and homology-based inference using BLAST. 

CATHe2 is an improved version of CAThe with a different architecture, using embeddings from the ProstT5 pLM. CATHe2 is also able to take secondary and tertiary structure information as input as well as protein primary sequences, via 3Di sequences derived from PDB files. This allows CATHe2 to reach an accuracy of 92.2% (F1 score of 82.3%).

To know more about CATHe2, see [CATHe2 paper](https://...)

# Project information
This project was tested on Ubuntu 22.04

**Memory space requirements:**

To run inferences with the former version of CAThe, you need **20 GB** of free disk space.

To run inferences with the new version you need **24 GB** of free disk space.

To be able to test the training process and check the CATHe2 data you need **70 GB** of free disk space.


# Setup

## 1.1
- Change the working directory to the project root
- Download CATHe2 models and place them in the right folder by running

```bash
chmod +x ./CATHe2_setup.sh
./CATHe2_setup.sh
```

## 1.2
- Verify you are the root user running

```bash
whoami
```

- If you are not the root user (if whoami does not return “root”) add your user to the sudo list

```bash
usermod -aG sudo your-username
```

## 2.1
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

## 2.2
- And activate it

```bash
source venv_1/bin/activate

or

source venv_2/bin/activate
```

## 3.1
Put the protein sequences for which you want to predict the CATH annotation into a `FASTA` file named `Sequences.fasta`, in the `./src/cathe-predict` folder.

## 3.2
If you want to use both 3Di sequences and Amino Acid (AA) sequences to predict the CATH annotation, ensure that the corresponding PDB files for the sequences in `Sequences.fasta` are placed in the `PDB_folder` folder located in `./src/cathe-predict`. Each PDB file must be prefixed by the index of its respective sequence in `Sequences.fasta`, followed by an underscore. For example:

If my `Sequences.fasta` has 3 sequences of protein domain in this order:  protein domain 3hhl, protein domain 4jkm, protein domain 3ddn, then the corresponding PDB files in `./src/cathe-predict/PDB_folder` should be renamed:

```
0_3hhl.pdb
1_4jkm.pdb
2_3ddn.pdb
```

## 4
Then you can launch the desired version of CATHe to predict CATH annotation.

- To use the old version (with model ProtT5 and input type AA only), run

```bash
python ./src/cathe-predict/cathe_predictions.py 
```

(venv_1 has to be activated)

- To use the new version of CATHe (for input type AA only)

run

```bash
python ./src/cathe-predict/cathe_predictions.py --model ProstT5 --input_type AA
```
(venv_2 has to be activated)

- To use the new version of CATHe  (with 3Di input as well as AA).

```bash
python ./src/cathe-predict/cathe_predictions.py --model ProstT5 --input_type AA+3Di
```
 (You need to fill `PDB_folder` accordingly to run this, see [paragraph 3.2](#32), make sure venv_2 is activated too)

# Data

The dataset used for training, optimizing, and testing CATHe2 was derived from the CATH database. The datasets, along with the weights for the CATHe2 artificial neural network as well as all the intermediary training files can be downloaded from Zenodo from this link: [Dataset](https://doi.org/10.5281/zenodo.14534966).
Or running the following code at the project root:

```bash
 wget https://zenodo.org/records/14568772/files/data.zip?download=1 -O data.zip
 unzip ./data.zip
 rm -f data.zip
 ```
 This is not necessary to run CATHe2 inferences.

 data.zip is 30 GB

# Pre-Print

If you found this work useful, please consider citing the following article:

```
@article {Name,
	author = {},
	title = {},
	elocation-id = {},
	year = {},
	doi = {},
	publisher = {},
	URL = {},
	journal = {}
}
```
