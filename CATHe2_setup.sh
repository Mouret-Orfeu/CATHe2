# Basic installations
apt update
apt install sudo
apt install wget
apt install zip unzip

# Dowloading saved models
wget https://zenodo.org/records/14568772/files/saved_models.zip?download=1 -O saved_models.zip
unzip ./saved_models.zip
rm -f saved_models.zip

# Moving the CATHe and CATHe2 models to the correct folder
cp -r ./saved_models/ann_ProstT5_full_2_blocks_dropout_0.1_layer_size_1024_pLDDT_0_support_threshold_0 ./src/cathe-predict/
cp -r ./saved_models/ann_ProstT5_full_2_blocks_dropout_0.3_layer_size_2048_pLDDT_24_support_threshold_0_AA+3Di ./src/cathe-predict/
cp ./saved_models/CATHe.h5 ./src/cathe-predict/

# create the folder ./src/cathe-predict/PDB_folder if not here
mkdir -p ./src/cathe-predict/PDB_folder
