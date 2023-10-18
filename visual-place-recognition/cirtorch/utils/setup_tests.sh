# download the dataset
mkdir -p ./data/test/ ./data/train/ ./data/networks/ ./data/networks/retrieval-SfM-120k_w_resnet101_gem/

wget -cO - https://cloudstor.aarnet.edu.au/plus/s/11IKe9VtmkdikIN/download > ./data/test/GardensPointWalking.zip
echo "Unzipping data..."
unzip -q ./data/test/GardensPointWalking.zip -d ./data/test/
rm ./data/test/GardensPointWalking.zip

echo "Creating GP data pickle (please set the conda environment or the dependencies first to avoid pickle encoding errors)"
python cirtorch/utils/create_pkl_gp.py

wget -cO - https://cloudstor.aarnet.edu.au/plus/s/kya3ZjF5N3WqLVL/download > ./data/models.zip
echo "Unzipping data..."
unzip -q ./data/models.zip -d ./data/
rm ./data/models.zip
