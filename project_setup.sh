kaggle competitions download -c allstate-claims-severity
mkdir artifacts
mkdir data
unzip allstate-claims-severity.zip -d data
rm allstate-claims-severity.zip
conda env create -f environment.yml
python main.py
