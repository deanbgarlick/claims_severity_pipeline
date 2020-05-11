kaggle competitions download -c allstate-claims-severity
mkdir data
unzip allstate-claims-severity.zip -d data
conda create --prefix ./envs
conda activate ./envs
pip install -r requirements.txt
conda env export > environment.yaml
python main.py