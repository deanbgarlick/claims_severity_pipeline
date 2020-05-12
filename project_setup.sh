kaggle competitions download -c allstate-claims-severity
mkdir artifacts
mkdir data
unzip allstate-claims-severity.zip -d data
rm allstate-claims-severity.zip
conda create --prefix ./envs python=3.6 -y
conda activate ./envs
pip install --noinput -r requirements.txt -y
conda env export > environment.yaml
python main.py
