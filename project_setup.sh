kaggle competitions download -c allstate-claims-severity
mkdir artifacts
mkdir data
unzip allstate-claims-severity.zip -d data
rm allstate-claims-severity.zip
conda env create -f environment.yml --prefix envs python=3.6
conda activate claims_severity_pipeline_env
python main.py
