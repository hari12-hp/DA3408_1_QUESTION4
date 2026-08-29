# DA3408_1_QUESTION4

PARTNER A:
 for this question i used the UCI Human Activity Recognition (UCI-HAR) dataset. Dataset metadata is tracked by DVC, while the remote data is stored in Amazon S3.

Repo structure:

DA3408_1_QUESTION4/
├── src/
│   └── train.py
├── data.dvc
├── .dvc/config
├── environment.yml
├── requirements.txt
└── results.json

Environment Setup:
ensure conda or mamba is downloaded.





Example recorded results

Architecture

Training Accuracy

Test Accuracy

(64, 32, 16)

0.99928

0.94062

Recorded experiment

0.99252

0.94096



A fresh machine can reproduce the project with:
1. Go to Terminal and run below commands:
   
git clone https://github.com/hari12-hp/DA3408_1_QUESTION4.git
cd DA3408_1_QUESTION4

mamba env create -f environment.yml
mamba activate aiops-q4

dvc pull

Start MLflow in one terminal:

mlflow server --host 127.0.0.1 --port 5000

Then train in another tab ensuring the mamba environment is present in this tab too:

run:

python src/train.py


from here partner B will write his results and share his part.
