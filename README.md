# DA3408_1_QUESTION4

## PARTNER A

For this question, I used the **UCI Human Activity Recognition (UCI-HAR)** dataset.

The dataset metadata is tracked using **DVC**, while the remote data is stored in **Amazon S3**.

---

## Repository Structure

```text
DA3408_1_QUESTION4/
├── src/
│   └── train.py
├── data.dvc
├── .dvc/config
├── environment.yml
├── requirements.txt
└── results.json
```

---

## Environment Setup

Ensure that **Conda or Mamba** is installed before running the project.

A fresh machine can reproduce the environment using:

```bash
mamba env create -f environment.yml
mamba activate aiops-q4
```

---

## Example Recorded Results

| Architecture / Run | Training Accuracy | Test Accuracy |
|---|---:|---:|
| `(64, 32, 16)` | `0.99928` | `0.94062` |
| `(50,25)` | `0.99252` | `0.94096` |
|`(100,50)`|`0.99741`|`0.945707`|

The results were tracked using **MLflow**.

---

# Reproducing the Project

A fresh machine can reproduce the project by following these steps.

### 1. Open a terminal and clone the repository

```bash
git clone https://github.com/hari12-hp/DA3408_1_QUESTION4.git
cd DA3408_1_QUESTION4
```

### 2. Create and activate the environment

```bash
mamba env create -f environment.yml
mamba activate aiops-q4
```

### 3. Retrieve the dataset using DVC

```bash
dvc pull
```

---

## Start MLflow

Start the MLflow tracking server in one terminal:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

The MLflow interface will be available at:

```text
http://127.0.0.1:5000
```

---

## Train the Model

Open another terminal and ensure that the Mamba environment is activated:

```bash
mamba activate aiops-q4
```

Then run:

```bash
python src/train.py
```

---

# PARTNER B


**partner B, add  the results after independent reproduction.**
