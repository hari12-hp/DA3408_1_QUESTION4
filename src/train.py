#IMPORTS

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from pathlib import Path
import subprocess
import json
from mlflow.tracking import MlflowClient

#automatic tracking

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def get_git_commit():
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT_ROOT,text=True).strip()
GIT_COMMIT = get_git_commit()


data_dir = (
    PROJECT_ROOT
    / "data"
    / "drsaeedmohsen"
    / "ucihar-dataset"
    / "versions"
    / "1"
    / "UCI-HAR Dataset"
)

# load dataset

x_train = np.loadtxt(data_dir/ "train" / "X_train.txt")
y_train = np.loadtxt(data_dir/ "train" / "y_train.txt")

x_test = np.loadtxt(data_dir/ "test" / "X_test.txt")
y_test = np.loadtxt(data_dir/ "test" / "y_test.txt")

#tracking in ML flow and training
best_accuracy=-1
best_model=None
best_architecture=None
import mlflow
import mlflow.sklearn
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("UCI-HAR-CLASSIFICATION")
architecture=[(100,50),(50,25),(64,32,16)]
seed=42

with mlflow.start_run(run_name='ARCHITECTURE') as parent:
    
    
        for arch in architecture:
            print(f'running the code with architecture={arch}')
        
            with mlflow.start_run(run_name=f'architecture-{arch}',nested=True):
                
                mlflow.set_tag('git_commit',GIT_COMMIT)
                mlflow.log_param('hidden layer ',str(arch))
                mlflow.log_param('seed',seed)
                mlp=MLPClassifier(  hidden_layer_sizes=arch, 
                max_iter=100, 
                alpha=1e-4,
                solver='adam', 
                verbose=False, 
                random_state=seed)
                
                mlp.fit(x_train,y_train)
                
                
                train_pred=mlp.predict(x_train)
                
                train_acc=accuracy_score(y_train,train_pred)
                
                y_pred=mlp.predict(x_test)
                
                test_acc=accuracy_score(y_test,y_pred)
                
                if test_acc>best_accuracy:
                    best_accuracy=test_acc
                    best_model=mlp
                    best_architecture=arch
                    
                
                mlflow.log_metric('train accuracy',train_acc)
                
                mlflow.log_metric('accuracy',test_acc)
                results = {
                    "architecture": list(arch),
                    "seed": seed,
                    "train_accuracy": train_acc,
                    "test_accuracy": test_acc
                }

                results_path = PROJECT_ROOT / "results.json"

                with open(results_path, "w") as f:
                    json.dump(results, f, indent=4)

                mlflow.log_artifact(str(results_path))

           
        MODEL_NAME = "UCI-HAR-CLASSIFIER"

        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="UCI-HAR CLASSIFIER",
            registered_model_name=MODEL_NAME,
            serialization_format="cloudpickle"
        )

        client=MlflowClient()
        latest_version=client.get_latest_versions(MODEL_NAME)[0].version
        client.transition_model_version_stage(name=MODEL_NAME,version=latest_version,stage='Staging')

        print(f'model {MODEL_NAME} ',f'version {latest_version} ','transition to staging is done' )
