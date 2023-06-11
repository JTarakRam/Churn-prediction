from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)
from config.config import ARTIFACTS_DIR

def load_metrics():
    with open(Path(ARTIFACTS_DIR, 'metrics.json'), 'r') as f:
        model_metrics = json.load(f)
        scores = model_metrics['scores']
        report = model_metrics['report']
        best_model_name = model_metrics['best_model_name']
 
    return scores, report, best_model_name 

scores, report, best_model_name = load_metrics()

print("Scores:")
print(scores)
print()

print("Classification Report:")
print(report)
print()

print("Best Model Name:")
print(best_model_name)
