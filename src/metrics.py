from pathlib import Path
import json
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.parent.absolute()
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

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
