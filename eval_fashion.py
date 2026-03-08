import subprocess
import json
output = subprocess.check_output(["python", "src/inference.py", "--model", "src/best_model.npy", "--config", "src/best_config.json", "--dataset", "fashion_mnist"], text=True)
print(output)
