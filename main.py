import subprocess
import time

def run_script(script_name, args=[]):
    """Execute python script with arguments."""
    cmd = ["python", script_name] + args
    print(f"\n{'='*20} EXECUTING: {script_name} {'='*20}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"\n[ERROR] Pipeline failed at: {script_name}")
        exit(1)

if __name__ == "__main__":
    print(f"{'='*60}")
    print(f"  DRAGON FRUIT DISEASE CLASSIFICATION - MAIN PIPELINE")
    print(f"{'='*60}")

    total_start = time.time()

    # ==========================================================================
    # STEP 1: DATA PREPROCESSING & ANALYSIS
    # ==========================================================================
    run_script("preprocess.py")

    # ==========================================================================
    # STEP 2: MODEL TRAINING (VGG16, RESNET50, MOBILENETV2, VIT)
    # ==========================================================================
    model_configs = [
        {"name": "vgg16", "epochs": 50, "lr": 0.001},
        {"name": "resnet50", "epochs": 50, "lr": 0.001},
        {"name": "mobilenetv2", "epochs": 50, "lr": 0.001},
        {"name": "vit", "epochs": 150, "lr": 0.0001}
    ]
    
    for config in model_configs:
        run_script("train.py", [
            "--model", config["name"], 
            "--epochs", str(config["epochs"]),
            "--lr", str(config["lr"])
        ])

    # ==========================================================================
    # STEP 3: VISUALIZATION & FINAL EVALUATION
    # ==========================================================================
    run_script("visualize_comparison.py")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    total_duration = (time.time() - total_start) / 60
    print(f"\n{'='*60}")
    print(f" PIPELINE COMPLETED SUCCESSFULLY")
    print(f" Total execution time: {total_duration:.2f} minutes")
    print(f"{'='*60}")
