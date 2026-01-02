"""
NEURAL CLASSIFIER - FULL CAPABILITY SHOWCASE

This script runs the COMPLETE pipeline to demonstrate everything the model can do:
1. Load Patient Data
2. Train the Model
3. Diagnose Patients (Prediction)
4. Explain the Diagnosis (Grad-CAM)
5. Predict Treatment Response (Drug Simulation)
"""

import subprocess
import sys
import time

def run_command(cmd, description):
    print(f"\nExample: {description}")
    print(f"Running: {cmd}...")
    subprocess.run(cmd, shell=True, check=True)
    print("✅ Done!")

if __name__ == "__main__":
    print("="*60)
    print("NEURAL CLASSIFIER - THE COMPLETE SHOWCASE")
    print("="*60)
    
    # 1. Train the Model & Diagnose
    # This runs the main pipeline: Data Loading -> Preprocessing -> Feature Extraction -> Training -> Evaluation
    run_command(
        f"{sys.executable} main.py --data data/sample_data.csv --epochs 20 --model cnn", 
        "1. Training the AI on Patient Data & Generating Diagnoses"
    )
    
    # 2. Treatment Simulation
    # This simulates how a specific patient would respond to medication
    run_command(
        f"{sys.executable} drug_demo.py", 
        "2. Simulating Clinical Treatment (Digital Twin)"
    )

    print("\n" + "="*60)
    print("SHOWCASE COMPLETE")
    print("="*60)
    
    print("\n📦 ALL OUTPUTS ARE READY:")
    print("   1. Diagnosis Accuracy & Metrics -> outputs/metrics.json")
    print("   2. The Trained Brain Model      -> outputs/best_model.pt")
    print("   3. Training Performance         -> outputs/plots/training_history.png")
    print("   4. Confusion Matrix             -> outputs/plots/confusion_matrix.png")
    print("   5. ROC Curves (Sensitivity)     -> outputs/plots/roc_curves.png")
    print("   6. Why it made that diagnosis   -> outputs/plots/grad_cam.png (Explainability)")
    print("   7. Drug Response Prediction     -> outputs/drug_simulation_demo.png")
    
    print("\n🚀 You can now present these results!")
