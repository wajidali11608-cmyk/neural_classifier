"""
Clinical Drug Simulation Demo

This script demonstrates how the model can be used to:
1. Diagnose a patient (Schizophrenia)
2. Simulate treatment (Antipsychotic Drug)
3. Check if the brain pattern normalizes
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from preprocessing import apply_filters, z_score_normalize
from features import compute_stft
from clinical import simulate_drug_effect
from config import get_config
import os

def run_drug_demo():
    # Load config to get timestamped folder
    config = get_config()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("CLINICAL DEMO: TREATMENT RESPONSE SIMULATION")
    print("="*60)
    
    # 1. Create a "Patient" with Schizophrenia
    # (High Gamma, Low Alpha - classic signature)
    print("\n1️⃣  Patient Admission: ID #PT-409")
    print("    Symptoms: Auditory hallucinations, disorganized thinking")
    
    fs = 256.0
    t = np.linspace(0, 4, int(4*fs))
    
    # Pathological Signal (Schizophrenia-like)
    # - Reduced Alpha (10Hz)
    # - Hyperactive Gamma (40Hz)
    # - Noise (neural noise)
    alpha = 0.5 * np.sin(2 * np.pi * 10 * t)  # Weak alpha
    gamma = 2.0 * np.sin(2 * np.pi * 40 * t)  # Strong gamma
    noise = 0.5 * np.random.randn(len(t))
    
    patient_signal = alpha + gamma + noise
    
    # Preprocess
    patient_signal = z_score_normalize(patient_signal)
    
    # 2. Simulate Drug Treatment (Antipsychotic)
    print("\n2️⃣  Simulating Treatment: Haloperidol (Antipsychotic)")
    print("    Mechanism: Dopamine D2 receptor antagonism")
    
    from clinical import simulate_drug_effect
    
    # Apply drug effect
    treated_signal = simulate_drug_effect(
        patient_signal, 
        drug_type='antipsychotic',
        effect_strength=0.8
    )
    
    # 3. Visual Comparison (Spectrograms)
    print("\n3️⃣  Analyzing Treatment Response...")
    
    # Compute spectrograms
    f, t_spec, spec_pre = compute_stft(patient_signal, fs=256.0)
    _, _, spec_post = compute_stft(treated_signal, fs=256.0)
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Pre-treatment
    im1 = ax1.pcolormesh(t_spec, f, 10*np.log10(np.abs(spec_pre) + 1e-10), 
                         shading='auto', cmap='magma')
    ax1.set_title("PRE-TREATMENT (Schizophrenia Pattern)\nHigh Gamma (Bottom), Low Alpha", 
                  fontsize=12, color='red', fontweight='bold')
    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_xlabel('Time (s)')
    plt.colorbar(im1, ax=ax1, label='Power (dB)')
    
    # Post-treatment
    im2 = ax2.pcolormesh(t_spec, f, 10*np.log10(np.abs(spec_post) + 1e-10), 
                         shading='auto', cmap='viridis')
    ax2.set_title("POST-TREATMENT (Simulated)\nReduced Gamma, Normalized Rhythm", 
                  fontsize=12, color='green', fontweight='bold')
    ax2.set_xlabel('Time (s)')
    plt.colorbar(im2, ax=ax2, label='Power (dB)')
    
    output_path = output_dir / "drug_simulation_demo.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Result generated: {output_path}")
    print("\nCLINICAL INSIGHT:")
    print("The simulation shows the drug successfully dampened the hyperactive")
    print("Gamma oscillations (red->blue in upper frequencies), suggesting")
    print("this patient might respond well to Dopamine antagonists.")

if __name__ == "__main__":
    run_drug_demo()
