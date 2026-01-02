"""
Drug Response Simulation for Neural Signals

Simulates effects of psychiatric medications on neural activity patterns
for hypothesis testing and treatment response prediction.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from scipy.signal import butter, filtfilt


def simulate_drug_effect(
    signal: np.ndarray,
    drug_type: str,
    effect_strength: float = 0.5,
    sample_rate: float = 256.0
) -> np.ndarray:
    """
    Simulate the effect of a drug on neural signal.
    
    Args:
        signal: Input signal, shape (n_channels, n_timepoints)
        drug_type: Type of drug ('antipsychotic', 'mood_stabilizer', 'antidepressant')
        effect_strength: Effect magnitude [0, 1]
        sample_rate: Sampling rate in Hz
        
    Returns:
        Modified signal with simulated drug effect
    """
    simulators = {
        'antipsychotic': simulate_dopamine_modulation,
        'mood_stabilizer': simulate_lithium_effect,
        'antidepressant': simulate_ssri_effect
    }
    
    if drug_type not in simulators:
        raise ValueError(f"Unknown drug type: {drug_type}")
    
    return simulators[drug_type](signal, effect_strength, sample_rate)


def simulate_dopamine_modulation(
    signal: np.ndarray,
    effect_strength: float = 0.5,
    sample_rate: float = 256.0
) -> np.ndarray:
    """
    Simulate dopamine D2 antagonist effects (antipsychotics).
    
    Effects modeled:
    - Reduced gamma band activity (associated with positive symptoms)
    - Slight increase in alpha synchronization
    - Reduced burst activity
    
    Args:
        signal: Input neural signal
        effect_strength: Drug effect magnitude [0, 1]
        sample_rate: Sampling rate
        
    Returns:
        Modified signal
    """
    modified = signal.copy()
    
    if signal.ndim == 1:
        modified = modified.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False
    
    n_channels, n_timepoints = modified.shape
    
    # Attenuate gamma band (30-100 Hz) - associated with psychotic symptoms
    gamma_attenuation = 1.0 - (0.4 * effect_strength)  # Up to 40% reduction
    
    nyq = sample_rate / 2
    low = 30.0 / nyq
    high = min(100.0 / nyq, 0.99)
    
    if low < high:
        # Design bandpass for gamma
        b, a = butter(4, [low, high], btype='band')
        
        for ch in range(n_channels):
            gamma_component = filtfilt(b, a, modified[ch])
            modified[ch] = modified[ch] - gamma_component * (1 - gamma_attenuation)
    
    # Enhance alpha band (8-13 Hz) - mild effect
    alpha_enhancement = 1.0 + (0.15 * effect_strength)
    
    low_alpha = 8.0 / nyq
    high_alpha = min(13.0 / nyq, 0.99)
    
    if low_alpha < high_alpha:
        b, a = butter(4, [low_alpha, high_alpha], btype='band')
        
        for ch in range(n_channels):
            alpha_component = filtfilt(b, a, modified[ch])
            modified[ch] = modified[ch] + alpha_component * (alpha_enhancement - 1)
    
    # Reduce amplitude variability (burst suppression)
    std_reduction = 1.0 - (0.2 * effect_strength)
    mean_signal = np.mean(modified, axis=1, keepdims=True)
    modified = mean_signal + (modified - mean_signal) * std_reduction
    
    if squeeze:
        modified = modified[0]
    
    return modified


def simulate_lithium_effect(
    signal: np.ndarray,
    effect_strength: float = 0.5,
    sample_rate: float = 256.0
) -> np.ndarray:
    """
    Simulate lithium effects (mood stabilizer).
    
    Effects modeled:
    - Stabilization of theta/beta ratio
    - Reduced extreme fluctuations
    - Enhanced low-frequency coherence
    
    Args:
        signal: Input neural signal
        effect_strength: Drug effect magnitude [0, 1]
        sample_rate: Sampling rate
        
    Returns:
        Modified signal
    """
    modified = signal.copy()
    
    if signal.ndim == 1:
        modified = modified.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False
    
    n_channels, n_timepoints = modified.shape
    
    # Reduce extreme amplitude fluctuations (mood stabilization)
    window_size = int(sample_rate * 0.5)  # 500ms windows
    
    for ch in range(n_channels):
        # Compute local variance
        for i in range(0, n_timepoints - window_size, window_size // 2):
            window = modified[ch, i:i+window_size]
            local_std = np.std(window)
            global_std = np.std(modified[ch])
            
            # If local variance is extreme, dampen it
            if local_std > 1.5 * global_std:
                damping = 1.0 - (0.3 * effect_strength)
                window_mean = np.mean(window)
                modified[ch, i:i+window_size] = window_mean + (window - window_mean) * damping
    
    # Slight reduction in beta power (reduces hyperactivity)
    nyq = sample_rate / 2
    low_beta = 13.0 / nyq
    high_beta = min(30.0 / nyq, 0.99)
    
    if low_beta < high_beta:
        b, a = butter(4, [low_beta, high_beta], btype='band')
        beta_reduction = 1.0 - (0.2 * effect_strength)
        
        for ch in range(n_channels):
            beta_component = filtfilt(b, a, modified[ch])
            modified[ch] = modified[ch] - beta_component * (1 - beta_reduction)
    
    if squeeze:
        modified = modified[0]
    
    return modified


def simulate_ssri_effect(
    signal: np.ndarray,
    effect_strength: float = 0.5,
    sample_rate: float = 256.0
) -> np.ndarray:
    """
    Simulate SSRI effects (antidepressants).
    
    Effects modeled:
    - Increased alpha power (relaxation/normalization)
    - Reduced theta asymmetry (frontal)
    - Enhanced connectivity in prefrontal regions
    
    Args:
        signal: Input neural signal
        effect_strength: Drug effect magnitude [0, 1]
        sample_rate: Sampling rate
        
    Returns:
        Modified signal
    """
    modified = signal.copy()
    
    if signal.ndim == 1:
        modified = modified.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False
    
    n_channels, n_timepoints = modified.shape
    
    nyq = sample_rate / 2
    
    # Enhance alpha (anxiolytic effect)
    low_alpha = 8.0 / nyq
    high_alpha = min(13.0 / nyq, 0.99)
    
    if low_alpha < high_alpha:
        b, a = butter(4, [low_alpha, high_alpha], btype='band')
        alpha_enhancement = 1.0 + (0.25 * effect_strength)
        
        for ch in range(n_channels):
            alpha_component = filtfilt(b, a, modified[ch])
            modified[ch] = modified[ch] + alpha_component * (alpha_enhancement - 1)
    
    # Reduce high theta (often elevated in depression)
    low_theta = 4.0 / nyq
    high_theta = 8.0 / nyq
    
    if low_theta < high_theta:
        b, a = butter(4, [low_theta, high_theta], btype='band')
        theta_reduction = 1.0 - (0.15 * effect_strength)
        
        for ch in range(n_channels):
            theta_component = filtfilt(b, a, modified[ch])
            modified[ch] = modified[ch] - theta_component * (1 - theta_reduction)
    
    # Add slight smoothing (calming effect)
    from scipy.ndimage import gaussian_filter1d
    smoothing_sigma = 1.0 * effect_strength
    
    for ch in range(n_channels):
        modified[ch] = gaussian_filter1d(modified[ch], sigma=smoothing_sigma)
    
    if squeeze:
        modified = modified[0]
    
    return modified


class DrugSimulator:
    """
    Comprehensive drug effect simulator for research purposes.
    
    Provides a unified interface for simulating various medication effects
    on neural signals.
    """
    
    # Drug effect profiles
    DRUG_PROFILES = {
        'haloperidol': {
            'type': 'antipsychotic',
            'primary_target': 'D2_receptor',
            'effects': {'gamma_reduction': 0.4, 'alpha_increase': 0.15}
        },
        'clozapine': {
            'type': 'antipsychotic',
            'primary_target': 'multi_receptor',
            'effects': {'gamma_reduction': 0.3, 'alpha_increase': 0.2}
        },
        'lithium': {
            'type': 'mood_stabilizer',
            'primary_target': 'multiple_pathways',
            'effects': {'variance_reduction': 0.3, 'beta_reduction': 0.2}
        },
        'valproate': {
            'type': 'mood_stabilizer',
            'primary_target': 'GABA_enhancement',
            'effects': {'variance_reduction': 0.25, 'beta_reduction': 0.15}
        },
        'fluoxetine': {
            'type': 'antidepressant',
            'primary_target': 'SERT',
            'effects': {'alpha_increase': 0.25, 'theta_reduction': 0.15}
        },
        'sertraline': {
            'type': 'antidepressant',
            'primary_target': 'SERT',
            'effects': {'alpha_increase': 0.2, 'theta_reduction': 0.15}
        }
    }
    
    def __init__(self, sample_rate: float = 256.0):
        self.sample_rate = sample_rate
        
    def apply_drug(
        self,
        signal: np.ndarray,
        drug_name: str,
        dose_factor: float = 1.0
    ) -> np.ndarray:
        """
        Apply drug effect to signal.
        
        Args:
            signal: Input signal
            drug_name: Name of drug
            dose_factor: Dose multiplier [0.5-2.0]
            
        Returns:
            Modified signal
        """
        if drug_name not in self.DRUG_PROFILES:
            available = list(self.DRUG_PROFILES.keys())
            raise ValueError(f"Unknown drug: {drug_name}. Available: {available}")
        
        profile = self.DRUG_PROFILES[drug_name]
        drug_type = profile['type']
        
        # Map to simulation function
        type_to_sim = {
            'antipsychotic': simulate_dopamine_modulation,
            'mood_stabilizer': simulate_lithium_effect,
            'antidepressant': simulate_ssri_effect
        }
        
        sim_func = type_to_sim[drug_type]
        effect_strength = min(1.0, 0.5 * dose_factor)
        
        return sim_func(signal, effect_strength, self.sample_rate)
    
    def simulate_treatment_response(
        self,
        baseline_signal: np.ndarray,
        drug_name: str,
        response_type: str = 'responder'
    ) -> np.ndarray:
        """
        Simulate treatment response over time.
        
        Args:
            baseline_signal: Pre-treatment signal
            drug_name: Drug to simulate
            response_type: 'responder', 'partial', or 'non_responder'
            
        Returns:
            Post-treatment signal
        """
        response_factors = {
            'responder': 1.0,
            'partial': 0.5,
            'non_responder': 0.1
        }
        
        factor = response_factors.get(response_type, 0.5)
        
        return self.apply_drug(baseline_signal, drug_name, dose_factor=factor)
    
    def get_available_drugs(self) -> Dict[str, Dict]:
        """Get information about available drugs."""
        return self.DRUG_PROFILES.copy()
    
    def predict_response_trajectory(
        self,
        baseline_signal: np.ndarray,
        drug_name: str,
        n_weeks: int = 8
    ) -> np.ndarray:
        """
        Simulate signal changes over treatment course.
        
        Returns:
            signals_over_time: Shape (n_weeks+1, *signal.shape)
        """
        trajectory = [baseline_signal.copy()]
        
        # Typical trajectory: gradual effect buildup
        for week in range(1, n_weeks + 1):
            # Effect builds up over first 4 weeks, then plateaus
            week_factor = min(week / 4.0, 1.0)  # Linear increase to week 4
            
            modified = self.apply_drug(
                baseline_signal,
                drug_name,
                dose_factor=week_factor
            )
            trajectory.append(modified)
        
        return np.array(trajectory)
