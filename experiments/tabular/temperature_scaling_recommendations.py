#!/usr/bin/env python3
"""
Temperature Scaling Hyperparameter Recommendations

Based on hyperparameter tuning results, this script provides optimized configurations
for temperature scaling calibration.
"""

import json
import numpy as np
from pathlib import Path

def analyze_tuning_results(results_file):
    """Analyze tuning results and provide recommendations."""

    with open(results_file, 'r') as f:
        results = json.load(f)

    print("="*80)
    print("TEMPERATURE SCALING HYPERPARAMETER ANALYSIS")
    print("="*80)

    # Learning Rate Analysis
    print("\n1. LEARNING RATE ANALYSIS")
    print("-" * 40)
    lr_results = results['learning_rates']
    print("Results summary:")
    for result in lr_results:
        if 'error' not in result:
            print(f"  LR {result['lr']:8.3f}: ECE={result['ece']:.6f}, Temp={result['final_temperature']:.4f}, Steps={result['converged_steps']}")
        else:
            print(f"  LR {result['lr']:8.3f}: FAILED - {result.get('error', 'Unknown error')}")

    valid_lr = [r for r in lr_results if 'error' not in r and r['ece'] != float('inf')]
    best_lr = min(valid_lr, key=lambda x: x['ece'])

    print(f"\n✓ RECOMMENDATION: Use learning rate = {best_lr['lr']}")
    print(f"  - Achieves ECE = {best_lr['ece']:.6f}")
    print(f"  - Converges in {best_lr['converged_steps']} steps")
    print(f"  - Final temperature = {best_lr['final_temperature']:.4f}")

    if best_lr['lr'] >= 1.0:
        print("  ⚠️  WARNING: High learning rates (≥1.0) converge faster but may be unstable")
    if best_lr['lr'] <= 0.001:
        print("  ⚠️  WARNING: Very low learning rates may require more steps to converge")

    # Max Steps Analysis
    print("\n2. MAX STEPS ANALYSIS")
    print("-" * 40)
    steps_results = results['max_steps']
    print("Results summary:")
    for result in steps_results:
        if 'error' not in result:
            print(f"  Steps {result['max_steps']:5d}: ECE={result['ece']:.6f}, Temp={result['final_temperature']:.4f}, Used={result['actual_steps']}")
        else:
            print(f"  Steps {result['max_steps']:5d}: FAILED - {result.get('error', 'Unknown error')}")

    valid_steps = [r for r in steps_results if 'error' not in r and r['ece'] != float('inf')]
    best_steps = min(valid_steps, key=lambda x: x['ece'])

    # Find minimum steps that achieve near-optimal ECE (within 1% of best)
    best_ece = best_steps['ece']
    threshold_ece = best_ece * 1.01  # Allow 1% degradation
    efficient_configs = [r for r in valid_steps if r['ece'] <= threshold_ece]
    most_efficient = min(efficient_configs, key=lambda x: x['max_steps'])

    print(f"\n✓ RECOMMENDATION: Use max_steps = {most_efficient['max_steps']}")
    print(f"  - Achieves ECE = {most_efficient['ece']:.6f} (within 1% of optimal)")
    print(f"  - More efficient than {best_steps['max_steps']} steps")
    print(f"  - Actual convergence typically in {most_efficient['actual_steps']} steps")

    # Temperature Initialization Analysis
    print("\n3. TEMPERATURE INITIALIZATION ANALYSIS")
    print("-" * 40)
    temp_results = results['temp_init']
    print("Results summary:")
    for result in temp_results:
        if 'error' not in result:
            print(f"  Init {result['temp_init']:4.1f}: ECE={result['ece']:.6f}, Final={result['final_temperature']:.4f}, Steps={result['convergence_steps']}")
        else:
            print(f"  Init {result['temp_init']:4.1f}: FAILED - {result.get('error', 'Unknown error')}")

    valid_temp = [r for r in temp_results if 'error' not in r and r['ece'] != float('inf')]
    best_temp_init = min(valid_temp, key=lambda x: x['ece'])
    fastest_temp_init = min(valid_temp, key=lambda x: x['convergence_steps'])

    print(f"\n✓ RECOMMENDATION: Use temperature initialization = {best_temp_init['temp_init']}")
    print(f"  - Achieves ECE = {best_temp_init['ece']:.6f}")
    print(f"  - Converges in {best_temp_init['convergence_steps']} steps")

    if fastest_temp_init['temp_init'] != best_temp_init['temp_init']:
        print(f"\n💡 ALTERNATIVE: For fastest convergence, use init = {fastest_temp_init['temp_init']}")
        print(f"  - Converges in only {fastest_temp_init['convergence_steps']} steps")
        print(f"  - ECE = {fastest_temp_init['ece']:.6f}")

    # Stability Analysis
    print("\n4. STABILITY ANALYSIS")
    print("-" * 40)

    # Check for problematic configurations
    problems = []

    # Negative temperatures indicate optimization issues
    negative_temps = [r for r in lr_results if 'final_temperature' in r and r['final_temperature'] < 0]
    if negative_temps:
        problems.append(f"Learning rates {[r['lr'] for r in negative_temps]} resulted in negative temperatures")

    # Very high temperatures indicate poor calibration
    high_temps = [r for r in valid_lr + valid_steps + valid_temp if r['final_temperature'] > 5.0]
    if high_temps:
        problems.append(f"Some configurations resulted in very high temperatures (>5.0), indicating poor calibration")

    # Configurations that didn't converge
    non_converged = []
    for r in valid_steps:
        if r['actual_steps'] == r['max_steps'] and r['final_loss'] > 1e-6:
            non_converged.append(r['max_steps'])

    if non_converged:
        problems.append(f"Steps {non_converged} may not have fully converged (final loss > 1e-6)")

    if problems:
        print("⚠️  STABILITY ISSUES DETECTED:")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print("✅ No stability issues detected")

    # Final Recommendations
    print("\n5. FINAL RECOMMENDATIONS")
    print("-" * 40)

    # Conservative recommendation (most stable)
    conservative_lr = 0.01  # Safe middle ground
    conservative_steps = most_efficient['max_steps']
    conservative_init = 1.0  # Standard initialization

    # Aggressive recommendation (fastest convergence)
    aggressive_lr = best_lr['lr'] if best_lr['lr'] <= 1.0 else 1.0
    aggressive_steps = min(500, most_efficient['max_steps'])  # Don't go below 500 for safety
    aggressive_init = fastest_temp_init['temp_init']

    print(f"🛡️  CONSERVATIVE (Most Stable):")
    print(f"   - Learning rate: {conservative_lr}")
    print(f"   - Max steps: {conservative_steps}")
    print(f"   - Temperature init: {conservative_init}")
    print(f"   - Expected ECE: ~{best_lr['ece']:.6f}")

    print(f"\n🚀 AGGRESSIVE (Fastest):")
    print(f"   - Learning rate: {aggressive_lr}")
    print(f"   - Max steps: {aggressive_steps}")
    print(f"   - Temperature init: {aggressive_init}")
    print(f"   - Expected ECE: ~{fastest_temp_init['ece']:.6f}")
    print(f"   - Expected convergence: ~{fastest_temp_init['convergence_steps']} steps")

    print(f"\n🎯 OPTIMAL (Best ECE):")
    print(f"   - Learning rate: {best_lr['lr']}")
    print(f"   - Max steps: {best_steps['max_steps']}")
    print(f"   - Temperature init: {best_temp_init['temp_init']}")
    print(f"   - Expected ECE: {best_temp_init['ece']:.6f}")

    return {
        'conservative': {
            'lr': conservative_lr,
            'max_steps': conservative_steps,
            'temp_init': conservative_init
        },
        'aggressive': {
            'lr': aggressive_lr,
            'max_steps': aggressive_steps,
            'temp_init': aggressive_init
        },
        'optimal': {
            'lr': best_lr['lr'],
            'max_steps': best_steps['max_steps'],
            'temp_init': best_temp_init['temp_init']
        }
    }


def generate_optimized_temperature_function():
    """Generate an optimized temperature scaling function with recommended hyperparameters."""

    code = '''
def apply_optimized_temperature_calibrator(outputs, labels, device, preset="conservative", **kwargs):
    """
    Apply temperature scaling calibrator with optimized hyperparameters.

    Args:
        outputs: Model outputs (n_fractions, n_samples, n_classes)
        labels: True labels
        device: Computing device
        preset: "conservative", "aggressive", or "optimal"
        **kwargs: Override specific parameters (lr, max_steps, temp_init)

    Returns:
        Calibrated outputs
    """
    from calibrators.temperature import TemperatureScaling
    import torch
    import numpy as np
    from tqdm import tqdm

    # Optimized hyperparameter presets
    presets = {
        "conservative": {"lr": 0.01, "max_steps": 1000, "temp_init": 1.0},
        "aggressive": {"lr": 1.0, "max_steps": 500, "temp_init": 0.1},
        "optimal": {"lr": 1.0, "max_steps": 2000, "temp_init": 0.1}
    }

    # Get preset and override with kwargs
    config = presets.get(preset, presets["conservative"]).copy()
    config.update(kwargs)

    n_fractions, n_samples, n_classes = outputs.shape
    transformed_outputs = np.zeros_like(outputs)

    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)

    # Fit on fraction 0 (unablated)
    unablated_probs = torch.tensor(outputs[0], dtype=torch.float32, device=device)
    calibrator = TemperatureScaling(num_classes=n_classes)
    calibrator.to(device)

    # Initialize temperature
    with torch.no_grad():
        calibrator.temperature.fill_(config['temp_init'])

    # Fit with optimized parameters
    calibrator.fit(
        ablated_probs=unablated_probs,
        labels=labels_tensor,
        max_steps=config['max_steps'],
        lr=config['lr'],
        verbose=False
    )

    # Apply to all fractions
    for fraction in tqdm(range(n_fractions), desc="Applying optimized temperature calibrator"):
        ablated_probs = torch.tensor(outputs[fraction], dtype=torch.float32, device=device)
        calibrated_probs = calibrator.forward(ablated_probs)
        transformed_outputs[fraction] = calibrated_probs.detach().cpu().numpy()

    return transformed_outputs
'''

    return code


if __name__ == "__main__":
    results_file = Path("temperature_tuning_results/temperature_tuning_results.json")

    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        print("Please run quick_temperature_tuning.py first")
        exit(1)

    # Analyze results
    recommendations = analyze_tuning_results(results_file)

    # Generate optimized function
    print("\n" + "="*80)
    print("OPTIMIZED TEMPERATURE SCALING FUNCTION")
    print("="*80)
    print("You can use this optimized function in your benchmark scripts:")
    print()
    print(generate_optimized_temperature_function())

    # Save recommendations
    recommendations_file = results_file.parent / "temperature_scaling_recommendations.json"
    with open(recommendations_file, 'w') as f:
        json.dump(recommendations, f, indent=2)

    print(f"\n📄 Recommendations saved to: {recommendations_file}")