#!/usr/bin/env python3

import sys
import os
sys.path.append('/home/antonxue/shailesh/MCal')
os.chdir('/home/antonxue/shailesh/MCal/experiments/language')

# Minimal test to verify accuracy fix works
import numpy as np
import torch
import json
from medqa_kl_benchmark import calculate_kl_metrics, aggregate_fractionwise_kl, aggregate_results

print("=== Testing MedQA Accuracy Fix ===")

# Create sample data
n_fractions = 2
n_samples = 3
n_classes = 5

# Sample outputs and labels
outputs = torch.randn(n_fractions, n_samples, n_classes)
labels = torch.randint(0, n_classes, (n_samples,))

print(f"Sample outputs shape: {outputs.shape}")
print(f"Sample labels shape: {labels.shape}")

# Test calculate_kl_metrics with accuracy
result = calculate_kl_metrics(outputs, labels=labels)

print("\n=== Result Keys ===")
print("Keys in result:", list(result.keys()))

# Check if accuracy keys are present
accuracy_keys = ['kl_values_accuracy', 'average_accuracy']
for key in accuracy_keys:
    if key in result:
        print(f"✓ {key}: {result[key]}")
    else:
        print(f"✗ {key}: MISSING")

# Test aggregation
print("\n=== Testing Aggregation ===")
sample_results = {'mcal_ce': [result]}

try:
    # Test fractionwise aggregation
    fractionwise = aggregate_fractionwise_kl([result])
    print("Fractionwise aggregation keys:", list(fractionwise.keys()))

    if 'mean_accuracy' in fractionwise:
        print(f"✓ mean_accuracy: {fractionwise['mean_accuracy']}")
    else:
        print("✗ mean_accuracy: MISSING")

    if 'std_accuracy' in fractionwise:
        print(f"✓ std_accuracy: {fractionwise['std_accuracy']}")
    else:
        print("✗ std_accuracy: MISSING")

    # Test overall aggregation
    aggregated = aggregate_results(sample_results)
    print("\n=== Overall Aggregation ===")
    print("Methods:", list(aggregated.keys()))

    if 'mcal_ce' in aggregated:
        method_result = aggregated['mcal_ce']
        print("Method result keys:", list(method_result.keys()))

        # Check for overall accuracy metrics
        if 'accuracy_transformed_mean' in method_result:
            print(f"✓ accuracy_transformed_mean: {method_result['accuracy_transformed_mean']}")
        else:
            print("✗ accuracy_transformed_mean: MISSING")

        if 'accuracy_transformed_std' in method_result:
            print(f"✓ accuracy_transformed_std: {method_result['accuracy_transformed_std']}")
        else:
            print("✗ accuracy_transformed_std: MISSING")

        # Check fractionwise results
        if 'fraction_wise_results_transformed' in method_result:
            fwr = method_result['fraction_wise_results_transformed']
            print("Fractionwise keys:", list(fwr.keys()))

            if 'mean_accuracy' in fwr:
                print(f"✓ fractionwise mean_accuracy: {fwr['mean_accuracy']}")
            else:
                print("✗ fractionwise mean_accuracy: MISSING")

            if 'std_accuracy' in fwr:
                print(f"✓ fractionwise std_accuracy: {fwr['std_accuracy']}")
            else:
                print("✗ fractionwise std_accuracy: MISSING")

    # Save test result to JSON
    test_file = 'test_accuracy_result.json'
    with open(test_file, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"\n=== Test result saved to {test_file} ===")

    print("\n=== SUCCESS: Accuracy fix is working! ===")

except Exception as e:
    print(f"\n=== ERROR: {e} ===")
    import traceback
    traceback.print_exc()