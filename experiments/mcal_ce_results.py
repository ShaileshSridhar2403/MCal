"""Save MCal_CE fit summaries to disk.

``MCal_CE.fit()`` records a summary of each fit in ``calibrator.fit_summary_``
and writes nothing itself. The benchmarks save one summary per ablation
fraction and then combine them into ``mcal_ce_combined_results_<id>.json``.
File names match the result files already in the repository.
"""

import glob
import json
import os
from pathlib import Path


def save_fit_summary(calibrator, results_dir):
    """Write ``calibrator.fit_summary_`` to ``results_dir`` and return the path.

    A summary from a fit with a ``fraction`` goes to
    ``temp_mcal_ce_fraction_<fraction>_<experiment_id>.json``; otherwise to
    ``mcal_ce_results_<head_type>_<num_classes>classes.json``.
    """
    summary = calibrator.fit_summary_
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    if "fraction" in summary:
        path = results_dir / f"temp_mcal_ce_fraction_{summary['fraction']}_{summary['experiment_id']}.json"
    else:
        path = results_dir / f"mcal_ce_results_{calibrator.head_type}_{calibrator.num_classes}classes.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    return path


def combine_fraction_results(results_dir, experiment_id="default", cleanup_temp_files=False):
    """Combine the per-fraction summaries in ``results_dir`` into one JSON file.

    Args:
        results_dir: Folder holding the ``temp_mcal_ce_fraction_*`` files
        experiment_id: Experiment identifier to match
        cleanup_temp_files: Delete the per-fraction files after combining

    Returns:
        Path to the combined file, or None if no per-fraction files were found
    """
    results_dir = Path(results_dir)
    temp_files = glob.glob(str(results_dir / f"temp_mcal_ce_fraction_*_{experiment_id}.json"))
    if not temp_files:
        print(f"No per-fraction MCal_CE results for experiment_id {experiment_id!r} in {results_dir}")
        return None

    combined_results = {'experiment_id': experiment_id, 'fractions': {}}
    for temp_file in sorted(temp_files):
        with open(temp_file) as f:
            fraction_data = json.load(f)
        fraction_num = fraction_data.pop('fraction')
        fraction_data.pop('experiment_id', None)
        combined_results['fractions'][str(fraction_num)] = fraction_data

    combined_filename = results_dir / f"mcal_ce_combined_results_{experiment_id}.json"
    with open(combined_filename, 'w') as f:
        json.dump(combined_results, f, indent=2)
    print(f"Combined MCal_CE results from {len(combined_results['fractions'])} fractions: {combined_filename}")

    if cleanup_temp_files:
        for temp_file in temp_files:
            os.remove(temp_file)
    return combined_filename
