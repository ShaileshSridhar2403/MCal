#!/usr/bin/env python3
"""
PhysioNet Challenge 2012 Data Processing Script

This script processes the raw PhysioNet Challenge 2012 data using the exact
preprocessing pipeline from XAI_Benchmark to create missingness-level files
compatible with the MCal tabular benchmark.

Usage:
    python process_physionet_data.py --input_file PhysionetChallenge2012-set-a.csv.gz
"""

import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path

from mcal.paths import DATA_ROOT


def save_data_by_missingness(df, output_dir):
    """
    Save data separately based on levels of missingness.

    Args:
    df (pandas.DataFrame): The dataset to be saved.
    output_dir (str): Directory to save the output files.
    """
    os.makedirs(output_dir, exist_ok=True)

    row_missing_percentage = df.isnull().sum(axis=1) / df.shape[1] * 100

    print(f"Missingness distribution:")
    print(f"  Min: {row_missing_percentage.min():.1f}%")
    print(f"  Max: {row_missing_percentage.max():.1f}%")
    print(f"  Mean: {row_missing_percentage.mean():.1f}%")
    print(f"  Median: {row_missing_percentage.median():.1f}%")

    total_saved = 0
    for i in range(0, 100, 10):
        level_name = f"{i:03d}_{i+10:03d}"
        if i == 90:
            mask = (row_missing_percentage > i) & (row_missing_percentage <= 100)
        else:
            mask = (row_missing_percentage > i) & (row_missing_percentage <= i+10)

        level_df = df[mask]
        if not level_df.empty:
            output_file = os.path.join(output_dir, f"missingness_{level_name}.csv.gz")
            level_df.to_csv(output_file, compression='gzip')
            print(f"Saved {level_df.shape[0]} rows to {output_file}")
            total_saved += level_df.shape[0]
        else:
            print(f"No data for missingness level {level_name}")

    print(f"\nTotal rows saved: {total_saved} / {len(df)}")


def delete_value(df, c, value=0):
    """Delete specific values from a parameter column."""
    idx = df['parameter'] == c
    idx = idx & (df['value'] == value)

    df.loc[idx, 'value'] = np.nan
    return df


def replace_value(df, c, value=np.nan, below=None, above=None):
    """Replace values in a parameter column based on conditions."""
    idx = df['parameter'] == c

    if below is not None:
        idx = idx & (df['value'] < below)

    if above is not None:
        idx = idx & (df['value'] > above)

    if 'function' in str(type(value)):
        # value replacement is a function of the input
        df.loc[idx, 'value'] = df.loc[idx, 'value'].apply(value)
    else:
        df.loc[idx, 'value'] = value

    return df


def process_physionet_from_txt_files(base_path, dataset='set-a'):
    """
    Process PhysioNet data from original TXT files.
    This is the exact processing from XAI_Benchmark.
    """
    print(f"Processing PhysioNet Challenge 2012 {dataset} from TXT files...")

    # Load all files into list of lists
    txt_all = list()
    txt_files = [f for f in os.listdir(os.path.join(base_path, dataset)) if f.endswith('.txt')]

    print(f"Found {len(txt_files)} TXT files to process...")

    for f in txt_files:
        with open(os.path.join(base_path, dataset, f), 'r') as fp:
            txt = fp.readlines()

        # get recordid to add as a column
        recordid = txt[1].rstrip('\n').split(',')[-1]
        txt = [t.rstrip('\n').split(',') + [int(recordid)] for t in txt]
        txt_all.extend(txt[1:])

    # convert to pandas dataframe
    df = pd.DataFrame(txt_all, columns=['time', 'parameter', 'value', 'recordid'])

    # extract static variables into a separate dataframe
    df_static = df.loc[df['time'] == '00:00', :].copy()

    # retain only one of the 6 static vars:
    static_vars = ['RecordID', 'Age', 'Gender', 'Height', 'ICUType', 'Weight']
    df_static = df_static.loc[df['parameter'].isin(static_vars)]

    # remove these from original df
    idxDrop = df_static.index
    df = df.loc[~df.index.isin(idxDrop), :]

    # to ensure there are no duplicates, group by recordid/parameter and take the last value
    df_static = df_static.groupby(['recordid', 'parameter'])[['value']].last()
    df_static.reset_index(inplace=True)

    # pivot on parameter so there is one column per parameter
    df_static = df_static.pivot(index='recordid', columns='parameter', values='value')

    # some conversions on columns for convenience
    df['value'] = pd.to_numeric(df['value'], errors='raise')
    df['time'] = df['time'].map(lambda x: int(x.split(':')[0])*60 + int(x.split(':')[1]))

    return df, df_static


def process_static_variables(df_static):
    """Process static variables with cleaning and preprocessing."""
    print("Processing static variables...")

    # convert static into numeric
    for c in df_static.columns:
        df_static[c] = pd.to_numeric(df_static[c])

    # preprocess
    for c in df_static.columns:
        x = df_static[c]
        if c == 'Age':
            # replace anon ages with 91.4
            idx = x > 130
            df_static.loc[idx, c] = 91.4
        elif c == 'Gender':
            idx = x < 0
            df_static.loc[idx, c] = np.nan
        elif c == 'Height':
            idx = x < 0
            df_static.loc[idx, c] = np.nan

            # fix incorrectly recorded heights
            # 1.8 -> 180
            idx = x < 10
            df_static.loc[idx, c] = df_static.loc[idx, c] * 100

            # 18 -> 180
            idx = x < 25
            df_static.loc[idx, c] = df_static.loc[idx, c] * 10

            # 81.8 -> 180 (inch -> cm)
            idx = x < 100
            df_static.loc[idx, c] = df_static.loc[idx, c] * 2.2

            # 1800 -> 180
            idx = x > 1000
            df_static.loc[idx, c] = df_static.loc[idx, c] * 0.1

            # 400 -> 157
            idx = x > 250
            df_static.loc[idx, c] = df_static.loc[idx, c] * 0.3937

        elif c == 'Weight':
            idx = x < 35
            df_static.loc[idx, c] = np.nan

            idx = x > 299
            df_static.loc[idx, c] = np.nan


def process_time_series_data(df):
    """Process time series data with cleaning and validation."""
    print("Processing time series data...")

    df = delete_value(df, 'DiasABP', -1)
    df = replace_value(df, 'DiasABP', value=np.nan, below=1)
    df = replace_value(df, 'DiasABP', value=np.nan, above=200)
    df = replace_value(df, 'SysABP', value=np.nan, below=1)
    df = replace_value(df, 'MAP', value=np.nan, below=1)

    df = replace_value(df, 'NIDiasABP', value=np.nan, below=1)
    df = replace_value(df, 'NISysABP', value=np.nan, below=1)
    df = replace_value(df, 'NIMAP', value=np.nan, below=1)

    df = replace_value(df, 'HR', value=np.nan, below=1)
    df = replace_value(df, 'HR', value=np.nan, above=299)

    df = replace_value(df, 'PaCO2', value=np.nan, below=1)
    df = replace_value(df, 'PaCO2', value=lambda x: x*10, below=10)

    df = replace_value(df, 'PaO2', value=np.nan, below=1)
    df = replace_value(df, 'PaO2', value=lambda x: x*10, below=20)

    # the order of these steps matters
    df = replace_value(df, 'pH', value=lambda x: x*10, below=0.8, above=0.65)
    df = replace_value(df, 'pH', value=lambda x: x*0.1, below=80, above=65)
    df = replace_value(df, 'pH', value=lambda x: x*0.01, below=800, above=650)
    df = replace_value(df, 'pH', value=np.nan, below=6.5)
    df = replace_value(df, 'pH', value=np.nan, above=8.0)

    # convert to farenheit
    df = replace_value(df, 'Temp', value=lambda x: x*9/5+32, below=10, above=1)
    df = replace_value(df, 'Temp', value=lambda x: (x-32)*5/9, below=113, above=95)

    df = replace_value(df, 'Temp', value=np.nan, below=25)
    df = replace_value(df, 'Temp', value=np.nan, above=45)

    df = replace_value(df, 'RespRate', value=np.nan, below=1)
    df = replace_value(df, 'WBC', value=np.nan, below=1)

    df = replace_value(df, 'Weight', value=np.nan, below=35)
    df = replace_value(df, 'Weight', value=np.nan, above=299)

    return df


def create_feature_matrix(df, df_static):
    """Create feature matrix with aggregated time series features."""
    print("Creating feature matrix...")

    # Initialize a dataframe with df_static
    X = df_static.copy()
    X.drop('RecordID', axis=1, inplace=True)

    # MICU is ICUType==3, and is used as the reference category
    X['CCU'] = (X['ICUType'] == 1).astype(int)
    X['CSRU'] = (X['ICUType'] == 2).astype(int)
    X['SICU'] = (X['ICUType'] == 4).astype(int)
    X.drop('ICUType', axis=1, inplace=True)

    # For the following features we extract: first, last, lowest, highest, median
    feats = ['DiasABP', 'GCS', 'Glucose', 'HR', 'MAP',
             'NIDiasABP', 'NIMAP', 'NISysABP',
             'RespRate', 'SaO2', 'Temp']

    idx = df['parameter'].isin(feats)
    df_tmp = df.loc[idx, :].copy()
    df_tmp = df_tmp.groupby(['recordid', 'parameter'])['value']

    for agg in ['first', 'last', 'lowest', 'highest', 'median']:
        if agg == 'first':
            X_add = df_tmp.first()
        elif agg == 'last':
            X_add = df_tmp.last()
        elif agg == 'lowest':
            X_add = df_tmp.min()
        elif agg == 'highest':
            X_add = df_tmp.max()
        elif agg == 'median':
            X_add = df_tmp.median()
        else:
            print('Unrecognized aggregation {}. Skipping.'.format(agg))

        X_add = X_add.reset_index()
        X_add = X_add.pivot(index='recordid', columns='parameter', values='value')
        X_add.columns = [x + '_' + agg for x in X_add.columns]

        X = X.merge(X_add, how='left', left_index=True, right_index=True)

    # For the following features we extract: first, last
    feats = ['Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN', 'Cholesterol',
             'Creatinine', 'FiO2', 'HCO3', 'HCT', 'K', 'Lactate', 'Mg', 'Na',
             'PaCO2', 'PaO2', 'pH', 'Platelets', 'SysABP', 'TroponinI', 'TroponinT',
             'WBC', 'Weight']

    idx = df['parameter'].isin(feats)
    df_tmp = df.loc[idx, :].copy()
    df_tmp = df_tmp.groupby(['recordid', 'parameter'])['value']

    for agg in ['first', 'last']:
        if agg == 'first':
            X_add = df_tmp.first()
        elif agg == 'last':
            X_add = df_tmp.last()
        else:
            print('Unrecognized aggregation {}. Skipping.'.format(agg))

        X_add = X_add.reset_index()
        X_add = X_add.pivot(index='recordid', columns='parameter', values='value')
        X_add.columns = [x + '_' + agg for x in X_add.columns]

        X = X.merge(X_add, how='left', left_index=True, right_index=True)

    # For MechVent we extract custom data
    idx = df['parameter'] == 'MechVent'
    if idx.any():
        df_tmp = df.loc[idx, :].copy().groupby('recordid')

        X0 = df_tmp[['time']].min()
        X0.columns = ['MechVentStartTime']

        X1 = df_tmp[['time']].max()
        X1.columns = ['MechVentEndTime']

        X_add = X0.merge(X1, how='inner', left_index=True, right_index=True)
        X_add['MechVentDuration'] = X_add['MechVentEndTime'] - X_add['MechVentStartTime']

        X_add['MechVentLast8Hour'] = (X_add['MechVentEndTime'] >= 2400).astype(int)
        X_add.drop('MechVentEndTime', axis=1, inplace=True)

        X = X.merge(X_add, how='left', left_index=True, right_index=True)

    # Urine output
    idx = df['parameter'] == 'Urine'
    if idx.any():
        df_tmp = df.loc[idx, :].copy().groupby('recordid')

        X_add = df_tmp[['value']].sum()
        X_add.columns = ['UrineOutputSum']

        X = X.merge(X_add, how='left', left_index=True, right_index=True)

    print(f"Feature matrix shape: {X.shape}")
    return X


def process_physionet_from_csv(input_file):
    """
    Process PhysioNet data from preprocessed CSV file.
    """
    print(f"Loading preprocessed PhysioNet data from {input_file}...")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Load the preprocessed data
    X = pd.read_csv(input_file, index_col=0)
    print(f"Loaded data shape: {X.shape}")
    print(f"Columns: {list(X.columns)}")

    return X


def process_physionet_data(input_path, output_dir=str(DATA_ROOT / "tabular" / "missingness_levels"), from_txt=False, dataset='set-a'):
    """
    Main processing function for PhysioNet Challenge 2012 data.

    Args:
        input_path (str): Path to input data (CSV file or directory with TXT files)
        output_dir (str): Output directory for missingness-level files
        from_txt (bool): If True, process from original TXT files
        dataset (str): Dataset name ('set-a' or 'set-b')
    """
    print("=" * 60)
    print("PhysioNet Challenge 2012 Data Processing")
    print("=" * 60)

    if from_txt:
        # Process from original TXT files (exact XAI_Benchmark pipeline)
        df, df_static = process_physionet_from_txt_files(input_path, dataset)

        # Process static variables
        process_static_variables(df_static)

        # Process time series data
        df = process_time_series_data(df)

        # Create feature matrix
        X = create_feature_matrix(df, df_static)

        # Load outcomes
        if dataset == 'set-a':
            y = pd.read_csv(os.path.join(input_path, 'Outcomes-a.txt'))
        elif dataset == 'set-b':
            y = pd.read_csv(os.path.join(input_path, 'Outcomes-b.txt'))

        y.set_index('RecordID', inplace=True)
        y.index.name = 'recordid'
        X = y.merge(X, how='inner', left_index=True, right_index=True)

        # Save the processed CSV
        output_csv = f'PhysionetChallenge2012-{dataset}.csv.gz'
        X.to_csv(output_csv, sep=',', index=True)
        print(f"Processed data saved to: {output_csv}")

    else:
        # Process from preprocessed CSV file
        X = process_physionet_from_csv(input_path)

    print(f"\nFinal dataset shape: {X.shape}")
    print(f"Columns: {list(X.columns)}")

    # Check for target column
    target_cols = ['In-hospital_death', 'SAPS-I', 'SOFA', 'Length_of_stay', 'Survival', 'In-hospital_death']
    found_targets = [col for col in target_cols if col in X.columns]
    print(f"Found target columns: {found_targets}")

    # Save data by missingness levels
    print(f"\nSaving data by missingness levels to: {output_dir}")
    save_data_by_missingness(X, output_dir)

    print(f"\n✅ PhysioNet data processing completed!")
    print(f"Data saved in '{output_dir}' directory with missingness-level files.")

    return X


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Process PhysioNet Challenge 2012 data")
    parser.add_argument("--input_path", type=str, required=True,
                       help="Path to input data (CSV file or directory with TXT files)")
    parser.add_argument("--output_dir", type=str, default=str(DATA_ROOT / "tabular" / "missingness_levels"),
                       help="Output directory for missingness-level files")
    parser.add_argument("--from_txt", action="store_true",
                       help="Process from original TXT files (requires TXT directory)")
    parser.add_argument("--dataset", type=str, default="set-a", choices=['set-a', 'set-b'],
                       help="Dataset name (only used with --from_txt)")

    args = parser.parse_args()

    # Validate input
    if args.from_txt:
        if not os.path.isdir(args.input_path):
            raise ValueError("When using --from_txt, input_path must be a directory containing TXT files")
    else:
        if not os.path.isfile(args.input_path):
            raise ValueError("When not using --from_txt, input_path must be a CSV file")

    # Process the data
    try:
        X = process_physionet_data(
            input_path=args.input_path,
            output_dir=args.output_dir,
            from_txt=args.from_txt,
            dataset=args.dataset
        )

        print(f"\n🎉 Processing completed successfully!")
        print(f"Generated {len(os.listdir(args.output_dir))} missingness-level files")

    except Exception as e:
        print(f"\n❌ Processing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)