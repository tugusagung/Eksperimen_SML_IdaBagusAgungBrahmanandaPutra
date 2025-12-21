"""
automate_IdaBagusAgungBrahmanandaPutra.py
Script untuk preprocessing otomatis data diabetes
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import argparse
import json
from datetime import datetime

# =========================================================
# PATH CONFIGURATION
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

DEFAULT_INPUT_PATH = os.path.join(PROJECT_DIR, "diabetes.csv")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "diabetes_preprocessed")

# =========================================================
# PREPROCESSING FUNCTION
# =========================================================
def preprocess_diabetes(
    input_path=DEFAULT_INPUT_PATH,
    output_dir=DEFAULT_OUTPUT_DIR,
    test_size=0.2,
    random_state=42
):
    print("=" * 70)
    print("AUTOMATED DIABETES DATA PREPROCESSING")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    start_time = datetime.now()

    # 1. LOAD DATA
    print("\n📥 1. LOADING DATA")
    print(f"Input path: {input_path}")

    if not os.path.exists(input_path):
        print(f"Error: File not found at {input_path}")
        return None

    try:
        df = pd.read_csv(input_path)
        print(f"Success: Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    # 2. DATA VALIDATION
    print("\n2. DATA VALIDATION")
    required_columns = [
        'Pregnancies', 'Glucose', 'BloodPressure',
        'SkinThickness', 'Insulin', 'BMI',
        'DiabetesPedigreeFunction', 'Age', 'Outcome'
    ]

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        return None
    print("All required columns present")

    # 3. HANDLE INVALID ZERO VALUES
    print("\n3. HANDLING INVALID ZERO VALUES")
    df_clean = df.copy()
    columns_with_zero_issue = [
        'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'
    ]

    zero_stats = {}
    for col in columns_with_zero_issue:
        zeros_before = (df_clean[col] == 0).sum()
        if zeros_before > 0:
            df_clean[col] = df_clean[col].replace(0, np.nan)
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)

            zero_stats[col] = {
                'zeros_before': int(zeros_before),
                'median_value': float(median_val)
            }

            print(f"   {col}: Replaced {zeros_before} zeros with median {median_val:.2f}")
        else:
            print(f"   {col}: No zeros found")

    # 4. REMOVE DUPLICATES
    print("\n4. REMOVING DUPLICATES")
    duplicates = df_clean.duplicated().sum()
    if duplicates > 0:
        df_clean = df_clean.drop_duplicates()
        print(f"   Removed {duplicates} duplicate rows")
    else:
        print("   No duplicates found")

    # 5. SPLIT FEATURES & TARGET
    print("\n5. SEPARATING FEATURES AND TARGET")
    X = df_clean.drop('Outcome', axis=1)
    y = df_clean['Outcome']

    # Class distribution
    class_counts = y.value_counts()
    print(f"   Class distribution: 0={class_counts.get(0,0)}, 1={class_counts.get(1,0)}")

    # 6. FEATURE SCALING
    print("\n6. FEATURE SCALING (StandardScaler)")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # 7. TRAIN TEST SPLIT
    print("\n7. TRAIN-TEST SPLIT")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # 8. SAVE OUTPUT
    print("\n8. SAVING PROCESSED DATA")
    os.makedirs(output_dir, exist_ok=True)

    # Save CSV
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    # Save scaler
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))

    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "input_file": input_path,
        "output_directory": output_dir,
        "rows_original": int(df.shape[0]),
        "rows_cleaned": int(df_clean.shape[0]),
        "test_size": test_size,
        "random_state": random_state,
        "zero_handling": zero_stats,
        "duplicates_removed": int(duplicates),
        "class_distribution": {
            "0": int(class_counts.get(0,0)),
            "1": int(class_counts.get(1,0))
        }
    }

    with open(os.path.join(output_dir, "preprocessing_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Save human-readable summary
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Diabetes Data Preprocessing Summary\n")
        f.write("="*40 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Execution time: {(datetime.now() - start_time).total_seconds():.2f} seconds\n\n")

        f.write("Data Statistics:\n")
        f.write(f"- Original rows: {df.shape[0]}\n")
        f.write(f"- Cleaned rows: {df_clean.shape[0]}\n")
        f.write(f"- Training set: {X_train.shape[0]} rows\n")
        f.write(f"- Test set: {X_test.shape[0]} rows\n\n")

        f.write("Class Distribution:\n")
        f.write(f"- 0: {class_counts.get(0,0)}\n")
        f.write(f"- 1: {class_counts.get(1,0)}\n\n")

        f.write("Zero Value Handling:\n")
        for col, stats in zero_stats.items():
            f.write(f"- {col}: Replaced {stats['zeros_before']} zeros with median {stats['median_value']:.2f}\n")

        if duplicates > 0:
            f.write(f"\nDuplicates Removed: {duplicates}\n")

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE! 🎉")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Execution time: {(datetime.now() - start_time).total_seconds():.2f} seconds")

    return X_train, X_test, y_train, y_test, scaler

# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Automated Diabetes Data Preprocessing"
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
        help="Path to diabetes.csv"
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42
    )

    args = parser.parse_args()

    result = preprocess_diabetes(
        input_path=args.input,
        output_dir=args.output,
        test_size=args.test_size,
        random_state=args.random_state
    )

    if result is None:
        print("\nPreprocessing failed!")
        exit(1)

if __name__ == "__main__":
    main()
