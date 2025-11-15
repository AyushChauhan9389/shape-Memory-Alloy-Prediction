"""
Data Diagnostics Script - Analyze SMA Dataset
Purpose: Understand why the improved model failed with negative R²
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('dataset/Combined_SMA_Dataset_Filled.csv')

print("="*80)
print("SMA Dataset Diagnostic Report")
print("="*80)

print(f"\n1. Dataset Shape: {df.shape}")
print(f"   Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# Check all columns
print(f"\n2. All Columns ({len(df.columns)}):")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2d}. {col}")

# Element features
element_features = [
    'Ag (at.%)', 'Al (at.%)', 'Au (at.%)', 'Cd (at.%)', 'Co (at.%)',
    'Cu (at.%)', 'Fe (at.%)', 'Hf (at.%)', 'Mn (at.%)', 'Nb (at.%)',
    'Ni (at.%)', 'Pd (at.%)', 'Pt (at.%)', 'Ru (at.%)', 'Si (at.%)',
    'Ta (at.%)', 'Ti (at.%)', 'Zn (at.%)', 'Zr (at.%)'
]

# Process features
process_features = [
    'Cooling Rate (°C/min)',
    'Heating Rate (°C/min)',
    'Calculated Density (g/cm^3)'
]

# Temperature features
temp_features = [
    'Austenite Finish Temperature - AF - (°C)',
    'Austenite Start Temperature - AS - (°C)',
    'Martensite Finish Temperature - MF - (°C)',
    'Martensite Start Temperature - MS - (°C)',
]

print(f"\n3. Missing Values Analysis:")
print("-" * 80)

# Check missing values for all features
all_features = element_features + process_features + temp_features
for feature in all_features:
    if feature in df.columns:
        missing_count = df[feature].isna().sum()
        missing_pct = (missing_count / len(df)) * 100
        print(f"   {feature:50s}: {missing_count:4d} missing ({missing_pct:5.1f}%)")
    else:
        print(f"   {feature:50s}: COLUMN NOT FOUND")

print(f"\n4. Target Variables Statistics:")
print("-" * 80)
for target in ['Austenite Finish Temperature - AF - (°C)',
               'Martensite Finish Temperature - MF - (°C)']:
    if target in df.columns:
        data = df[target].dropna()
        print(f"\n   {target}:")
        print(f"      Count:  {len(data)}")
        print(f"      Mean:   {data.mean():.2f}°C")
        print(f"      Std:    {data.std():.2f}°C")
        print(f"      Min:    {data.min():.2f}°C")
        print(f"      Max:    {data.max():.2f}°C")
        print(f"      Range:  {data.max() - data.min():.2f}°C")

print(f"\n5. Potential Input Features (AS, MS) Statistics:")
print("-" * 80)
for feature in ['Austenite Start Temperature - AS - (°C)',
                'Martensite Start Temperature - MS - (°C)']:
    if feature in df.columns:
        data = df[feature].dropna()
        print(f"\n   {feature}:")
        print(f"      Count:  {len(data)}")
        print(f"      Mean:   {data.mean():.2f}°C")
        print(f"      Std:    {data.std():.2f}°C")
        print(f"      Missing: {df[feature].isna().sum()} ({(df[feature].isna().sum()/len(df)*100):.1f}%)")

print(f"\n6. Correlation Analysis (Temperatures):")
print("-" * 80)

# Create correlation matrix for temperature features
temp_cols = [col for col in temp_features if col in df.columns]
if temp_cols:
    corr_matrix = df[temp_cols].corr()
    print("\n   Correlation Matrix:")
    print(corr_matrix.to_string())

    # Show strongest correlations
    print(f"\n   Key Correlations with AF:")
    if 'Austenite Finish Temperature - AF - (°C)' in corr_matrix.index:
        af_corr = corr_matrix['Austenite Finish Temperature - AF - (°C)'].sort_values(ascending=False)
        for feature, corr in af_corr.items():
            if feature != 'Austenite Finish Temperature - AF - (°C)':
                print(f"      {feature:50s}: {corr:.4f}")

    print(f"\n   Key Correlations with MF:")
    if 'Martensite Finish Temperature - MF - (°C)' in corr_matrix.index:
        mf_corr = corr_matrix['Martensite Finish Temperature - MF - (°C)'].sort_values(ascending=False)
        for feature, corr in mf_corr.items():
            if feature != 'Martensite Finish Temperature - MF - (°C)':
                print(f"      {feature:50s}: {corr:.4f}")

print(f"\n7. Data Quality Issues:")
print("-" * 80)

# Check for rows with valid targets but missing AS/MS
if all(col in df.columns for col in ['Austenite Finish Temperature - AF - (°C)',
                                       'Martensite Finish Temperature - MF - (°C)',
                                       'Austenite Start Temperature - AS - (°C)',
                                       'Martensite Start Temperature - MS - (°C)']):

    has_af_mf = df[['Austenite Finish Temperature - AF - (°C)',
                    'Martensite Finish Temperature - MF - (°C)']].notna().all(axis=1)
    has_as = df['Austenite Start Temperature - AS - (°C)'].notna()
    has_ms = df['Martensite Start Temperature - MS - (°C)'].notna()

    print(f"   Rows with AF & MF (targets): {has_af_mf.sum()}")
    print(f"   Rows with AS (feature):      {has_as.sum()}")
    print(f"   Rows with MS (feature):      {has_ms.sum()}")
    print(f"   Rows with AF & MF & AS & MS: {(has_af_mf & has_as & has_ms).sum()}")
    print(f"\n   ⚠ Data Loss if using AS/MS as features:")
    print(f"      Original samples with AF & MF: {has_af_mf.sum()}")
    print(f"      Remaining after requiring AS & MS: {(has_af_mf & has_as & has_ms).sum()}")
    print(f"      Loss: {has_af_mf.sum() - (has_af_mf & has_as & has_ms).sum()} samples ({((has_af_mf.sum() - (has_af_mf & has_as & has_ms).sum())/has_af_mf.sum()*100):.1f}%)")

print(f"\n8. Element Composition Statistics:")
print("-" * 80)
element_data = df[element_features].describe()
print(f"   Non-zero values per element:")
for col in element_features:
    non_zero = (df[col] > 0).sum()
    print(f"      {col:20s}: {non_zero:4d} samples ({non_zero/len(df)*100:5.1f}%)")

print(f"\n9. Recommended Features for Model:")
print("-" * 80)

# Features with < 5% missing values
good_features = []
for feature in element_features + process_features:
    if feature in df.columns:
        missing_pct = (df[feature].isna().sum() / len(df)) * 100
        if missing_pct < 5:
            good_features.append(feature)

print(f"   Features with <5% missing values ({len(good_features)}):")
for i, feature in enumerate(good_features, 1):
    missing_pct = (df[feature].isna().sum() / len(df)) * 100
    print(f"      {i:2d}. {feature:50s} ({missing_pct:.1f}% missing)")

print(f"\n10. Diagnosis Summary:")
print("="*80)
print("   KEY FINDINGS:")

# Check if AS/MS have high missing rates
if 'Austenite Start Temperature - AS - (°C)' in df.columns:
    as_missing = (df['Austenite Start Temperature - AS - (°C)'].isna().sum() / len(df)) * 100
    ms_missing = (df['Martensite Start Temperature - MS - (°C)'].isna().sum() / len(df)) * 100

    if as_missing > 20 or ms_missing > 20:
        print(f"\n   ❌ PROBLEM: AS/MS have high missing values!")
        print(f"      - AS missing: {as_missing:.1f}%")
        print(f"      - MS missing: {ms_missing:.1f}%")
        print(f"      - Using AS/MS as features causes significant data loss")
        print(f"      - This is why the 'improved' model failed!")

    if as_missing == 0 and ms_missing == 0:
        print(f"\n   ✓ AS/MS are complete - correlations should be checked")
        if 'Austenite Finish Temperature - AF - (°C)' in corr_matrix.index:
            as_corr = corr_matrix.loc['Austenite Start Temperature - AS - (°C)',
                                       'Austenite Finish Temperature - AF - (°C)']
            print(f"      - AS-AF correlation: {as_corr:.4f}")

print("\n" + "="*80)
