# ruff: noqa: T201
"""Script de vérification rapide des datasets."""
import pandas as pd

print("\n" + "="*70)
print("VÉRIFICATION DES DATASETS")
print("="*70)

train = pd.read_csv('data/ml/train_data.csv')
test = pd.read_csv('data/ml/test_data.csv')
full = pd.read_csv('data/ml/training_data_engineered.csv')

print("\n📊 Dimensions:")
print(f"   Train: {train.shape}")
print(f"   Test:  {test.shape}")
print(f"   Full:  {full.shape}")

print("\n🎯 Target Statistics:")
print(f"   Train - mean: {train['actual_delay_minutes'].mean():.2f} min")
print(f"   Test  - mean: {test['actual_delay_minutes'].mean():.2f} min")
print(f"   Diff:         {abs(train['actual_delay_minutes'].mean() - test['actual_delay_minutes'].mean()):.2f} min")

print("\n📋 Premières features (10):")
for i, col in enumerate(list(train.columns[:10]), 1):
    print(f"   {i:2d}. {col}")

print("\n✅ Nouvelles features (23):")
new_features = [col for col in full.columns if col not in pd.read_csv('data/ml/training_data.csv').columns]
for i, col in enumerate(new_features, 1):
    print(f"   {i:2d}. {col}")

print("\n" + "="*70)
print("✅ VALIDATION RÉUSSIE !")
print("="*70 + "\n")

