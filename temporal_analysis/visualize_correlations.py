import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df_since = pd.read_csv('temporal_analysis/finalsince.csv')
df_to = pd.read_csv('temporal_analysis/finalto.csv')
df = pd.concat([df_since, df_to], ignore_index=True)

# Convert year and score to numeric (in case they're strings)
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['score'] = pd.to_numeric(df['score'], errors='coerce')
df['difficulty'] = pd.to_numeric(df['difficulty'], errors='coerce')

# Remove rows with NaN values
df_clean = df.dropna(subset=['year', 'score', 'difficulty'])

# Transform year to thousands of years (divide by 1000)
df_clean['year'] = df_clean['year'] / 1000.0

print(f"Total rows: {len(df)}")
print(f"Rows after cleaning: {len(df_clean)}")
print(f"\nYear range (in thousands): {df_clean['year'].min():.3f} - {df_clean['year'].max():.3f}")
print(f"Score range: {df_clean['score'].min():.3f} - {df_clean['score'].max():.3f}")
print(f"Difficulty range: {df_clean['difficulty'].min():.3f} - {df_clean['difficulty'].max():.3f}")

# Calculate covariance and correlation
# Year vs Score
cov_year_score = np.cov(df_clean['year'], df_clean['score'])[0, 1]
corr_year_score = df_clean['year'].corr(df_clean['score'])

# Year vs Difficulty
cov_year_difficulty = np.cov(df_clean['year'], df_clean['difficulty'])[0, 1]
corr_year_difficulty = df_clean['year'].corr(df_clean['difficulty'])

print("\n" + "="*60)
print("STATISTICS")
print("="*60)
print(f"\nYear vs GPT Hallucination Rate:")
print(f"  Covariance: {cov_year_score:.6f}")
print(f"  Correlation Coefficient: {corr_year_score:.6f}")

print(f"\nYear vs Question Difficulty (Flesch):")
print(f"  Covariance: {cov_year_difficulty:.6f}")
print(f"  Correlation Coefficient: {corr_year_difficulty:.6f}")

# Create scatterplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Year vs Score
axes[0].scatter(df_clean['year'], df_clean['score'], alpha=0.6, s=50, color='steelblue', edgecolors='black', linewidth=0.5)
axes[0].set_xlabel('Question Year (thousands)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Hallucination Rate', fontsize=12, fontweight='bold')
axes[0].set_title(f'Question Year vs GPT Hallucination Rate (Correlation: {corr_year_score:.3f})', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, linestyle='--')
axes[0].tick_params(labelsize=10)

# Add trend line for Year vs Score
z_score = np.polyfit(df_clean['year'], df_clean['score'], 1)
p_score = np.poly1d(z_score)
axes[0].plot(df_clean['year'], p_score(df_clean['year']), "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z_score[0]:.6f}x+{z_score[1]:.2f}')
axes[0].legend(fontsize=9)

# Plot 2: Year vs Difficulty
axes[1].scatter(df_clean['year'], df_clean['difficulty'], alpha=0.6, s=50, color='coral', edgecolors='black', linewidth=0.5)
axes[1].set_xlabel('Question Year (thousands)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Difficulty', fontsize=12, fontweight='bold')
axes[1].set_title(f'Year vs Question Difficulty (Flesch)\n(Correlation: {corr_year_difficulty:.3f})', fontsize=13, fontweight='bold')


axes[1].grid(True, alpha=0.3, linestyle='--')
axes[1].tick_params(labelsize=10)

# Add trend line for Year vs Difficulty
z_difficulty = np.polyfit(df_clean['year'], df_clean['difficulty'], 1)
p_difficulty = np.poly1d(z_difficulty)
axes[1].plot(df_clean['year'], p_difficulty(df_clean['year']), "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z_difficulty[0]:.6f}x+{z_difficulty[1]:.2f}')
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig('correlations_plot.png', dpi=300, bbox_inches='tight')
print("\n" + "="*60)
print(f"Plots saved to: correlations_plot.png")
print("="*60)

plt.show()

