
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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

# Residualize Year and Hallucination Rate w.r.t. Difficulty for later analysis
X_diff = df_clean[['difficulty']].values

reg_year_diff = LinearRegression()
reg_year_diff.fit(X_diff, df_clean['year'].values)
year_hat = reg_year_diff.predict(X_diff)
year_residual = df_clean['year'].values - year_hat

reg_score_diff = LinearRegression()
reg_score_diff.fit(X_diff, df_clean['score'].values)
score_hat = reg_score_diff.predict(X_diff)
score_residual = df_clean['score'].values - score_hat

residual_df = pd.DataFrame({
    'year_residual': year_residual,
    'score_residual': score_residual
})

print(f"Total rows: {len(df)}")
print(f"Rows after cleaning: {len(df_clean)}")
print(f"\nYear range: {df_clean['year'].min()} - {df_clean['year'].max()}")
print(f"Score range: {df_clean['score'].min():.3f} - {df_clean['score'].max():.3f}")
print(f"Difficulty range: {df_clean['difficulty'].min():.3f} - {df_clean['difficulty'].max():.3f}")

# Calculate overall correlations
corr_year_score = df_clean['year'].corr(df_clean['score'])
corr_year_difficulty = df_clean['year'].corr(df_clean['difficulty'])
corr_score_difficulty = df_clean['score'].corr(df_clean['difficulty'])

# Calculate R² (coefficient of determination) for Year vs Score
X_year = df_clean[['year']].values
y_score = df_clean['score'].values

# Fit linear regression: Score ~ Year
reg = LinearRegression()
reg.fit(X_year, y_score)
y_pred = reg.predict(X_year)
r2_year_score = r2_score(y_score, y_pred)

print("\n" + "="*60)
print("R² (COEFFICIENT OF DETERMINATION): Year vs Score")
print("="*60)
print(f"R² = {r2_year_score:.6f}")
print(f"Interpretation: {r2_year_score*100:.2f}% of the variance in Score is explained by Year")
print(f"  - R² ranges from 0 to 1")
print(f"  - R² = 0 means Year explains none of the variance in Score")
print(f"  - R² = 1 means Year perfectly explains all variance in Score")
print(f"  - R² = {r2_year_score:.4f} means Year explains {r2_year_score*100:.2f}% of Score variance")
print(f"  - The remaining {100-r2_year_score*100:.2f}% of variance is unexplained by Year")

# Calculate correlation of Year vs Score conditional on Difficulty
print("\n" + "="*60)
print("CONDITIONAL CORRELATION: Year vs Score given Difficulty")
print("="*60)

# Create difficulty bins for grouping
# Use equal-width bins (constant difficulty range) instead of quantiles
difficulty_min = df_clean['difficulty'].min()
difficulty_max = df_clean['difficulty'].max()
num_bins = 5
bin_width = (difficulty_max - difficulty_min) / num_bins

# Create 5 equal-width bins
bin_edges = [difficulty_min + i * bin_width for i in range(num_bins + 1)]
print(f"\nDifficulty range: {difficulty_min:.3f} - {difficulty_max:.3f}")
print(f"Bin width: {bin_width:.3f}")
print(f"Bin edges (5 equal-width bins): {[f'{edge:.3f}' for edge in bin_edges]}")

# Method 1: Group by difficulty bins (5 equal-width bins)
df_clean['difficulty_bin'] = pd.cut(
    df_clean['difficulty'], 
    bins=bin_edges,
    labels=['Very Easy', 'Easy', 'Medium', 'Hard', 'Very Hard'],
    include_lowest=True
)

print("\n--- Correlation by Difficulty Bins (5 bins) ---")
for bin_name in ['Very Easy', 'Easy', 'Medium', 'Hard', 'Very Hard']:
    bin_data = df_clean[df_clean['difficulty_bin'] == bin_name]
    if len(bin_data) >= 2:  # Need at least 2 points for correlation
        corr = bin_data['year'].corr(bin_data['score'])
        cov = np.cov(bin_data['year'], bin_data['score'])[0, 1] if len(bin_data) > 1 else np.nan
        print(f"\n{bin_name} Difficulty (n={len(bin_data)}):")
        print(f"  Difficulty range: {bin_data['difficulty'].min():.3f} - {bin_data['difficulty'].max():.3f}")
        print(f"  Covariance: {cov:.6f}")
        print(f"  Correlation: {corr:.6f}")
    else:
        print(f"\n{bin_name} Difficulty: Not enough data (n={len(bin_data)})")

# Method 2: Calculate correlation for each unique difficulty value (if there are few unique values)
# Or use smaller bins
print("\n--- Correlation by Difficulty (Fine-grained bins) ---")
# Create 5 equal-width bins
df_clean['difficulty_bin_fine'] = pd.cut(
    df_clean['difficulty'],
    bins=5,
    labels=['Very Easy', 'Easy', 'Medium', 'Hard', 'Very Hard'],
)

for bin_name in df_clean['difficulty_bin_fine'].cat.categories:
    bin_data = df_clean[df_clean['difficulty_bin_fine'] == bin_name]
    if len(bin_data) >= 2:
        corr = bin_data['year'].corr(bin_data['score'])
        cov = np.cov(bin_data['year'], bin_data['score'])[0, 1] if len(bin_data) > 1 else np.nan
        print(f"\n{bin_name} (n={len(bin_data)}):")
        print(f"  Difficulty range: {bin_data['difficulty'].min():.3f} - {bin_data['difficulty'].max():.3f}")
        print(f"  Correlation: {corr:.6f}")

# Store conditional correlations for visualization (5 bins)
conditional_corrs = []
conditional_bins = []
bin_probs = []  # P(Difficulty = d) for marginalization
bin_corrs = []  # Correlation within each bin

for bin_name in ['Very Easy', 'Easy', 'Medium', 'Hard', 'Very Hard']:
    bin_data = df_clean[df_clean['difficulty_bin'] == bin_name]
    if len(bin_data) >= 2:
        corr = bin_data['year'].corr(bin_data['score'])
        conditional_corrs.append(corr)
        conditional_bins.append(bin_name)
        # Calculate probability of this difficulty bin: P(Difficulty = d)
        prob = len(bin_data) / len(df_clean)
        bin_probs.append(prob)
        bin_corrs.append(corr)

# Marginalize out difficulty using: P(Year, Score) = ∑_Difficulty P(Year, Score, Difficulty)
# For correlation, we marginalize by weighting correlations by difficulty probabilities
print("\n" + "="*60)
print("MARGINAL CORRELATION: Year vs Score (marginalizing out Difficulty)")
print("="*60)
print("Using formula: P(X, Y) = ∑_Z P(X, Y, Z) for discrete variables")
print("\nCorrelation within each difficulty bin (conditional):")
for i, (bin_name, corr, prob) in enumerate(zip(conditional_bins, bin_corrs, bin_probs)):
    print(f"  {bin_name}: Correlation = {corr:.6f}, P(Difficulty={bin_name}) = {prob:.4f}")

# Calculate marginal correlation as weighted average
# This is an approximation: we weight the conditional correlations by difficulty probabilities
if len(bin_corrs) > 0:
    marginal_corr = np.average(bin_corrs, weights=bin_probs)
    print(f"\nMarginal Correlation (weighted by difficulty distribution): {marginal_corr:.6f}")
    print(f"  This represents the correlation between Year and Score after marginalizing out Difficulty")
    print(f"  Each bin's correlation is weighted by its probability P(Difficulty = d)")
else:
    marginal_corr = np.nan
    print("\nNot enough data to calculate marginal correlation")

# Residualized bins with weighted correlation
print("\n" + "="*60)
print("RESIDUALIZED BIN ANALYSIS")
print("="*60)

resid_min = residual_df['year_residual'].min()
resid_max = residual_df['year_residual'].max()
bin_start = int(np.floor(resid_min))
bin_end = int(np.ceil(resid_max)) + 1
bin_edges = list(range(bin_start, bin_end + 1))

residual_df['year_residual_bin'] = pd.cut(residual_df['year_residual'], bins=bin_edges, include_lowest=True)

resid_bin_stats = residual_df.groupby('year_residual_bin').agg({
    'score_residual': 'mean',
    'year_residual': ['mean', 'count']
}).reset_index()

resid_bin_stats.columns = ['year_resid_bin', 'avg_score_resid', 'bin_center_resid', 'count']
resid_bin_stats = resid_bin_stats[resid_bin_stats['count'] > 0]

print(f"\nNumber of residual bins: {len(resid_bin_stats)}")
print("\nResidual Bin Statistics:")
for _, row in resid_bin_stats.iterrows():
    print(f"  {row['year_resid_bin']}: Center={row['bin_center_resid']:.3f}, Avg Residual HR={row['avg_score_resid']:.3f}, Count={row['count']}")

# Calculate weighted correlation
def weighted_corr(x, y, weights):
    """Calculate weighted correlation coefficient."""
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    # Calculate weighted means
    x_mean = np.average(x, weights=weights)
    y_mean = np.average(y, weights=weights)
    
    # Calculate weighted covariance and variances
    x_centered = x - x_mean
    y_centered = y - y_mean
    
    cov_xy = np.sum(weights * x_centered * y_centered)
    var_x = np.sum(weights * x_centered ** 2)
    var_y = np.sum(weights * y_centered ** 2)
    
    # Correlation
    if var_x > 0 and var_y > 0:
        corr = cov_xy / np.sqrt(var_x * var_y)
    else:
        corr = np.nan
    
    return corr

if len(resid_bin_stats) >= 2:
    weighted_corr_resid = weighted_corr(
        resid_bin_stats['bin_center_resid'].values,
        resid_bin_stats['avg_score_resid'].values,
        resid_bin_stats['count'].values
    )
    print(f"\nWeighted Correlation (Residual Year vs Residual HR, binned): {weighted_corr_resid:.6f}")
else:
    weighted_corr_resid = np.nan
    print("\nNot enough bins for weighted residual correlation calculation")

# Residualized scatterplot: remove effect of difficulty from Year and Hallucination Rate
print("\n" + "="*60)
print("RESIDUALIZED SCATTERPLOT: (Year - Year̂(Difficulty)) vs (HR - HR̂(Difficulty))")
print("="*60)

print(f"\nResidual Year range: {residual_df['year_residual'].min():.3f} to {residual_df['year_residual'].max():.3f}")
print(f"Residual Hallucination Rate range: {residual_df['score_residual'].min():.3f} to {residual_df['score_residual'].max():.3f}")

corr_residual = residual_df['year_residual'].corr(residual_df['score_residual'])
cov_residual = np.cov(residual_df['year_residual'], residual_df['score_residual'])[0, 1]

print(f"\nCorrelation (Residual Year vs Residual Hallucination Rate): {corr_residual:.6f}")
print(f"Covariance: {cov_residual:.6f}")
print("  This represents the correlation after removing the effect of difficulty from both variables.")

# Create plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Residualized Year vs Residualized Hallucination Rate
axes[0].scatter(residual_df['year_residual'], residual_df['score_residual'], 
                alpha=0.6, s=40, color='steelblue', edgecolors='black', linewidth=0.4)
axes[0].set_xlabel("Residualized Question Year", fontsize=12, fontweight='bold')
axes[0].set_ylabel("Residualized Hallucination Rate", fontsize=12, fontweight='bold')
axes[0].set_title(f"Residualized Year vs Hallucination Rate\nCorrelation: {corr_residual:.3f}", fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, linestyle='--')
axes[0].tick_params(labelsize=10)

# Add trend line
if len(residual_df) >= 2:
    z_resid = np.polyfit(residual_df['year_residual'], residual_df['score_residual'], 1)
    p_resid = np.poly1d(z_resid)
    axes[0].plot(residual_df['year_residual'], p_resid(residual_df['year_residual']),
                 "r--", alpha=0.8, linewidth=2,
                 label=f"Trend: y={z_resid[0]:.6f}x+{z_resid[1]:.2f}")
    axes[0].legend(fontsize=9)

# Plot 2: Conditional Correlation by Difficulty (5 bins)
if conditional_corrs:
    # Use 5 colors for 5 bins (gradient from green to red)
    colors_5 = ['#2E7D32', '#4CAF50', '#FFC107', '#FF9800', '#F44336']
    bars = axes[1].bar(conditional_bins, conditional_corrs, color=colors_5[:len(conditional_corrs)], 
                       alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    axes[1].set_xlabel('Question Difficulty', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Correlation (Year vs Hallucination Rate)', fontsize=12, fontweight='bold')
    axes[1].set_title('Conditional Correlation: \nYear vs Hallucination Rate given Question Difficulty', 
                      fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--', axis='y')
    axes[1].tick_params(labelsize=9)
    # Rotate x-axis labels if needed for better readability
    axes[1].tick_params(axis='x', rotation=15)
    
    # Add value labels on bars
    for bar, corr in zip(bars, conditional_corrs):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{corr:.3f}',
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=11, fontweight='bold')
else:
    axes[1].text(0.5, 0.5, 'Not enough data\nfor conditional correlation', 
                ha='center', va='center', transform=axes[1].transAxes,
                fontsize=12)
    axes[1].set_title('Conditional Correlation:\nYear vs Score given Difficulty', 
                      fontsize=13, fontweight='bold')

# Plot 3: Residualized bins with weighted correlation
if len(resid_bin_stats) > 0:
    # Scatter plot with point sizes proportional to bin counts
    scatter = axes[2].scatter(resid_bin_stats['bin_center_resid'], resid_bin_stats['avg_score_resid'], 
                             s=resid_bin_stats['count']*10,  # Scale point size by count
                             alpha=0.6, c=resid_bin_stats['count'], 
                             cmap='viridis', edgecolors='black', linewidth=1)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes[2])
    cbar.set_label('Number of Data Points', fontsize=11, fontweight='bold')
    
    # Add trend line using weighted regression
    if len(resid_bin_stats) >= 2 and not np.isnan(weighted_corr_resid):
        # Weighted linear regression
        weights = resid_bin_stats['count'].values
        weights_norm = weights / weights.sum()
        
        x = resid_bin_stats['bin_center_resid'].values
        y = resid_bin_stats['avg_score_resid'].values
        
        # Weighted least squares
        x_mean = np.average(x, weights=weights_norm)
        y_mean = np.average(y, weights=weights_norm)
        
        numerator = np.sum(weights_norm * (x - x_mean) * (y - y_mean))
        denominator = np.sum(weights_norm * (x - x_mean) ** 2)
        
        if denominator > 0:
            slope = numerator / denominator
            intercept = y_mean - slope * x_mean
            
            x_trend = np.linspace(x.min(), x.max(), 100)
            y_trend = slope * x_trend + intercept
            axes[2].plot(x_trend, y_trend, 'r--', linewidth=2, alpha=0.8,
                        label=f'Weighted Trend: y={slope:.6f}x+{intercept:.2f}')
            axes[2].legend(fontsize=9)
    
    axes[2].set_xlabel('Residualized Question Year (Bin Center)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Average Residualized Hallucination Rate', fontsize=12, fontweight='bold')
    title = f'Residualized Year vs Residualized HR (binned)\n'
    if not np.isnan(weighted_corr_resid):
        title += f'Weighted Correlation: {weighted_corr_resid:.3f}'
    axes[2].set_title(title, fontsize=13, fontweight='bold')
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].tick_params(labelsize=10)
    
    # Add text annotation showing bin counts
    for _, row in resid_bin_stats.iterrows():
        axes[2].annotate(f'n={int(row["count"])}', 
                        xy=(row['bin_center_resid'], row['avg_score_resid']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
else:
    axes[2].text(0.5, 0.5, 'Not enough data\nfor residualized bin analysis', 
                ha='center', va='center', transform=axes[2].transAxes,
                fontsize=12)
    axes[2].set_title('Residualized Year vs Residualized HR (binned)', 
                      fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('conditional_and_bins_plot.png', dpi=300, bbox_inches='tight')
print("\n" + "="*60)
print(f"Plots saved to: conditional_and_bins_plot.png")
print("="*60)

plt.show()

