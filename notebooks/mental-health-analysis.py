#%%
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_ind
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Updated paths to cleaned datasets
prevalence_path = BASE_DIR / "data" / "processed" / "cleaned_mental_illnesses_prevalence.csv"
dalys_path = BASE_DIR / "data" / "processed" / "cleaned_disease_burden_dalys.csv"
treatment_gap_path = BASE_DIR / "data" / "processed" / "cleaned_anxiety_treatment_gap.csv"

prevalence_df = pd.read_csv(prevalence_path)
dalys_df = pd.read_csv(dalys_path)
treatment_df = pd.read_csv(treatment_gap_path)

#%%
# Descriptive Statistics for Depression and Anxiety Prevalence
print("Descriptive Statistics for Mental Illness Prevalence:")
print(prevalence_df[["Depression (%)", "Anxiety (%)"]].describe())

#%%
# Visualization 1: Global Average Prevalence of Depression and Anxiety Over Time
trend = prevalence_df.groupby("Year")[["Depression (%)", "Anxiety (%)"]].mean().reset_index()

plt.figure(figsize=(10, 6))
plt.plot(trend["Year"], trend["Depression (%)"], label="Depression")
plt.plot(trend["Year"], trend["Anxiety (%)"], label="Anxiety")
plt.title("Global Average Prevalence of Depression and Anxiety Over Time")
plt.xlabel("Year")
plt.ylabel("Prevalence (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
# Visualization 2: Top 10 Countries with Highest Untreated Anxiety
top_untreated = treatment_df.sort_values(by="Untreated (%)", ascending=False).head(10)

plt.figure(figsize=(10, 6))
plt.barh(top_untreated["Country"], top_untreated["Untreated (%)"], color='salmon')
plt.title("Top 10 Countries with Highest Rates of Untreated Anxiety")
plt.xlabel("Percentage Untreated (%)")
plt.ylabel("Country")
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#%%
# Visualization 3: Depression Prevalence vs. DALYs from Depression
merged_df = pd.merge(
    prevalence_df[["Country", "Year", "Depression (%)"]],
    dalys_df[["Country", "Year", "DALYs Depression"]],
    on=["Country", "Year"]
).dropna()

plt.figure(figsize=(8, 6))
plt.scatter(merged_df["Depression (%)"], merged_df["DALYs Depression"], alpha=0.6)
plt.title("Depression Prevalence vs. DALYs from Depression")
plt.xlabel("Depression Prevalence (%)")
plt.ylabel("DALYs from Depression (per 100,000)")
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
# Hypothesis Test 1: Correlation between Depression and DALYs
correlation, p_value = pearsonr(merged_df["Depression (%)"], merged_df["DALYs Depression"])
print("Hypothesis Test 1: Pearson Correlation")
print("Correlation Coefficient:", round(correlation, 3))
print("p-value:", p_value)
if p_value < 0.05:
    print("Result: Statistically significant positive correlation.")
else:
    print("Result: No statistically significant correlation.")

#%%
# Hypothesis Test 2: T-test on Depression Rates - High vs. Low Treatment Coverage
# Define threshold as the median of "Adequate Treatment (%)"
threshold = treatment_df["Adequate Treatment (%)"].median()
high_coverage = treatment_df[treatment_df["Adequate Treatment (%)"] > threshold]["Country"]
low_coverage = treatment_df[treatment_df["Adequate Treatment (%)"] <= threshold]["Country"]

# Get depression rates by coverage group
high_depression = prevalence_df[prevalence_df["Country"].isin(high_coverage)]["Depression (%)"]
low_depression = prevalence_df[prevalence_df["Country"].isin(low_coverage)]["Depression (%)"]

# T-test
t_stat, t_p_value = ttest_ind(high_depression.dropna(), low_depression.dropna(), equal_var=False)

print("\nHypothesis Test 2: T-test - Depression Rates by Treatment Access")
print("T-statistic:", round(t_stat, 3))
print("p-value:", t_p_value)
if t_p_value < 0.05:
    print("Result: Statistically significant difference in depression rates.")
else:
    print("Result: No statistically significant difference.")

# %%
# Merge relevant DALY features with depression prevalence
regression_df = pd.merge(
    prevalence_df[["Country", "Year", "Depression (%)"]],
    dalys_df[[
        "Country", "Year", "DALYs Depression", "DALYs Anxiety", "DALYs Bipolar",
        "DALYs Schizophrenia", "DALYs Eating Disorders"
    ]],
    on=["Country", "Year"]
).dropna()

# Define features and target
X = regression_df[[
    "DALYs Depression", "DALYs Anxiety", "DALYs Bipolar", 
    "DALYs Schizophrenia", "DALYs Eating Disorders"
]]
y = regression_df["Depression (%)"]

# Fit linear regression model
reg = LinearRegression()
reg.fit(X, y)

# Predict and evaluate
y_pred = reg.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
coefficients = dict(zip(X.columns, reg.coef_))

print("\nLinear Regression: Predicting Depression Prevalence")
print("Mean Squared Error:", round(mse, 5))
print("R-squared Score:", round(r2, 4))
print("Feature Coefficients:")
for feature, coef in coefficients.items():
    print(f"{feature}: {coef:.6f}")
#%%
# Visualization: Actual vs Predicted Depression Prevalence
plt.figure(figsize=(8,6))
plt.scatter(y, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.xlabel("Actual Depression Prevalence (%)")
plt.ylabel("Predicted Depression Prevalence (%)")
plt.title("Actual vs Predicted Depression Prevalence")
plt.grid(True)
plt.tight_layout()
plt.show()
#%%
