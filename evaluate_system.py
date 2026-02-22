import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, cohen_kappa_score, confusion_matrix, accuracy_score
from scipy.stats import pearsonr

# --- CONFIGURATION ---
INPUT_FILE = "adaptive_results.csv"

# --- 1. LOAD DATA ---
print(f"📂 Loading results from '{INPUT_FILE}'...")
try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"❌ Error: '{INPUT_FILE}' not found. Please run 'run_grading_math.py' first.")
    exit()

# Clean Data
df = df.dropna(subset=['human_score', 'ai_score'])
human = df['human_score'].values
ai = df['ai_score'].values
n = len(df)

print(f"--- 📊 DETAILED EVALUATION REPORT (N={n} Scripts) ---")

# --- 2. STATISTICAL METRICS ---
mae = mean_absolute_error(human, ai)
mse = mean_squared_error(human, ai)
rmse = np.sqrt(mse)
corr, _ = pearsonr(human, ai)

print(f"\n1. NUMERICAL ACCURACY:")
print(f"   - MAE (Mean Absolute Error): {mae:.3f} marks")
print(f"     (On average, the AI deviates by {mae:.3f} points from the lecturer)")
print(f"   - RMSE (Root Mean Sq Error): {rmse:.3f}")
print(f"   - Pearson Correlation (r):   {corr:.3f}")
print(f"     (Strength of relationship: 0.7+ is Strong, 0.9+ is Very Strong)")

# --- 3. CATEGORICAL AGREEMENT (Letter Grades) ---
# Function to convert scores to grades (Standard 10-point scale assumption, scalable)
def get_grade(score, max_score):
    percentage = (score / max_score) * 100
    if percentage >= 75: return 'A'
    elif percentage >= 65: return 'B'
    elif percentage >= 50: return 'C'
    elif percentage >= 35: return 'S'
    else: return 'F'

# Apply grading logic
# Using 'max_marks' from CSV if available, else defaulting to 10 for safety
if 'max_marks' in df.columns:
    df['human_grade'] = df.apply(lambda x: get_grade(x['human_score'], x['max_marks']), axis=1)
    df['ai_grade'] = df.apply(lambda x: get_grade(x['ai_score'], x['max_marks']), axis=1)
else:
    # Fallback if max_marks missing (Assuming raw score is out of 10)
    print("⚠️ 'max_marks' column missing. Assuming scores are out of 10.")
    df['human_grade'] = df.apply(lambda x: get_grade(x['human_score'], 10), axis=1)
    df['ai_grade'] = df.apply(lambda x: get_grade(x['ai_score'], 10), axis=1)

# Categorical Accuracy
acc = accuracy_score(df['human_grade'], df['ai_grade'])
print(f"\n2. CATEGORICAL ACCURACY (Grades A/B/C/S/F):")
print(f"   - Exact Grade Match: {acc*100:.1f}%")

# Weighted Kappa (Reliability)
# We map grades to numbers for Kappa: F=0, S=1, C=2, B=3, A=4
grade_map = {'F':0, 'S':1, 'C':2, 'B':3, 'A':4}
h_k = df['human_grade'].map(grade_map).fillna(0).astype(int)
a_k = df['ai_grade'].map(grade_map).fillna(0).astype(int)
kappa = cohen_kappa_score(h_k, a_k, weights='quadratic')

print(f"   - Quadratic Weighted Kappa:  {kappa:.3f}")
print("     ( Interpretation: <0.2 Poor | 0.2-0.4 Fair | 0.4-0.6 Moderate | 0.6-0.8 Good | >0.8 Excellent )")

# --- 4. ERROR ANALYSIS (Outliers) ---
df['abs_error'] = abs(df['human_score'] - df['ai_score'])
worst_cases = df.sort_values(by='abs_error', ascending=False).head(5)

print("\n3. TOP 5 WORST ERRORS (Outlier Analysis):")
print(f"{'Q_ID':<8} | {'Human':<6} | {'AI':<6} | {'Diff':<6}")
print("-" * 35)
for _, row in worst_cases.iterrows():
    print(f"{row['question_id']:<8} | {row['human_score']:<6} | {row['ai_score']:<6} | {row['abs_error']:.2f}")

# --- 5. VISUALIZATION 1: Scatter Plot ---
plt.figure(figsize=(10, 6))
# Using standard matplotlib scatter if seaborn is missing, but seaborn is preferred
try:
    sns.regplot(x=human, y=ai, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
except:
    plt.scatter(human, ai, alpha=0.5)
    
plt.plot([min(human), max(human)], [min(human), max(human)], 'g--', label='Ideal (x=y)')
plt.xlabel("Lecturer Score")
plt.ylabel("AI Predicted Score")
plt.title(f"Grading Correlation (r={corr:.2f}, N={n})")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("correlation_plot.png")
print("\n✅ Saved 'correlation_plot.png' (Include this in your Results chapter)")


# --- 6. VISUALIZATION 2: Confusion Matrix ---
labels = ['F', 'S', 'C', 'B', 'A']
# Ensure labels exist in data to prevent errors
existing_labels = sorted(list(set(df['human_grade'].unique()) | set(df['ai_grade'].unique())))
cm = confusion_matrix(df['human_grade'], df['ai_grade'], labels=labels)

plt.figure(figsize=(8, 6))
try:
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
except:
    print("Warning: Seaborn not installed. Skipping heatmap generation.")

plt.xlabel("AI Predicted Grade")
plt.ylabel("Actual Human Grade")
plt.title("Grade Confusion Matrix")
plt.savefig("confusion_matrix.png")
print("✅ Saved 'confusion_matrix.png' (Shows which grades are confused most often)")


print("\n" + "="*50)
print("Evaluation Complete. Use these metrics for your Thesis Defense.")
print("="*50)