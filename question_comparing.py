import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

# 1. LOAD YOUR DATA
# Replace with your actual file name
try:
    df = pd.read_csv("batch_results_v4.csv")
except:
    # Creating dummy data if file not found (for demonstration)
    print("⚠️ File not found. Using synthetic data for demo.")
    data = []
    questions = [f'Q{i}' for i in range(1, 12)]
    max_marks_list = [5, 5, 10, 10, 15, 20, 25, 30, 35, 40, 50]
    for q, mm in zip(questions, max_marks_list):
        for _ in range(20):
            h = np.random.uniform(mm*0.5, mm)
            a = h * (0.9 if mm < 20 else 0.6) + np.random.normal(0, 2)
            data.append({'question_id': q, 'max_marks': mm, 'human_score': h, 'ai_score': a})
    df = pd.DataFrame(data)

# Ensure numeric types
df['human_score'] = pd.to_numeric(df['human_score'], errors='coerce')
df['ai_score'] = pd.to_numeric(df['ai_score'], errors='coerce')
df['max_marks'] = pd.to_numeric(df['max_marks'], errors='coerce')

# 2. SETUP THE PLOT GRID
unique_qs = df['question_id'].unique()
n_plots = len(unique_qs)
n_cols = 4  # Number of columns in the grid
n_rows = math.ceil(n_plots / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
axes = axes.flatten()  # Flatten to 1D array for easy looping

# 3. LOOP THROUGH EACH QUESTION AND PLOT
for i, q_id in enumerate(unique_qs):
    ax = axes[i]
    subset = df[df['question_id'] == q_id]
    
    # Get max marks for this specific question to set axis limits
    max_val = subset['max_marks'].max()
    
    # A. Scatter Plot (The actual student scores)
    sns.scatterplot(
        data=subset, 
        x='human_score', 
        y='ai_score', 
        ax=ax, 
        color='royalblue', 
        s=80, 
        alpha=0.6,
        edgecolor='w', 
        linewidth=0.5
    )
    
    # B. The "Perfect Agreement" Line (y=x) - RED DASHED
    # If points fall on this line, AI = Human
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Match (y=x)')
    
    # C. The "Trend Line" (Regression) - GREEN
    # If this line is below the Red line, AI is stricter.
    if len(subset) > 1:
        sns.regplot(
            data=subset, 
            x='human_score', 
            y='ai_score', 
            ax=ax, 
            scatter=False, 
            color='green', 
            line_kws={"linewidth": 2, "label": "Actual Trend"}
        )
    
    # Formatting
    ax.set_title(f"Question: {q_id} (Max: {max_val})", fontsize=14, fontweight='bold')
    ax.set_xlabel("Human Score", fontsize=11)
    ax.set_ylabel("AI Score", fontsize=11)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Add correlation text on the plot
    if len(subset) > 1:
        corr = subset['human_score'].corr(subset['ai_score'])
        ax.text(0.05, 0.9, f'r = {corr:.2f}', transform=ax.transAxes, 
                fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

# Remove empty subplots if any
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Question-wise Grading Comparison: Human vs. AI Strictness", fontsize=22, y=1.02)
plt.tight_layout()
plt.savefig('question_wise_analysis4.png', bbox_inches='tight')
print("✅ Graph saved as 'question_wise_analysis.png'")
plt.show()