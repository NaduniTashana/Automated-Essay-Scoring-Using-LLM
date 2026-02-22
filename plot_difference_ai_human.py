# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the data
# data = pd.read_csv("hybrid_results_gated.csv")

# # Group by max_marks
# grouped_data = data.groupby("max_marks")[["ai_score", "human_score"]].mean().reset_index()

# # Plotting
# plt.figure(figsize=(10, 6))
# sns.barplot(x="max_marks", y="value", hue="variable", 
#             data=grouped_data.melt(id_vars="max_marks", value_vars=["ai_score", "human_score"]))

# # Customize the plot
# plt.title("AI Score vs Human Score Grouped by Max Marks", fontsize=14)
# plt.xlabel("Max Marks", fontsize=12)
# plt.ylabel("Average Score", fontsize=12)
# plt.legend(title="Score Type")
# plt.grid(axis="y", linestyle="--", alpha=0.7)

# # Show the plot
# plt.tight_layout()
# plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. LOAD YOUR DATA
# Replace with your actual file name
df = pd.read_csv("batch_results_v4.csv") 

# Ensure columns are numeric
df['human_score'] = pd.to_numeric(df['human_score'], errors='coerce')
df['ai_score'] = pd.to_numeric(df['ai_score'], errors='coerce')
df['max_marks'] = pd.to_numeric(df['max_marks'], errors='coerce')

# 2. CALCULATE THE GAP
# We care about the MAGNITUDE of the difference
df['score_gap'] = abs(df['human_score'] - df['ai_score'])

# 3. SET UP THE PLOTS
plt.figure(figsize=(16, 6))

# --- PLOT A: SCATTER PLOT (The Proof of Correlation) ---
plt.subplot(1, 2, 1)

# Create the scatter plot
sns.scatterplot(
    data=df, 
    x='max_marks', 
    y='score_gap', 
    hue='question_id', 
    palette='viridis', 
    s=100, 
    alpha=0.7
)

# Add a Regression Line (The trend line)
sns.regplot(
    data=df, 
    x='max_marks', 
    y='score_gap', 
    scatter=False, 
    color='red', 
    line_kws={"linestyle": "--"}
)

plt.title('Correlation: Question Complexity vs. Grading Disagreement', fontsize=14, fontweight='bold')
plt.xlabel('Question Max Marks (Essay Complexity)', fontsize=12)
plt.ylabel('Score Gap (|Human - AI|)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

# --- PLOT B: BAR CHART (The Question Breakdown) ---
plt.subplot(1, 2, 2)

# Group by Question ID and calculate the AVERAGE gap for that question
# We sort by Max Marks so the graph goes from Small -> Large questions
grouped = df.groupby(['question_id', 'max_marks'])['score_gap'].mean().reset_index()
grouped = grouped.sort_values('max_marks')

# Create the bar chart
sns.barplot(
    data=grouped, 
    x='question_id', 
    y='score_gap', 
    hue='max_marks', 
    palette='rocket', 
    dodge=False
)

plt.title('Average Error per Question (Sorted by Size)', fontsize=14, fontweight='bold')
plt.xlabel('Question ID', fontsize=12)
plt.ylabel('Average Marks Difference', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Max Marks')
plt.grid(axis='y', linestyle='--', alpha=0.5)

# 4. SAVE AND SHOW
plt.tight_layout()
plt.savefig('thesis_error_analysis4.png', dpi=300)
print("✅ Graph saved as 'thesis_error_analysis.png'")
plt.show()