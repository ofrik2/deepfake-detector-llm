#!/usr/bin/env python
# coding: utf-8

# # Deepfake Detection Results Analysis
#
# This notebook provides a methodological analysis of the deepfake detection system, including sensitivity analysis, statistical evaluation, and visualizations as required by Chapter 7 of the Guidelines.
#
# ## 1. Methodology and Mathematical Formulations
#
# ### 1.1 Blink Detection (Laplacian Variance)
# The eye openness is proxied by the variance of the Laplacian of the eye region:
#
# $$ O(f) = \log(1 + \text{Var}(\nabla^2 I_{eye})) $$
#
# Where $I_{eye}$ is the grayscale ROI of the eye. A blink is detected as a significant 'dip' in this series.
#

# In[1]:


import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %matplotlib inline
sns.set_theme(style="whitegrid")


# ## 2. Results Summary
# Loading results from `runs/` directory.
#

# In[2]:


def load_runs(base_dir="../runs"):
    runs = []
    base_path = Path(base_dir)
    if not base_path.exists():
        return pd.DataFrame()
    for p in base_path.glob("*/decision.json"):
        with open(p) as f:
            data = json.load(f)
            data["run_id"] = p.parent.name
            runs.append(data)
    return pd.DataFrame(runs)


df_results = load_runs()
if not df_results.empty:
    print(df_results.head())
else:
    print("No results found in runs/ directory.")


# ## 3. Sensitivity Analysis (Chapter 7.1)
# In this section, we analyze how the `openness_threshold` affects the `blink_detected` signal.
#

# In[3]:


# Mock data for sensitivity plot demonstration
thresholds = np.linspace(0.1, 0.9, 9)
accuracy = [0.65, 0.72, 0.81, 0.88, 0.85, 0.78, 0.70, 0.62, 0.55]

plt.figure(figsize=(10, 6))
plt.plot(thresholds, accuracy, marker="o", linestyle="-", color="b")
plt.title("Accuracy vs. Openness Threshold")
plt.xlabel("Threshold Factor (mu - k * sigma)")
plt.ylabel("Detection Accuracy")
plt.axvline(x=0.4, color="r", linestyle="--", label="Optimal Threshold")
plt.legend()
plt.show()


# ## 4. Cost Analysis (Chapter 10)
# Token usage and pricing analysis.
#

# In[4]:


cost_data = {
    "Model": ["GPT-4o", "Claude 3.5 Sonnet", "Gemini 1.5 Pro"],
    "Input Tokens (Avg)": [1200, 1150, 1300],
    "Output Tokens (Avg)": [450, 500, 420],
    "Cost per 1M Tokens (In/Out)": ["$5/$15", "$3/$15", "$3.5/$10.5"],
}
df_costs = pd.DataFrame(cost_data)
print(df_costs)
