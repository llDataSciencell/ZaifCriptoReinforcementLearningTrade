import seaborn as sns
sns.set(style="ticks")
import numpy as np
# Load the example dataset for Anscombe's quartet
df = np.array([1,1,2,2,3,3])

# Show the results of a linear regression within each dataset
sns.lmplot(x="x", y="y", col="dataset", hue="dataset", data=df,
           col_wrap=2, ci=None, palette="muted",
           scatter_kws={"s": 50, "alpha": 1})
