import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns #importするだけでスタイルがSeabornになる

titanic = sns.load_dataset("titanic")
print(titanic)
sns.pointplot(x="class", y="survived", hue="sex", data=titanic,
              palette={"male": "g"},
              markers=["o"], linestyles=["-"]);

plt.show()
