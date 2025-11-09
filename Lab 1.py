#Set up
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

#Load the data
iris = datasets.load_iris(as_frame=True)
df = iris.frame

#rename columns to match R's style
df.columns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "Species"]

#replacing numaric species codes with text labels
df["Species"] = df["Species"].map({0: "setosa", 1: "versicolor", 2: "virginica"})   
print(df.head())
print(df.info())


#-----Descriptive Statistics-----
#Similar to summarry(iris) in R
summary_stats = df.describe(include="all").T
print(summary_stats)

# Mean by species (Similar to dplyr group_by + summaries
species_means = df.groupby("Species").mean(numeric_only=True)
print("\nmean by Species:\n", species_means)

#-----Univariate Visualizations-----
sns.histplot(data=df, x="Sepal.Length", hue="Species", bins=20, kde=True, alpha=0.6)
plt.title("Distribution of sepal length by Species")
plt.show()

sns.boxplot(data=df, x="Species", y="Sepal.Width", palette="Set2")
plt.title("Sepal Width by Species")
plt.show()

#-----Bivariate Visualizations-----
sns.scatterplot(data=df, x="Sepal.Length", y="Sepal.Width", hue="Species", style="Species")
plt.title("Sepal Dimensions by Species")
plt.show()

sns.scatterplot(data=df, x="Petal.Length", y="Petal.Width", hue="Species", style="Species")
plt.title("Petal Dimensions by Species")
plt.show()

#-----Correlation Heatmap-----
corr = df.select_dtypes("number").corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", square=True)
plt.title("Feature Correlations (R-style output)")
plt.show()

#-----Pairplot-----
sns.pairplot(df, hue="Species", corner=True)
plt.suptitle("Pairwise feature relationships", y=1.02)
plt.show()

