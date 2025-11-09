import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, cohen_kappa_score, roc_auc_score, roc_curve
)

#load the dataset
df = pd.read_csv("/Users/sophiewalker/Downloads/spambase.csv")

#Clean taret column 
df["type"] = df["type"].str.lower().str.strip()
df["type"] = df["type"].replace({"nonspam": 0, "spam": 1})
print("The shape of the database", df.shape)
print("The number of observations(Rows) is: ", df.shape[0])
print("The number of attributes (Columns) is: ", df.shape[1])

#Partition the dataset
x = df.drop(columns="type")
y = df["type"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, stratify=y, random_state=42
)

print(f"Training set: {x_train.shape[0]} rows")
print(f"Testing set: {x_test.shape[0]} rows")

tree = DecisionTreeClassifier(random_state=42)
tree.fit(x_train, y_train)

plt.figure(figsize=(20,10))
plot_tree(
    tree,
    feature_names=list(x.columns),
    class_names=["Nonspam", "Spam"],
    filled=True,
    rounded=True,
    fontsize=12,
    max_depth=3
)
plt.title("Decision Tree - spam vs. nonspam", fontsize=16, weight="bold")
plt.show()

importances = pd.Series(tree.feature_importances_, index=x.columns)
importances = importances.sort_values(ascending=False)
print("Top 10 Important Features:")
print(importances.head(10))

y_pred = tree.predict(x_test)
y_prob = tree.predict_proba(x_test)[:, 1]

prediction = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred,
    "Pro_Spam": y_prob
})
print(prediction.head())

cm = confusion_matrix(y_test, y_pred)
labels = np.array([["True Negative", "False Positive"], ["False Negative", "True Positive"]])

cm_percent = cm / cm.sum() * 100

annot = np.empty_like(cm).astype(str)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        annot[i, j] = f"{labels[i, j]}\n{cm[i, j]}\n{cm_percent[i, j]:.1f}%"

plt.figure(figsize=(6,5))
sns.heatmap(cm,annot=annot, fmt="", cmap="Reds",
            xticklabels=["Nonspam", "Spam"],
            yticklabels=["Nonspam", "Spam"],
            cbar_kws={"label": "Count"})

plt.title("Confusion Matrix - Decision Tree Spam Classifer", fontsize=14, weight="bold")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()
plt.show()

