# spam_lab.py

# =========================
# 1. Imports & global style
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    matthews_corrcoef,
    cohen_kappa_score,
    classification_report
)

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 120, "font.size": 12})

RANDOM_STATE = 42


# =========================
# 2. Load & clean the data
# =========================
# Change this path if needed
df = pd.read_csv(r"/Users/Downloads/spambase.csv")

# Clean columns:
df["type"] = df["type"].astype(str).str.strip().str.lower()

# Force binary class order: nonspam then spam
class_order = ["nonspam", "spam"]
df = df[df["type"].isin(class_order)].copy()
df["type"] = pd.Categorical(df["type"], categories=class_order, ordered=True)

# Features / target split
y = df["type"].cat.codes       # nonspam -> 0, spam -> 1
X = df.drop(columns=["type"])

print("Class counts:\n", df["type"].value_counts())
print("\nDataFrame info:")
print(df.info())
print("\nHead:\n", df.head())


# =========================================
# 3. Train/test split (70/30 stratified)
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTrain size:", X_train.shape, " Test size:", X_test.shape)


# ===================================================
# 4. 5-fold cross-validation setup (Stratified, k=5)
# ===================================================
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


# =====================================================================================
# 5. k-Nearest Neighbors model, tuning 'k' (neighbors = k)
# =====================================================================================
knn_pipeline = Pipeline(steps=[
    ("zv", VarianceThreshold(threshold=0.0)),        # drop zero-variance cols
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("knn", KNeighborsClassifier())
])

knn_param_grid = {
    "knn__n_neighbors": [3, 5, 7, 9, 11, 15]
}

knn_gridsearch = GridSearchCV(
    estimator=knn_pipeline,
    param_grid=knn_param_grid,
    scoring="accuracy",
    cv=cv5,
    n_jobs=-1,
    refit=True,
    return_train_score=True
)

knn_gridsearch.fit(X_train, y_train)

print("\n=== k-NN Cross-Validation Results ===")
cv_results_knn = pd.DataFrame(knn_gridsearch.cv_results_)
print(
    cv_results_knn[
        [
            "param_knn__n_neighbors",
            "mean_test_score",
            "std_test_score",
            "mean_train_score"
        ]
    ].sort_values(by="mean_test_score", ascending=False)
)

print("\nBest k-NN params:", knn_gridsearch.best_params_)
print("Best CV accuracy:", knn_gridsearch.best_score_)

# Final fitted k-NN model
knn_best = knn_gridsearch.best_estimator_


# ==========================================
# 6. Evaluate best k-NN on the held-out test
# ==========================================
y_pred_knn = knn_best.predict(X_test)
y_proba_knn = knn_best.predict_proba(X_test)[:, 1]  # prob of "spam" class 1

cm_knn = confusion_matrix(y_test, y_pred_knn, labels=[0,1])
print("\nConfusion Matrix (k-NN):\n", cm_knn)

# Quick confusion matrix plot (like autoplot(conf_mat))
plt.figure(figsize=(4,3))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_knn,
                              display_labels=class_order)
disp.plot(cmap="Blues", values_format="d", colorbar=False)
plt.title("k-NN — Confusion Matrix")
plt.tight_layout()
plt.show()

# Build labeled confusion matrix dataframe
tn, fp, fn, tp = cm_knn.ravel()
total = cm_knn.sum()

cm_knn_df = pd.DataFrame({
    "Truth":      ["nonspam","nonspam","spam","spam"],
    "Prediction": ["nonspam","spam","nonspam","spam"],
    "Count":      [tn, fp, fn, tp]
})

labels_map = {
    ("spam","spam"): "True Positive",
    ("nonspam","nonspam"): "True Negative",
    ("nonspam","spam"): "False Positive",
    ("spam","nonspam"): "False Negative"
}

cm_knn_df["Label"] = [
    labels_map.get((row["Truth"], row["Prediction"]), "")
    for _, row in cm_knn_df.iterrows()
]
cm_knn_df["Percent"] = np.round(cm_knn_df["Count"] / total * 100, 1)

print("\nDetailed CM (k-NN):\n", cm_knn_df)

# Heatmap with annotations like ggplot2 tile + geom_text
pivot_counts = cm_knn_df.pivot(
    index="Truth", columns="Prediction", values="Count"
)

plt.figure(figsize=(5,4))
sns.heatmap(
    pivot_counts,
    annot=False,
    fmt="d",
    cmap="Blues",
    cbar=True,
    linewidths=0.5,
    linecolor="white"
)

# Add multiline labels to each tile
for truth_i, truth_val in enumerate(pivot_counts.index):
    for pred_j, pred_val in enumerate(pivot_counts.columns):
        cell = cm_knn_df[
            (cm_knn_df["Truth"]==truth_val) &
            (cm_knn_df["Prediction"]==pred_val)
        ].iloc[0]
        lab = f"{cell['Label']}\n{cell['Count']}\n({cell['Percent']}%)"
        plt.text(
            pred_j+0.5,
            truth_i+0.5,
            lab,
            ha='center',
            va='center',
            fontsize=9,
            fontweight='bold'
        )

plt.title("k-NN — Confusion Matrix (Labeled)")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()
plt.show()

# Metrics to mimic your full_metrics()
acc_knn  = accuracy_score(y_test, y_pred_knn)
prec_knn = precision_score(y_test, y_pred_knn, pos_label=1)
rec_knn  = recall_score(y_test, y_pred_knn, pos_label=1)
f1_knn   = f1_score(y_test, y_pred_knn, pos_label=1)
spec_knn = tn / (tn + fp) if (tn + fp) > 0 else np.nan
mcc_knn  = matthews_corrcoef(y_test, y_pred_knn)
kap_knn  = cohen_kappa_score(y_test, y_pred_knn)
auc_knn  = roc_auc_score(y_test, y_proba_knn)

knn_metrics_out = pd.DataFrame({
    "metric": [
        "accuracy","precision","recall","f1",
        "specificity","MCC","kappa","roc_auc"
    ],
    "value": [
        acc_knn, prec_knn, rec_knn, f1_knn,
        spec_knn, mcc_knn, kap_knn, auc_knn
    ]
})
print("\nk-NN Metrics on Test Set:\n", knn_metrics_out)


# =====================================================================================
# 7. Radial Basis Function SVM (SVM with RBF kernel)
# =====================================================================================
svm_pipeline = Pipeline(steps=[
    ("zv", VarianceThreshold(threshold=0.0)),
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("svm", SVC(kernel="rbf", probability=True))
])

svm_param_grid = {
    "svm__C":     [0.1, 1, 10, 100],
    "svm__gamma": [0.01, 0.1, 1, 10]
}

svm_gridsearch = GridSearchCV(
    estimator=svm_pipeline,
    param_grid=svm_param_grid,
    scoring="accuracy",
    cv=cv5,
    n_jobs=-1,
    refit=True,
    return_train_score=True
)

svm_gridsearch.fit(X_train, y_train)

print("\n=== SVM (RBF) Cross-Validation Results ===")
cv_results_svm = pd.DataFrame(svm_gridsearch.cv_results_)
print(
    cv_results_svm[
        [
            "param_svm__C",
            "param_svm__gamma",
            "mean_test_score",
            "std_test_score",
            "mean_train_score"
        ]
    ].sort_values(by="mean_test_score", ascending=False)
)

print("\nBest SVM params:", svm_gridsearch.best_params_)
print("Best CV accuracy:", svm_gridsearch.best_score_)

svm_best = svm_gridsearch.best_estimator_


# ==========================================
# 8. Evaluate best SVM on the held-out test
# ==========================================
y_pred_svm  = svm_best.predict(X_test)
y_proba_svm = svm_best.predict_proba(X_test)[:, 1]

cm_svm = confusion_matrix(y_test, y_pred_svm, labels=[0,1])
print("\nConfusion Matrix (SVM RBF):\n", cm_svm)

plt.figure(figsize=(4,3))
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_svm,
                               display_labels=class_order)
disp2.plot(cmap="Blues", values_format="d", colorbar=False)
plt.title("SVM (RBF) — Confusion Matrix")
plt.tight_layout()
plt.show()

tn, fp, fn, tp = cm_svm.ravel()
total2 = cm_svm.sum()

cm_svm_df = pd.DataFrame({
    "Truth":      ["nonspam","nonspam","spam","spam"],
    "Prediction": ["nonspam","spam","nonspam","spam"],
    "Count":      [tn, fp, fn, tp]
})
cm_svm_df["Label"] = [
    labels_map.get((row["Truth"], row["Prediction"]), "")
    for _, row in cm_svm_df.iterrows()
]
cm_svm_df["Percent"] = np.round(cm_svm_df["Count"] / total2 * 100, 1)
print("\nDetailed CM (SVM):\n", cm_svm_df)

pivot_counts2 = cm_svm_df.pivot(
    index="Truth", columns="Prediction", values="Count"
)

plt.figure(figsize=(5,4))
sns.heatmap(
    pivot_counts2,
    annot=False,
    cmap="Blues",
    cbar=True,
    linewidths=0.5,
    linecolor="white"
)

for truth_i, truth_val in enumerate(pivot_counts2.index):
    for pred_j, pred_val in enumerate(pivot_counts2.columns):
        cell = cm_svm_df[
            (cm_svm_df["Truth"]==truth_val) &
            (cm_svm_df["Prediction"]==pred_val)
        ].iloc[0]
        lab = f"{cell['Label']}\n{cell['Count']}\n({cell['Percent']}%)"
        plt.text(
            pred_j+0.5,
            truth_i+0.5,
            lab,
            ha='center',
            va='center',
            fontsize=9,
            fontweight='bold'
        )

plt.title("SVM (RBF) — Confusion Matrix (Labeled)")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()
plt.show()

# Metrics (same style as k-NN)
acc_svm  = accuracy_score(y_test, y_pred_svm)
prec_svm = precision_score(y_test, y_pred_svm, pos_label=1)
rec_svm  = recall_score(y_test, y_pred_svm, pos_label=1)
f1_svm   = f1_score(y_test, y_pred_svm, pos_label=1)
spec_svm = tn / (tn + fp) if (tn + fp) > 0 else np.nan
mcc_svm  = matthews_corrcoef(y_test, y_pred_svm)
kap_svm  = cohen_kappa_score(y_test, y_pred_svm)
auc_svm  = roc_auc_score(y_test, y_proba_svm)

svm_metrics_out = pd.DataFrame({
    "metric": [
        "accuracy","precision","recall","f1",
        "specificity","MCC","kappa","roc_auc"
    ],
    "value": [
        acc_svm, prec_svm, rec_svm, f1_svm,
        spec_svm, mcc_svm, kap_svm, auc_svm
    ]
})
print("\nSVM Metrics on Test Set:\n", svm_metrics_out)


# =====================================
# 9. ROC curve for the SVM (like pROC)
# =====================================
fpr, tpr, thresholds = roc_curve(y_test, y_proba_svm, pos_label=1)

plt.figure(figsize=(4,4))
plt.plot(fpr, tpr, label=f"SVM (AUC = {auc_svm:.3f})")
plt.plot([0,1],[0,1], "--", color="gray")
plt.title("SVM (RBF) — ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


# =====================================================================================
# 10. Decision boundary visualization in 2D feature space
# =====================================================================================

def plot_svm_decision_boundary_2d(df_full,
                                  feature_x,
                                  feature_y,
                                  target_col="type",
                                  C=1.0,
                                  gamma=0.1):
    """
    df_full: dataframe with target_col containing 'nonspam'/'spam'
    feature_x, feature_y: column names to visualize in 2D
    """

    viz = df_full[[target_col, feature_x, feature_y]].copy()
    viz[target_col] = viz[target_col].astype("category")
    viz["y_bin"] = viz[target_col].cat.codes  # nonspam=0, spam=1

    # scale each chosen feature to [0,1]
    for col in [feature_x, feature_y]:
        col_min = viz[col].min()
        col_max = viz[col].max()
        viz[col + "_scaled"] = (viz[col] - col_min) / (col_max - col_min + 1e-9)

    X_small = viz[[feature_x + "_scaled", feature_y + "_scaled"]].values
    y_small = viz["y_bin"].values

    svm_small = SVC(
        kernel="rbf",
        C=C,
        gamma=gamma,
        probability=False
    )
    svm_small.fit(X_small, y_small)

    # grid in [0,1] x [0,1]
    grid_x, grid_y = np.meshgrid(
        np.linspace(0, 1, 300),
        np.linspace(0, 1, 300)
    )
    grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]

    grid_pred = svm_small.predict(grid_points)
    decision_vals = svm_small.decision_function(grid_points)

    Z_class = grid_pred.reshape(grid_x.shape)
    Z_decision = decision_vals.reshape(grid_x.shape)

    plt.figure(figsize=(5,4))

    # background shading: predicted class
    plt.contourf(
        grid_x, grid_y, Z_class,
        alpha=0.25,
        levels=[-0.5,0.5,1.5],
        colors=["#74add1","#f46d43"]
    )

    # dashed decision boundary (margin ~ 0)
    plt.contour(
        grid_x, grid_y, Z_decision,
        levels=[0],
        colors="black",
        linestyles="--",
        linewidths=1
    )

    # sample data so plot doesn't explode
    sample_idx = np.random.choice(len(viz), size=min(300, len(viz)), replace=False)
    sample_plot = viz.iloc[sample_idx]

    plt.scatter(
        sample_plot[feature_x + "_scaled"],
        sample_plot[feature_y + "_scaled"],
        c=sample_plot["y_bin"],
        cmap=ListedColormap(["#2b83ba", "#d73027"]),
        s=12,
        alpha=0.7,
        edgecolor="none",
        label="Actual data"
    )

    # mark SVM support vectors
    sv = svm_small.support_vectors_
    plt.scatter(
        sv[:,0], sv[:,1],
        marker="x",
        s=30,
        linewidths=1.2,
        c="black",
        label="Support Vectors"
    )

    plt.title(f"SVM Decision Boundary (RBF)\n{feature_x} vs {feature_y}")
    plt.xlabel(f"{feature_x} (scaled 0–1)")
    plt.ylabel(f"{feature_y} (scaled 0–1)")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_capital_features(df_full, target_col="type", C=1.0, gamma=1.0):
    """
    Replicates the 'capitalAve vs capitalTotal' style plot in R,
    using your actual column names: capitalAve, capitalTotal.
    """

    # Build working df
    viz = df_full[[target_col, "capitalAve", "capitalTotal"]].copy()
    viz[target_col] = viz[target_col].astype("category")
    viz["y_bin"] = viz[target_col].cat.codes  # nonspam=0, spam=1

    # log1p transform, then scale [0,1]
    viz["x"] = np.log1p(viz["capitalAve"])
    viz["y"] = np.log1p(viz["capitalTotal"])

    for col in ["x","y"]:
        cmin = viz[col].min()
        cmax = viz[col].max()
        viz[col] = (viz[col] - cmin) / (cmax - cmin + 1e-9)

    X_small = viz[["x","y"]].values
    y_small = viz["y_bin"].values

    svm_small = SVC(
        kernel="rbf",
        C=C,
        gamma=gamma
    )
    svm_small.fit(X_small, y_small)

    grid_x, grid_y = np.meshgrid(
        np.linspace(0,1,300),
        np.linspace(0,1,300)
    )
    grid_pts = np.c_[grid_x.ravel(), grid_y.ravel()]
    grid_pred = svm_small.predict(grid_pts)
    decision_vals = svm_small.decision_function(grid_pts)

    Z_class = grid_pred.reshape(grid_x.shape)
    Z_decision = decision_vals.reshape(grid_x.shape)

    plt.figure(figsize=(5,4))

    plt.contourf(
        grid_x, grid_y, Z_class,
        alpha=0.25,
        levels=[-0.5,0.5,1.5],
        colors=["#74add1","#f46d43"]
    )

    plt.contour(
        grid_x, grid_y, Z_decision,
        levels=[0],
        colors="black",
        linewidths=0.8
    )

    sample_idx = np.random.choice(len(viz), size=min(300, len(viz)), replace=False)
    sample_plot = viz.iloc[sample_idx]

    plt.scatter(
        sample_plot["x"],
        sample_plot["y"],
        c=sample_plot["y_bin"],
        cmap=ListedColormap(["#2b83ba", "#d73027"]),
        s=12,
        alpha=0.7,
        edgecolor="none"
    )

    plt.title("SVM Decision Boundary (RBF Kernel)")
    plt.xlabel("Avg length of uppercase runs (scaled)")
    plt.ylabel("Total uppercase characters (scaled)")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()


# Example visualizations:
# (1) make vs address (word frequency features in your dataset)
if "make" in df.columns and "address" in df.columns:
    plot_svm_decision_boundary_2d(
        df_full=df,
        feature_x="make",
        feature_y="address",
        target_col="type",
        C=1.0,
        gamma=0.1
    )

# (2) capitalAve vs capitalTotal (stylistic features)
if "capitalAve" in df.columns and "capitalTotal" in df.columns:
    plot_capital_features(df, target_col="type", C=1.0, gamma=1.0)


# ===============================
# 11. Bonus: classification report
# ===============================
print("\nClassification report (SVM on Test):")
print(classification_report(y_test, y_pred_svm, target_names=class_order))

print("\nDone.")
