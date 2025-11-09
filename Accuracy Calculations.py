TP = 481
TN = 751
FN = 63
FP = 86

accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1_Score = 2 * (Precision * Recall) / (Precision + Recall)
print(f"Precision: {Precision:.4f} ({Precision*100:.2f}% )")
print(f"Recall: {Recall:.4f} ({Recall*100:.2f}% )")
print(f"F1 Score: {F1_Score:.4f} ({F1_Score*100:.2f}% )")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}% )")
