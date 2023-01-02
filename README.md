# tf_tools
tools for deep learning with transcription factors


#Evaluation example
from evaluation_metrics import EvaluationMetrics

true_labels = [[1, 0, 1], [0, 1, 1]]
predicted_labels = [[0.8, 0.2, 0.6], [0.2, 0.9, 0.7]]
objective = "binary"

mean, std = EvaluationMetrics.calculate_metrics(true_labels, predicted_labels, objective)

Available Metrics
The following metrics are available for calculation:

Binary classification:
Accuracy
Area Under the Receiver Operating Characteristic curve (AUROC)
Area Under the Precision-Recall curve (AUPR)
Categorical classification:
Accuracy
Area Under the Receiver Operating Characteristic curve (AUROC)
Area Under the Precision-Recall curve (AUPR)
Regression:
Pearson correlation coefficient
R-squared
Slope
