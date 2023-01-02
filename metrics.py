import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, roc_auc_score


class EvaluationMetrics:
    """
    A class for calculating various evaluation metrics for classification and regression tasks.
    """
    
    @staticmethod
    def accuracy(true_labels, predicted_labels):
        """
        Calculates the accuracy for each label column in true_labels and predicted_labels.
        
        Parameters:
        true_labels (ndarray): 2D array of true labels with shape (num_samples, num_labels)
        predicted_labels (ndarray): 2D array of predicted labels with shape (num_samples, num_labels)
        
        Returns:
        ndarray: 1D array of accuracy scores with shape (num_labels,)
        """
        num_labels = true_labels.shape[1]
        accuracy_by_label = np.zeros((num_labels))
        for i in range(num_labels):
            accuracy_by_label[i] = accuracy_score(true_labels[:, i], np.round(predicted_labels[:, i]))
        return accuracy_by_label

    @staticmethod
    def auroc(true_labels, predicted_labels):
        """
        Calculates the Area Under the Receiver Operating Characteristic curve (AUROC) 
        for each label column in true_labels and predicted_labels.
        
        Parameters:
        true_labels (ndarray): 2D array of true labels with shape (num_samples, num_labels)
        predicted_labels (ndarray): 2D array of predicted labels with shape (num_samples, num_labels)
        
        Returns:
        ndarray: 1D array of AUROC scores with shape (num_labels,)
        """
        num_labels = true_labels.shape[1]
        auroc_by_label = np.zeros((num_labels))
        for i in range(num_labels):
            fpr, tpr, thresholds = roc_curve(true_labels[:, i], predicted_labels[:, i])
            score = auc(fpr, tpr)
            auroc_by_label[i] = score
        return auroc_by_label
    
    @staticmethod
    def auprc(true_labels, predicted_labels):
        """
        Calculates the Area Under the Precision-Recall curve (AUPR) 
        for each label column in true_labels and predicted_labels.
        
        Parameters:
        true_labels (ndarray): 2D array of true labels with shape (num_samples, num_labels)
        predicted_labels (ndarray): 2D array of predicted labels with shape (num_samples, num_labels)
        
        Returns:
        ndarray: 1D array of AUPR scores with shape (num_labels,)
        """
        num_labels = true_labels.shape[1]
        aupr_by_label = np.zeros((num_labels))
        for i in range(num_labels):
            precision, recall, thresholds = precision_recall_curve(true_labels[:, i], predicted_labels[:, i])
            score = auc(recall, precision)
            aupr_by_label[i] = score
        return aupr_by_label

    @staticmethod
    def pearsonr(true_labels, predicted_labels, mask_value=None):
        """
        Calculates the Pearson correlation coefficient for each label column in true_labels and predicted_labels.
        
        Parameters:
        true_labels (ndarray): 2D array of true labels with shape (num_samples, num_labels)
        predicted_labels (ndarray): 2D array of predicted labels with shape (num_samples, num_labels)
        mask_value (float or int, optional): Value to be masked in true_labels. Default is None.
        
        Returns:
        ndarray: 1D array of Pearson correlation coefficients with shape (num_labels,)
        """
        num_labels = true_labels.shape[1]
        pearsonr_by_label = np.zeros((num_labels))
        for i in range(num_labels):
            if mask_value:
                indices = np.where(true_labels[:, i] != mask_value)[0]
                pearsonr_by_label[i] = stats.pearsonr(true_labels[indices, i], predicted_labels[indices, i])[0]
            else:
                pearsonr_by_label[i] = stats.pearsonr(true_labels[:, i], predicted_labels[:, i])[0]
        return pearsonr_by_label

    @staticmethod
    def compute_rsquare_and_slope_by_label(true_labels, predicted_labels):
        """
        Calculates the R-squared and slope for each label column in true_labels and predicted_labels.
        
        Parameters:
        true_labels (ndarray): 2D array of true labels with shape (num_samples, num_labels)
        predicted_labels (ndarray): 2D array of predicted labels with shape (num_samples, num_labels)
        
        Returns:
        tuple: 
            ndarray: 1D array of R-squared values with shape (num_labels,)
            ndarray: 1D array of slope values with shape (num_labels,)
        """
        num_labels = true_labels.shape[1]
        rsquare_by_label = np.zeros((num_labels))
        slope_by_label = np.zeros((num_labels))
        for i in range(num_labels):
            y = true_labels[:, i]
            X = predicted_labels[:, i]
            m = np.dot(X, y) / np.dot(X, X)
            resid = y - m*X
            ym = y - np.mean(y)
            rsquare_by_label[i] = 1 - np.dot(resid.T, resid) / np.dot(ym.T, ym)
            slope_by_label[i] = m
        return rsquare_by_label, slope_by_label


    @staticmethod
    def calculate_metrics(true_labels, predicted_labels, objective):
        """
        Calculates various evaluation metrics for classification and regression tasks.
        
        Parameters:
        true_labels (ndarray): 2D array of true labels with shape (num_samples, num_labels)
        predicted_labels (ndarray): 2D array of predicted labels with shape (num_samples, num_labels)
        objective (str): Type of task. Valid values are 'binary', 'categorical', and 'squared_error'.

        Returns:
        list: 
            list: Mean values for each metric
            list: Standard deviation values for each metric
        """

        mean = []
        std = []
        if objective == "binary":
            acc = EvaluationMetrics.accuracy(true_labels, predicted_labels)
            auc_roc = EvaluationMetrics.auroc(true_labels, predicted_labels)
            auc_pr = EvaluationMetrics.auprc(true_labels, predicted_labels)
            mean = [np.nanmean(acc), np.nanmean(auc_roc), np.nanmean(auc_pr)]
            std = [np.nanstd(acc), np.nanstd(auc_roc), np.nanstd(auc_pr)]
        elif objective == "categorical":
            acc = np.mean(np.equal(np.argmax(true_labels, axis=1), np.argmax(predicted_labels, axis=1)))
            auc_roc = EvaluationMetrics.auroc(true_labels, predicted_labels)
            auc_pr = EvaluationMetrics.auprc(true_labels, predicted_labels)
            mean = [np.nanmean(acc), np.nanmean(auc_roc), np.nanmean(auc_pr)]
            std = [np.nanstd(acc), np.nanstd(auc_roc), np.nanstd(auc_pr)]
        elif objective == "squared_error":
            corr = EvaluationMetrics.pearsonr(true_labels, predicted_labels)
            rsqr, slope = EvaluationMetrics.compute_rsquare_and_slope_by_label(true_labels, predicted_labels)
            mean = [np.nanmean(corr), np.nanmean(rsqr), np.nanmean(slope)]
            std = [np.nanstd(corr), np.nanstd(rsqr), np.nanstd(slope)]
        else:
            raise ValueError("Invalid objective. Valid values are 'binary', 'categorical', and 'squared_error'.")
            
        return [mean, std]

        

        
