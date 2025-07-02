import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.feature_selection import mutual_info_classif

# Тренира Logistic Regression модел върху данните
# X - 2D numpy масив с входни характеристики
# y - 1D numpy масив с класове (0/1)
def train_logistic_regression(X, y):
    model = LogisticRegression(solver='liblinear')
    model.fit(X, y)
    return model

# Изчислява метриките за даден модел и данни
def calculate_metrics(model, X, y_true):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    metrics = {
        'logloss': log_loss(y_true, y_proba),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    return metrics

# Изчислява information gain (mutual information) за всяка характеристика
def calculate_information_gain(X, y, feature_names=None):
    info_gain = mutual_info_classif(X, y, discrete_features='auto')
    if feature_names is not None:
        return dict(zip(feature_names, info_gain))
    return info_gain.tolist() 