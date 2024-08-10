import numpy as np

# Returns TPR, TNR, FPR, FNR
def expand_classification_rates(tpr, fpr):
    tnr = 1 - fpr
    fnr = 1 - tpr
    return tpr, tnr, fpr, fnr

def classification_rates_to_nominal(tpr, tnr, fpr, fnr, class1_num, class0_num):
    tp = tpr * class1_num
    tn = tnr * class0_num
    fp = fpr * class0_num
    fn = fnr * class1_num
    
    return tp, tn, fp, fn

def balanced_accuracy_binary(tpr, tnr):
    return (tpr + tnr) / 2

# How likely positive cases are in test set
def prevalence(tp, fn, pop_size):
    return (tp + fn) / pop_size

# How likely classifier predicts positive for test set
def bias(tp, fp, pop_size):
    return (tp + fp) / pop_size

# Class 1 gives precision, class 0 gives true negative accuracy or inverse precision
def class_precision(true_rate, false_rate, class1_num, class0_num, class_ = 1):
    if class_ == 1:
        ratio = class0_num / class1_num
    elif class_ == 0:
        ratio = class1_num / class0_num
    else:
        print("class_ must be 1 or 0")
        return
    
    return true_rate / (true_rate + (ratio * false_rate))

# Measure of how informed a system is about positives and negatives
# Ranges from -1 to +1
# Magnitude indicates informedness at predicting positives and negatives, polairty indicates correctness
# E.g., +1 indicates TP = RP, FP = 0, -1 indicates TP = 0, FP = RN
# Recall + Inverse Recall - 1 = TPR + TNR - 1
def informedness(tpr, tnr):
    return tpr + tnr - 1

# Measure of trustworthiness of predictions by the system
# Ranges from -1 to +1
# Magnitude indicates trustworthiness in predictions, polarity indicates correctness
# E.g., +1 indicates TP = PP, FN = 0, -1 indicates TP = 0, FN = PN
# Precision + Inverse Precision - 1
# https://rvprasad.medium.com/informedness-and-markedness-20e3f54d63bc
# https://www.researchgate.net/publication/228529307_Evaluation_From_Precision_Recall_and_F-Factor_to_ROC_Informedness_Markedness_Correlation
def markedness(tpr, fpr, class1_num, class0_num):
    tpr, tnr, fpr, fnr = expand_classification_rates(tpr, fpr)
    precision = class_precision(tpr, fpr, class1_num, class0_num, class_ = 1)
    inverse_precision = class_precision(tnr, fnr, class1_num, class0_num, class_ = 0)
    
    return precision + inverse_precision - 1

# Matthews Correlation Coefficient measures the correlation of the true classes with predicted labels
# Equivalent to the geometric mean of informednes and markedness
# When MCC is high, the classifier is well informed and trustworthy.
# MCC > informedness and markedness when dataset imbalance > prediction imbalance
# https://biodatamining.biomedcentral.com/articles/10.1186/s13040-021-00244-z
def matthews_cc(tp, tn, fp, fn):
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    return numerator / denominator