from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import hamming_loss


def model_report(y_pred, y_true):
    clf_report = classification_report(y_true, y_pred)

    report = ''
    report += 'AUROC: {} \n'.format(roc_auc_score(y_true, y_pred))
    report += 'Accuracy: {} \n'.format(accuracy_score(y_true, y_pred))
    report += 'Average precision score: {} \n'.format(
        average_precision_score(y_true, y_pred))
    report += 'F1: {} \n'.format(f1_score(y_true, y_pred))
    report += 'Hamming loss: {} \n'.format(hamming_loss(y_true, y_pred))
    report += '\n'
    report += clf_report

    return report
