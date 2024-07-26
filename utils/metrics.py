import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import beautifultable as bt
from matplotlib import pyplot as plt


def get_mse(y_pred, y_target):
    non_zero_pos = np.where(y_target != 0)
    return np.mean((y_pred[non_zero_pos] - y_target[non_zero_pos]) ** 2)

def get_multi_confusion_matrix(y_target, y_pred, labels=None):
    # 用beautiful table打印混淆矩阵
    cm = confusion_matrix(y_target, y_pred)
    cm_tb = bt.BeautifulTable()
    cm_tb.column_headers = labels
    for i in range(len(cm)):
        cm_tb.append_row(cm[i])
    cm_tb.rows.header = labels
    
    return cm, cm_tb

def get_confusion_matrix_plot(confusion_matrix, classes=None, normalize=False, path=None):
    # 画混淆矩阵
    normalized_cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    
    if normalize:
        disp = ConfusionMatrixDisplay(confusion_matrix=normalized_cm, display_labels=classes)
        disp.plot(
            include_values=True,            # 混淆矩阵每个单元格上显示具体数值
            cmap='Blues',                    # 颜色 OrRd, tab20b, Blues
            ax=None,                        # 
            xticks_rotation=30,             # x轴标签旋转角度
            values_format='.2f'             # 显示的数值格式
        )
    else :
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=classes)
        disp.plot(
            include_values=True,            # 混淆矩阵每个单元格上显示具体数值
            cmap='OrRd',                    # 颜色
            ax=None,                        # 
            xticks_rotation=30,             # x轴标签旋转角度
            values_format='d'               # 显示的数值格式
        )
    plt.tight_layout()
    plt.savefig(path, dpi=300, format='pdf') if path else None
    
    return plt

def get_multi_evaluation(y_target, y_pred):
    # 用beautiful table打印评价指标
    acc = accuracy_score(y_target, y_pred)
    f1 = f1_score(y_target, y_pred, average='weighted')
    precision = precision_score(y_target, y_pred, average='weighted')
    recall = recall_score(y_target, y_pred, average='weighted')
    
    place = 8
    
    # 格式化输出
    acc_str = f"{acc:.{place}f}"
    f1_str = f"{f1:.{place}f}"
    precision_str = f"{precision:.{place}f}"
    recall_str = f"{recall:.{place}f}"
    
    eval_tb = bt.BeautifulTable(precision=place)
    eval_tb.column_headers = ['ACC', 'F1', 'Precision', 'Recall']
    eval_tb.append_row([acc_str, f1_str, precision_str, recall_str])
    
    return acc, f1, precision, recall, eval_tb

def get_binary_evaluation(y_target, y_pred):
    # 用beautiful table打印评价指标
    acc = accuracy_score(y_target, y_pred)
    f1 = f1_score(y_target, y_pred, average='macro')
    precision = precision_score(y_target, y_pred, average='macro')
    recall = recall_score(y_target, y_pred, average='macro')
    fpr = get_fpr(confusion_matrix(y_target, y_pred))
    auc = roc_auc_score(y_target, y_pred)
    
    place = 8
    
    # 格式化输出
    acc_str = f"{acc:.{place}f}"
    f1_str = f"{f1:.{place}f}"
    precision_str = f"{precision:.{place}f}"
    recall_str = f"{recall:.{place}f}"
    fpr_str = f"{fpr:.{place}f}"
    auc_str = f"{auc:.{place}f}"
    
    eval_tb = bt.BeautifulTable(precision=place)
    eval_tb.column_headers = ['ACC', 'F1', 'Precision', 'Recall', 'FPR', 'AUC']
    eval_tb.append_row([acc_str, f1_str, precision_str, recall_str, fpr_str, auc_str])
    
    return acc, f1, precision, recall, fpr, auc, eval_tb

def get_fpr(confusion_matrix):
    return confusion_matrix[0][1] / (confusion_matrix[0][1] + confusion_matrix[1][1])