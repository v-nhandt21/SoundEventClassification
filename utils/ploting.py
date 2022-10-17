# Plot ROC multisclass: https://stackoverflow.com/questions/56090541/how-to-plot-precision-and-recall-of-multiclass-classifier

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn
import pandas as pd

def gen_example():
    mnist = fetch_openml("mnist_784")
    y = mnist.target
    y = y.astype(np.uint8)
    n_classes = len(set(y))
    Y = label_binarize(y, classes=[0,1, 2, 3,4,5,6,7,8,9])
    X_train, X_test, y_train, y_test_label_target = train_test_split(mnist.data, Y, random_state = 42)
    clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=50, max_depth=3, random_state=0))
    clf.fit(X_train, y_train)
    y_score_predict = clf.predict_proba(X_test)

    return y_test_label_target, y_score_predict

############################################# precision recall curve
def plot_precision_recall_curve(y_test_label_target, y_score_predict, labels, save_file):
    precision = dict()
    recall = dict()
    for i in range(len(labels)):
        precision[i], recall[i], _ = precision_recall_curve(y_test_label_target[:, i],
                                                            y_score_predict[:, i])
        plt.plot(recall[i], precision[i], lw=1, label='class {}'.format(str(labels[i])))
        
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title("precision vs. recall curve")
    plt.savefig(save_file)
    # plt.show()
    return plt
############################################ roc curve
def plot_roc_curve(y_test_label_target, y_score_predict, labels, save_file):
    fpr = dict()
    tpr = dict()

    for i in range(len(labels)):
        fpr[i], tpr[i], _ = roc_curve(y_test_label_target[:, i], y_score_predict[:, i])
        plt.plot(fpr[i], tpr[i], lw=0.5, label='class {}'.format(str(labels[i])))

    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.legend(loc="best")
    plt.title("ROC curve")
    plt.savefig(save_file)
    # plt.show()
    return plt

def report(y_true, y_pred, labels, save_file):
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)

    print(labels)
    labels = [0,1, 2, 3,4,5,6,7,8,9]

    print(classification_report(y_true, y_pred, labels=labels))

    cfmt = confusion_matrix(y_true, y_pred)

    df_cm = pd.DataFrame(cfmt, index = labels, columns = labels)
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(save_file)

if __name__ == '__main__':
    y_test_label_target, y_score_predict = gen_example()
    y_test_label_target, y_score_predict = y_test_label_target[:100], y_score_predict[:100]

    print(y_test_label_target)
    print(y_score_predict)

    print(y_test_label_target.shape)
    print(y_score_predict.shape)

    plot_precision_recall_curve(y_test_label_target, y_score_predict, [0,1, 2, 3,4,5,6,7,8,9], "Outdir/test_prc.png")
    plot_roc_curve(y_test_label_target, y_score_predict, [0,1, 2, 3,4,5,6,7,8,9], "Outdir/test_roc.png")

    report(y_test_label_target, y_score_predict, [0,1, 2, 3,4,5,6,7,8,9], "Outdir/test_confusion_matrix.png")
