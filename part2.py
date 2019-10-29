import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
import imblearn as il
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")


    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

data = pd.read_csv("creditcard.csv")
count_classes = pd.value_counts(data['Class'], sort = True).sort_index()
print(count_classes)

plt.figure(figsize=(10,5))
count_classes.plot(kind = 'bar')
plt.title("Fraud class histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()

print(data.head())
data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
data = data.drop(['Time','Amount'],axis=1)
cols = list(data.columns.values)
print(cols)
data = data[
    ['normAmount', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15',
     'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Class']]
X = data.iloc[:, data.columns != 'Class']
Y = data.iloc[:, data.columns == 'Class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=0)

sm = SMOTE(random_state=12, ratio = 'auto', kind = 'regular')
x_train_res, y_train_res = sm.fit_sample(x_train, y_train)
print('Resampled dataset shape {}'.format(Counter(y_train_res)))



print("------------------------------------------------------------------")
sm_logr = LogisticRegression(random_state=0)
sm_logr.fit(x_train_res, y_train_res)
y_pred=sm_logr.predict(X_test)

print('Validation Results for logistic regression SMOTE:')
print('Accuracy: ', sm_logr.score(x_val, y_val))
print('Recall Score: ', recall_score(y_val, sm_logr.predict(x_val)))
print('Test Results')
print('Accuracy: ', sm_logr.score(X_test, Y_test))
print('Recall Score: ', recall_score(Y_test, sm_logr.predict(X_test)))



cnf_matrix = confusion_matrix(Y_test, y_pred)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Fraud','Non fraud'],title='Confusion matrix, without normalization')
tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)

Accuracy = (tn+tp)*100/(tp+tn+fp+fn)
print("Accuracy"+str(Accuracy))

Precision = tp/(tp+fp)
print("Precision {:0.2f}".format(Precision))

Recall = tp/(tp+fn)
print("Recall {:0.2f}".format(Recall))





Y_score = sm_logr.decision_function(X_test)

average_precision = average_precision_score(Y_test, Y_score)

precision, recall, _ = precision_recall_curve(Y_test, Y_score)

plt.figure(figsize=(8, 8))
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AUC={0:0.2f}'.format(average_precision))
plt.show()



print("----------------------------------------------------------------")
sm_rf = RandomForestClassifier(random_state=12)
sm_rf.fit(x_train_res, y_train_res)
y_pred=sm_rf.predict(X_test)
print('Validation Results for random classifier with smote')
print('Accuracy: ', sm_rf.score(x_val, y_val))
print('Recall Score: ', recall_score(y_val, sm_rf.predict(x_val)))
print('Test Results')
print('Accuracy: ', sm_rf.score(X_test, Y_test))
print('Recall Score: ', recall_score(Y_test, sm_rf.predict(X_test)))

cnf_matrix = confusion_matrix(Y_test, y_pred)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Fraud','Non fraud'],title='Confusion matrix, without normalization')
tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)

Accuracy = (tn+tp)*100/(tp+tn+fp+fn)
print("Accuracy"+str(Accuracy))

Precision = tp/(tp+fp)
print("Precision {:0.2f}".format(Precision))

Recall = tp/(tp+fn)
print("Recall {:0.2f}".format(Recall))

print("-----------------------------------------------------------------")

se = SMOTEENN(random_state=12, ratio = 'auto')
x_train_res, y_train_res = se.fit_sample(x_train, np.ravel(y_train))
print('Resampled dataset shape {}'.format(Counter(y_train_res)))

print("-----------------------------------------------------------------")
se_logr = LogisticRegression(random_state=0)
se_logr.fit(x_train_res, y_train_res)
y_pred=se_logr.predict(X_test)

print("Results for SMOTEENN logistic regression")
print('Validation Results')
print('Accuracy: ', se_logr.score(x_val, y_val))
print('Recall Score: ', recall_score(y_val, se_logr.predict(x_val)))

print('Test Results')
print('Accuracy: ', se_logr.score(X_test, Y_test))
print('Recall Score: ', recall_score(Y_test, se_logr.predict(X_test)))

cnf_matrix = confusion_matrix(Y_test, y_pred)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Fraud','Non fraud'],title='Confusion matrix, without normalization')
tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)

Accuracy = (tn+tp)*100/(tp+tn+fp+fn)
print("Accuracy"+str(Accuracy))

Precision = tp/(tp+fp)
print("Precision {:0.2f}".format(Precision))

Recall = tp/(tp+fn)
print("Recall {:0.2f}".format(Recall))

Y_score = se_logr.decision_function(X_test)
average_precision = average_precision_score(Y_test, Y_score)
precision, recall, _ = precision_recall_curve(Y_test, Y_score)
plt.figure(figsize=(8, 8))
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AUC={0:0.2f}'.format(average_precision))
plt.show()


print("-----------------------------------------------------------------")

se_rf = RandomForestClassifier(random_state=12)
se_rf.fit(x_train_res, y_train_res)
y_pred=se_rf.predict(X_test)

print("Results for SMOTEENN Random forest classification")
print('Validation Results')
print('Accuracy: ', se_rf.score(x_val, y_val))
print('Recall Score: ', recall_score(y_val, se_rf.predict(x_val)))
print('Test Results')
print('Accuracy: ', se_rf.score(X_test, Y_test))
print('Recall Score: ', recall_score(Y_test, se_rf.predict(X_test)))

cnf_matrix = confusion_matrix(Y_test, y_pred)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Fraud','Non fraud'],title='Confusion matrix, without normalization')
tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)

Accuracy = (tn+tp)*100/(tp+tn+fp+fn)
print("Accuracy"+str(Accuracy))

Precision = tp/(tp+fp)
print("Precision {:0.2f}".format(Precision))

Recall = tp/(tp+fn)
print("Recall {:0.2f}".format(Recall))


print("-----------------------------------------------------------------")

kn=KNeighborsClassifier(n_neighbors=7)
kn.fit(x_train_res, y_train_res)
y_pred=kn.predict(X_test)

print("Results for SMOTEENN Random forest classification")
print('Validation Results')
print('Accuracy: ', kn.score(x_val, y_val))
print('Recall Score: ', recall_score(y_val, kn.predict(x_val)))
print('Test Results')
print('Accuracy: ', kn.score(X_test, Y_test))
print('Recall Score: ', recall_score(Y_test, kn.predict(X_test)))

cnf_matrix = confusion_matrix(Y_test, y_pred)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Fraud','Non fraud'],title='Confusion matrix, without normalization')
tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)

Accuracy = (tn+tp)*100/(tp+tn+fp+fn)
print("Accuracy"+str(Accuracy))

Precision = tp/(tp+fp)
print("Precision {:0.2f}".format(Precision))

Recall = tp/(tp+fn)
print("Recall {:0.2f}".format(Recall))