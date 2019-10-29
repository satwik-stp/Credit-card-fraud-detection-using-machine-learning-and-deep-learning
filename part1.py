""" In this part :
we are classifying the problem using:
 1)Logistic regression-done
 2)SVM-done
 3)Bayes naive classifier
 4)Random forest-done
 5)Decision tress
 6)K-nearest neighbour
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools

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




# Data analysis and preparation


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
print("-------------------------------------------------------------------------------------")


# 1) Logistic regression
lr = LogisticRegression(random_state=0)
lr.fit(x_train, y_train)
y_pred=lr.predict(X_test)


print("Results for logistic regression : ")
6
5
print('Validation Results')
print('Accuracy: ', lr.score(x_val, y_val))
print('Recall Score: ', recall_score(y_val, lr.predict(x_val)))

print('Test Results')
print('Accuracy: ', lr.score(X_test, Y_test))
print('Recall Score: ', recall_score(Y_test, lr.predict(X_test)))

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

Y_score = lr.decision_function(X_test)
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

print("-------------------------------------------------------------------------------------")

 # 2) Random forest
clf_rf = RandomForestClassifier(random_state=12)
clf_rf.fit(x_train, y_train)
y_pred=clf_rf.predict(X_test)

print("Results for random forests")

print('Validation Results')
print('Accuracy: ', clf_rf.score(x_val, y_val))
print('Recall Score: ', recall_score(y_val, clf_rf.predict(x_val)))

print('Test Results')
print('Accuracy: ', clf_rf.score(X_test, Y_test))
print('Recall Score: ', recall_score(Y_test, clf_rf.predict(X_test)))

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




print("-------------------------------------------------------------------------------------")

# 3)SVM

svm=SVC()
svm.fit(x_train, y_train)
y_pred= svm.predict(X_test)

print("Results for support vector machine  : ")
print('Validation Results')
print('Accuracy: ', svm.score(x_val, y_val))
print('Recall Score: ', recall_score(y_val, svm.predict(x_val)))

print('Test Results')
print('Accuracy: ', lr.score(X_test, Y_test))
print('Recall Score: ', recall_score(Y_test, svm.predict(X_test)))



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


print("-------------------------------------------------------------------------------------")


# 4) Naives bayes classification

nb=GaussianNB()
nb.fit(x_train, y_train)
y_pred=nb.predict(X_test)


print("Results for naive bayes classification  : ")
print('Validation Results')
print('Accuracy: ', nb.score(x_val, y_val))
print('Recall Score: ', recall_score(y_val, nb.predict(x_val)))

print('Test Results')
print('Accuracy: ', nb.score(X_test, Y_test))
print('Recall Score: ', recall_score(Y_test, nb.predict(X_test)))

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


print("-------------------------------------------------------------------------------------")


# 5) k nearest neighbours

kn=KNeighborsClassifier(n_neighbors=3)
kn.fit(x_train, y_train)
y_pred=kn.predict(X_test)

print("Results for k nearest neighbours  : ")
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



# # 5) decision trees
#
# dt = DecisionTreeClassifier(random_state = 0,criterion = 'gini',  splitter='best', min_samples_leaf=1, min_samples_split=2)
# dt.fit(X_train, y_train)
#
# print("Results for Decision trees  : ")
# print('Validation Results')
# print('Accuracy: ', dt.score(x_val, y_val))
# print('Recall Score: ', recall_score(y_val, dt.predict(x_val)))
#
# print('Test Results')
# print('Accuracy: ', dt.score(X_test, Y_test))
# print('Recall Score: ', recall_score(Y_test, dt.predict(X_test)))
