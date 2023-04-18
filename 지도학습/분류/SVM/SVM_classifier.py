import sklearn.svm as svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,stratify=cancer.target)


svm_clf =svm.SVC(kernel = 'linear')
svm_clf.fit(X_train, y_train)

train_accuracy = svm_clf.score(X_train,y_train)
test_accuracy = svm_clf.score(X_test,y_test)
print("train_accuracy:{:.3f}".format(train_accuracy)) 
print("test_accuracy:{:.3f}".format(test_accuracy))

# 교차검증
scores = cross_val_score(svm_clf, cancer.data, cancer.target, cv = 5)
print(scores)

