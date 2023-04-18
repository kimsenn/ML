from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

cancer = load_breast_cancer()
X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,stratify=cancer.target)

nb = GaussianNB()
nb.fit(X_train,y_train)

train_accuracy = nb.score(X_train,y_train)
test_accuracy = nb.score(X_test,y_test)
print("train_accuracy:{:.3f}".format(train_accuracy)) 
print("test_accuracy:{:.3f}".format(test_accuracy))

y_pred = nb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d proints : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))

