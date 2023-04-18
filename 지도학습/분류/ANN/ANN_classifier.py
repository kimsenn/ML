import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier ## neural net
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, \
    accuracy_score, f1_score, precision_score, recall_score, \
    roc_auc_score, roc_curve, auc, \
    ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler  # 변수 표준화

# 데이터 준비 및 분할, 입력변수 표준화
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

# 변수 표준화
scaler = StandardScaler() # 변수 표준화 클래스
scaler.fit(X_train)  # 표준화를 위해 변수별 파라미터(평균, 표준편차) 계산
# scaler.mean_, scaler.scale_
X_train_std = scaler.transform(X_train)  # 훈련자료 표준화 변환
X_test_std = scaler.transform(X_test)    # 검증자료 표준화 변환

# NN 모형세팅 (비표준화)
nn = MLPClassifier(hidden_layer_sizes=(10, 10, 5), random_state=1, max_iter=1000)
nn.fit(X_train, y_train) # 비표준화 훈련자료를 이용한 학습

train_accuracy = nn.score(X_train,y_train)
test_accuracy = nn.score(X_test,y_test)
print("train_accuracy:{:.3f}".format(train_accuracy)) 
print("test_accuracy:{:.3f}".format(test_accuracy))

# NN 모형세팅 (표준화)
nn_s = MLPClassifier(hidden_layer_sizes=(10, 10, 5), random_state=1, max_iter=1000)
nn_s.fit(X_train_std, y_train) # 표준화 훈련자료를 이용한 학습
# 성능 
train_accuracy = nn_s.score(X_train_std,y_train)
test_accuracy = nn_s.score(X_test_std,y_test)
print("표준화 train_accuracy:{:.3f}".format(train_accuracy)) 
print("표준화 test_accuracy:{:.3f}".format(test_accuracy))

plt.figure(dpi=100)
plt.plot(nn.loss_curve_, label='non-standardization', linestyle='--')
plt.plot(nn_s.loss_curve_, label='standardization', linestyle="-.")
plt.xlabel("Number of iterations")
plt.ylabel("Training loss"),
plt.legend()
plt.show()
