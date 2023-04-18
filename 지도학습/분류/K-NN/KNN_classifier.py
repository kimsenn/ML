# python libraries 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer

# KNN_Classifier 함수 
def KNN_Classifier(data, target, k_nb, test_size):
    """ train - test(validation) set 분리 """
    X_train,X_test,y_train,y_test = train_test_split(
        data,target,stratify=target, test_size=test_size, random_state=66
    )
    # X_train: 426  X_test: 143
    # y_train: 426  y_test: 143
    
    """ k후보군 1 ~ k_nb 개로 모델 생성 및 학습"""
    neighbors=range(1,k_nb+1)
    training_accuracy, test_accuracy = [], []
    for n in neighbors:
        # k=n으로 모델 생성
        clf = KNeighborsClassifier(n_neighbors=n)
        # 모델에 학습시키기
        clf.fit(X_train,y_train)
        # 모델의 정확도 확인
        training_accuracy.append(clf.score(X_train,y_train))
        test_accuracy.append(clf.score(X_test,y_test))
        print("K=",n)
        print("train accuracy:",clf.score(X_train, y_train))
        print("test accuracy:", clf.score(X_test, y_test))
        
    
    """ 그래프로 k 값에 따른 정확도 확인 """
    plt.plot(neighbors, training_accuracy, label="train accuracy")
    plt.plot(neighbors, test_accuracy, label="test accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("K (n_neighbors)")
    plt.title("KNN - Breast Cancer Classifier Accuracy")
    plt.legend()
    plt.savefig('KNN_python.png', #파일이름 
                facecolor='#eeeeee', #여백색
                edgecolor='black', #테두리색
                format='png', #파일형식
                dpi=100) #해상도(default:100)
    plt.show()

""" 유방암 데이터 load """
cancer = load_breast_cancer()
# print(cancer.feature_names) # X
# print(cancer.data) # X
# print(cancer.target) # label(Y)
# print(cancer.DESCR)

""" KNN_Classifier 함수 실행 """
KNN_Classifier(cancer.data, cancer.target, 10, 0.25)