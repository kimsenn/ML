import matplotlib.pyplot as plt # 시각화
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn import tree

def Random_Forest_Classifier(dataset, max_depth, test_size):
    """ train - test(validation) set 분리 """
    X_train,X_test,y_train,y_test = train_test_split(
        dataset.data,dataset.target,stratify=dataset.target, 
        test_size=test_size, random_state=60
    )    
    """ 랜덤포레스트 모델 생성 및 학습"""
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=max_depth, random_state = 0)
    rf_clf.fit(X_train, y_train)
    
    """ train, test 정확도 확인 """
    train_accuracy = rf_clf.score(X_train,y_train)
    test_accuracy = rf_clf.score(X_test,y_test)
    print("train_accuracy:{:.3f}".format(train_accuracy)) 
    print("test_accuracy:{:.3f}".format(test_accuracy))
    
    """ 랜덤포레스트 중요도 feature_importances """
    features = cancer.feature_names
    importances = rf_clf.feature_importances_
    importance_dict = {name:value for name, value in zip(features, importances)}
    print("Feature Importances: \n", importance_dict)
    
    plt.figure(figsize=(15,5)) 
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), importances, align="center")
    plt.yticks(np.arange(n_features), features)
    plt.xlabel("feature importance")
    plt.ylabel("feature")
    plt.ylim(-1, n_features)
    plt.savefig('RF_python.png', #파일이름 
            facecolor='#eeeeee', #여백색
            edgecolor='black', #테두리색
            format='png', #파일형식
            dpi=100) #해상도(default:100)
    plt.show()

""" 유방암 데이터구조 살펴보기 """
cancer = load_breast_cancer()

""" 랜덤포레스트 함수 실행 """
Random_Forest_Classifier(cancer, 4, 0.25)
