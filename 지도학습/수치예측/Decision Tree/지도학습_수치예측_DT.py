from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
import math


def Decision_Tree_Regression(dataset, test_size):
    """ 데이터 준비 """
    n_samples = len(dataset.data)
    data = dataset.data.reshape((n_samples, -1))
    # print(dataset.target)
    
    """ train - test(validation) set 분리 """
    X_train,X_test,y_train,y_test = train_test_split(
        data,dataset.target, test_size=test_size
    )
    
    """ 트리 모델 생성 및 학습"""
    dt_reg = DecisionTreeRegressor(random_state=1)
    dt_reg.fit(X_train, y_train)
    
    """ 성능 측정 """
    y_train_pred = dt_reg.predict(X_train)
    y_test_pred = dt_reg.predict(X_test)
    print("train_rmse: ",  math.sqrt(mean_squared_error(y_train, y_train_pred)))
    print("test_rmse : ", math.sqrt(mean_squared_error(y_test, y_test_pred)))
     

""" boston 데이터  """
boston = load_boston()

""" 의사결정트리 함수 실행 """
Decision_Tree_Regression(boston, 0.3)
