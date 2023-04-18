from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
import math

def Random_Forest_Regression(dataset, test_size):
    """ 데이터 준비 """
    n_samples = len(dataset.data)
    data = dataset.data.reshape((n_samples, -1))
    
    """ train - test(validation) set 분리 """
    X_train,X_test,y_train,y_test = train_test_split(
        data,dataset.target, test_size=test_size
    )
    
    """ 랜덤포레스트 모델 생성 및 학습"""
    rf_reg = RandomForestRegressor(n_estimators=100, random_state = 0)
    rf_reg.fit(X_train, y_train)
    
    """ 성능 측정 """
    y_train_pred = rf_reg.predict(X_train)
    y_test_pred = rf_reg.predict(X_test)
    print("train_rmse: ",  math.sqrt(mean_squared_error(y_train, y_train_pred)))
    print("test_rmse : ", math.sqrt(mean_squared_error(y_test, y_test_pred)))
     
    """ 랜덤포레스트 중요도 feature_importances """
    features = dataset.feature_names
    importances = rf_reg.feature_importances_
    importance_dict = {name:round(value, 3) for name, value in zip(features, importances)}
    print("Feature Importances: \n", importance_dict)

""" boston 데이터  """
boston = load_boston()
""" 랜덤포레스트 함수 실행 """
Random_Forest_Regression(boston, 0.3)
