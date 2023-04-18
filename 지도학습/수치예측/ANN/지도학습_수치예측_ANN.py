import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# 붓꽃 데이터셋 로딩
iris = sns.load_dataset("iris")
print(iris.head())
# 원 핫 인코딩
X = iris.iloc[:,0:4].values
y = iris.iloc[:,4].values
encoder = LabelEncoder()
y1 = encoder.fit_transform(y)
Y = pd.get_dummies(y1).values

# train / test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# 모델 생성
model = Sequential()
model.add(Dense(64, input_shape=(4,), activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])
model.summary()

# 모델 학습
hist = model.fit(X_train, y_train, epochs=1500, batch_size=64)
print(hist.history['loss'][-1], hist.history['accuracy'][-1])

# test set 
score_test = model.evaluate(X_test, y_test)
print('Test Score:', score_test)

# train set 
score_train = model.evaluate(X_train, y_train)
print('Train Score:', score_train)

# 그래프 출력
plt.figure(figsize = (12, 8))
plt.plot(hist.history['loss'])
plt.legend(['loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid()
plt.show()
