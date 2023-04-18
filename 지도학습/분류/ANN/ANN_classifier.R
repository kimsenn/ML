# 인공신경망 ANN
# install.packages("nnet")# 신경망 라이브러리
library(nnet)
nn.iris <- nnet(Species~., data=iris, size=2, rang=0.1, decay=5e-4, maxit=200)

# 1.size : hidden node 개수
# 2. maxit : 최대반복횟수
# 3. decay : overfitting을 피하기 위해 사용하는 weight decay parameter
# 4. rang : 초기 랜덤 가중치. weights on [-rang, rang]. 기본값 = 0.5
summary(nn.iris)

# 신경망 모형 정오분류표(confusion_matrix)
table(iris$Species, predict(nn.iris, iris, type="class"))

## nnet 신경망 시각화
# install.packages("devtools")
# install.packages("scales")
library(devtools)
library(scales)
source('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')
plot.nnet(nn.iris)
