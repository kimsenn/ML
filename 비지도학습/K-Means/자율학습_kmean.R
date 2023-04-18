# 자율학습 - K평균 클러스터링
# iris 데이터 -> 결과가 실제 분류별로 군집화 되었는지 테스트

# 전처리 : iris 목표변수 Y 제거
iris2 <- iris[,1:4]

#
km.out.withness <- c() # 군집내 제곱합
km.out.between <- c() # 군집간 제곱합
km.num <- c() # 중심점 개수

# 중심점의 개수 2개부터 7개까지
for (i in 2:7){
  set.seed(1)
  km.out <- kmeans(iris2, centers = i)
  km.out.withness[i-1] <- km.out$tot.withinss
  km.out.between[i-1] <- km.out$betweenss
  km.num[i-1] <- i
  # print(km.out)
}

# plot 확인 -> k=3이 가장 적절
data.frame(km.num, km.out.withness, km.out.between)
plot(km.num, km.out.between, type = "o")
plot(km.num, km.out.withness, type = "o")

# k=3으로 분석 수행
km.out.k3 <- kmeans(iris2, centers = 3)
km.out.k3$centers # 각 군집의 중심점 출력
km.out.k3$cluster # 각 관측치의 군집번호 출력
km.out.k3$size # 각 군집의 관측치 개수 출력
table(km.out.k3$cluster, iris$Species) # 클러스터링 결과와 비교

# 시각화
plot(iris2[,1:2], col=km.out.k3$cluster,
     pch=ifelse(km.out.k3$cluster==1,16,
          ifelse(km.out.k3$cluster==2, 17,18)),cex=2)
points(km.out.k3$centers, col=1:3, pch=16:18, cex=5)
