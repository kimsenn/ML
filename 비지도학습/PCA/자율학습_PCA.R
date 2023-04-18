# 자율학습 - 주성분 분석 PCA
# USArrests 데이터터
# 미국의 50개 주의 세가지 강력범죄(Murder, Assault, Rape)와 도시 인구비율(UrbanPop)
str(USArrests)

# 주성분 분석
# princomp:주성분분석 함수, cor : 상관행렬 여부
pc1 <- princomp(USArrests, cor=T)
summary(pc1)
screeplot(pc1, type = "lines", pch=1)

#
pc1$center # 주성분 분석 이전의 변수 평균
pc1$scale # 주성분 분석 이전의 변수 표준편차
pc1$loadings # 변수들의 가중치 확인
pc1$scores # 실제 주성분 값

# 주요 주성분 2개 선택, 시각화
plot(pc1$scores[,1], pc1$scores[,2], xlab="PC1", ylab="PC2")
abline(v=0, h=0, col="gray")

# 행렬도 기법
biplot(pc1, cex=0.7, xlab="PC1", ylab="PC2")
abline(v=0, h=0, col="gray")
