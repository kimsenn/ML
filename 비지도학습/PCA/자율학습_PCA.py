# 유방암 샘플 양성, 음성 - PCA
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cancer=load_breast_cancer()
X = cancer.data
y = cancer.target

# 표준화
X_ = StandardScaler().fit_transform(X)

# 2D PCA
pca = PCA(n_components=2)
pc = pca.fit_transform(X_)

pc_y = np.c_[pc,y]
df = pd.DataFrame(pc_y,columns=['PC1','PC2','diagnosis'])
print("PCA table", df)

# 시각화
fig, ax = plt.subplots()
scatter = ax.scatter(x=df['PC1'],y=df['PC2'],c=df['diagnosis'])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(*scatter.legend_elements(),) 
