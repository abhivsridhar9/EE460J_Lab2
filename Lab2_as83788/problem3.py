import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv('../Lab2_Data/DF1',index_col=0,header=None)

# part a
# pandas correlation scatter plot
pd.plotting.scatter_matrix(df)
plt.show()

# seaborn correlation scatter plot
sns.pairplot(df)
plt.show()

# part b
print(df.cov())

# no correlation for covariance matrix coefficients with absolute values close to 0
# strong correlation for covariance matrix coefficients with absolute values close to 1
# correlation coefficients with values above 1 indicate that columns were correlated against themselves

# part c
mean_matrix=[0,0,0]
cov_matrix=[[10,0,0],[0,20,0.8],[0,0.8,30]]

gaus=np.random.multivariate_normal(mean_matrix,cov_matrix,10000)
sample_cov=np.cov(gaus, rowvar=False)

print(sample_cov)

cov_x_list=[]
cov_y_list=[]
for i in range(10,10000,100):
    gaus = np.random.multivariate_normal(mean_matrix, cov_matrix, i)
    cov_x_list.append(i)
    cov_y_list.append(np.cov(gaus, rowvar=False).tolist()[1][2]-0.8)

plt.scatter(x=cov_x_list,y=cov_y_list)
plt.show()
