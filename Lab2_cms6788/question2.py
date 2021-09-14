import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plot

#Problem 2
df2 = pd.read_csv("../Lab2_Data/DF2", index_col=0)
sns.scatterplot(data=df2, x='0', y='1')

cov_matrix = np.cov(df2, rowvar=False)
eig_vals , q = np.linalg.eig(cov_matrix)
print("Q",q)
q_inv = q.T
print("Qinv", q_inv)


print("cov matrix:", cov_matrix)
df_feature1 = df2.iloc[:, 0].tolist()
df_feature2 = df2.iloc[:, 1].tolist()
z = np.matrix([df_feature1, df_feature2])
z_outlier1 = np.matrix([[-1],[1]])
z_outlier2 = np.matrix([[5.5],[5]])
print("eig", eig_vals)
print(z)
print(z.shape)
y_outlier1  = q_inv @ z_outlier1
y_outlier2 = q_inv @ z_outlier2
y_outlier1 = np.linalg.inv(np.sqrt(np.diag(eig_vals))) @ y_outlier1
y_outlier2 = np.linalg.inv(np.sqrt(np.diag(eig_vals))) @ y_outlier2
y = q_inv @ z
y = np.linalg.inv(np.sqrt(np.diag(eig_vals))) @ y
print(y.shape)
print("(-1,1) transformed: ", y_outlier1)
print("(5.5,5) transformed: ", y_outlier2)
plot.figure()
plot.title("NEW DATA")
plot.xlim(-20,20)
plot.ylim(-20,20)
plot.scatter(x=y[0,:].tolist()[0], y = y[1,:].tolist()[0])
plot.show()