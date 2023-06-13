start here
tt="hello i am here"
label='''
  #get the numeric columns
  df_numeric = df.select_dtypes(include=['float64', 'int64'])

  #get the categorical columns
  df_categorical = df.select_dtypes(include=['object'])

  #label encoding
  from sklearn.preprocessing import LabelEncoder
  le = LabelEncoder()
  df_categorical = df_categorical.apply(le.fit_transform)

  #concatenate the two dataframes : df_dummies, df_numeric
  df_new = pd.concat([df_dummies, df_numeric], axis=1)
 '''
names= '''
 outliers,
 pca,
 one_hot,
 label,
 alpha-betaG,
 perceptron_learning,
 mlp,
 regressionS,
 kmeanM,
 kmeanS,
 bfs_dfs_visualization,
 maze,
 bfs,
 dfs,
 pre-processing
 ''' 
outliers='''
  #check for outliers using iqr
  Q1 = df.quantile(0.25)
  Q3 = df.quantile(0.75)
  IQR = Q3 - Q1
  print(IQR)
  #remove outliers
  df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]

  #check for outliers using z-score
  from scipy import stats
  import numpy as np
  z = np.abs(stats.zscore(df))
  #remove outliers
  df = df[(z < 3).all(axis=1)]
  
  #check for outliers using boxplot
  import seaborn as sns
  sns.boxplot(x=df['Age'])
'''
pca='''
  #get the numeric columns
  df_numeric = df.select_dtypes(include=['float64', 'int64'])
  from sklearn.decomposition import PCA
  pca = PCA(n_components=2)
  pca.fit(df_numeric)
  pca_data = pca.transform(df_numeric)
  pca_data
  '''

end here
