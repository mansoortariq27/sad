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
names= 
 '''
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
end here
