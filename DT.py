from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
import pandas as pd
#create a dataframe for iris dataset
iris_df=pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['Target']=iris.target
corr=iris_df.corr()
print(corr)
import seaborn as sns
sns.heatmap(corr, annot=True, cmap="crest")