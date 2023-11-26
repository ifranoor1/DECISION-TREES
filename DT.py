from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
import pandas as pd
#create a dataframe for iris dataset
iris_df=pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['Target']=iris.target