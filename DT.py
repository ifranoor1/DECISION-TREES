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
#setting a threshold of 50% those fearure whose correlation with target variable(price) is less than or equal to 0.5 will drop
cor_target = abs(corr["Target"])

relevant_features = cor_target[cor_target>0.7]
#it will print those feature variable that has high correlation with price
print(relevant_features)
rel_feature= iris_df
rel_feature_heatmap=rel_feature[['sepal length (cm)',
'petal length (cm)',
'petal width (cm)' ,
'Target'   ]]
corr_rel_feature= rel_feature_heatmap.corr()
print(corr_rel_feature)
#x is the feature variable
X=iris_df['petal length (cm)']
#y is the dependent or target variable
y=iris_df['Target']
X=np.array(X).reshape(-1,1)
y=np.array(y).reshape(-1,1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.3, random_state=42)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
