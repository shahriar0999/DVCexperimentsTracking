import seaborn as sns
import pandas as pd

iris = sns.load_dataset('iris')
iris = iris[iris['species'].isin(['setosa','virginica'])]
iris["species"] = iris["species"].map({"setosa":0, "virginica":1})
print(iris.head())
iris.to_csv('iris.csv', index=False)
print("Iris Data saved")