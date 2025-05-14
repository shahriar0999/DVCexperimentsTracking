import seaborn as sns
import pandas as pd

iris = sns.load_dataset('iris')
iris["species"] = iris["species"].map({"setosa":1, "versicolor":2, "virginica":3})
iris.to_csv('iris.csv', index=False)
print("Iris Data saved")
