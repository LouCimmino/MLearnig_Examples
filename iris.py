import pandas as pd
import mglearn
import matplotlib as plt
import seaborn as sns

from sklearn.datasets import load_iris

iris_dataset = load_iris()

print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))

print(iris_dataset['DESCR'][:2675] + "\n")
#print(len(iris_dataset['DESCR']))
#print("Data in row: \n{}".format(iris_dataset['data'][:150]))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

iris_dataframe = pd.DataFrame(x_train, columns=iris_dataset.feature_names)

pd.plotting.scatter_matrix(iris_dataframe, alpha=.9, c=y_train, figsize=(15, 15), hist_kwds={'bins': 100}, marker='o', s=10, cmap=mglearn.cm3)
iris_dataframe['y_train'] = y_train

plt.pyplot.show()
