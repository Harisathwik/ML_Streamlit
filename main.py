import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets, svm
import numpy as np

st.title("Support Vector Machine Demo")
# Load the Iris dataset
iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target

# Sidebar for selecting the range of data
st.sidebar.header('Select Data Range')
range_slider = st.sidebar.slider('Select a range of data:', 0, len(iris_df), (0, len(iris_df)))

# Filter the dataset based on the selected range
iris_df_filtered = iris_df.iloc[range_slider[0]:range_slider[1]]

# Create a Support Vector Machine model
model = svm.SVC()

# Train the model on the filtered dataset
X = iris_df_filtered.iloc[:, :-1].values
y = iris_df_filtered.iloc[:, -1].values
model.fit(X, y)

# Create a meshgrid to visualize the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Make predictions on the meshgrid
Z = model.predict(np.c_[xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel()), np.zeros_like(xx.ravel())])
Z = Z.reshape(xx.shape)

# Plot the decision boundary and the data points
fig, ax = plt.subplots()
sns.scatterplot(data=iris_df_filtered, x='sepal length (cm)', y='sepal width (cm)', hue='species', ax=ax)
ax.contourf(xx, yy, Z, alpha=0.3)
ax.set_title('SVM Decision Boundary')
st.pyplot(fig)
