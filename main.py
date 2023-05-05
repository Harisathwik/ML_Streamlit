import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset
iris_df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                      header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])

# Sidebar for selecting the range of data
st.sidebar.header('Select Data Range')
range_slider = st.sidebar.slider('Select a range of data:', 0, len(iris_df), (0, len(iris_df)))

# Filter the dataset based on the selected range
iris_df_filtered = iris_df.iloc[range_slider[0]:range_slider[1]]

# Plot a scatter plot of sepal length and sepal width with color-coded species
fig, ax = plt.subplots()
sns.scatterplot(data=iris_df_filtered, x='sepal_length', y='sepal_width', hue='species', ax=ax)
ax.set_title('Scatter Plot of Sepal Length vs. Sepal Width')
st.pyplot(fig)
