import pandas as pd
import snowflake.connector as sf
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import streamlit as st
import matplotlib.pyplot as plt

st.title("SVM Classifier on Iris Dataset")
# Connect to Snowflake database
conn = sf.connect(
    account=st.secrets.connections.snowpark.account,
    user=st.secrets.connections.snowpark.user,
    password=st.secrets.connections.snowpark.password,
    database=st.secrets.connections.snowpark.database,
    schema=st.secrets.connections.snowpark.schema,
    warehouse=st.secrets.connections.snowpark.warehouse
)

# Retrieve iris dataset from Snowflake database
query = 'SELECT * FROM DT'
iris_df = pd.read_sql(query, conn)

# Plot iris dataset
fig, ax = plt.subplots()
ax.scatter(iris_df['SEPAL_LENGTH'], iris_df['SEPAL_WIDTH'], c=iris_df['SPECIES'])
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
st.pyplot(fig)

# Implement SVM on iris dataset
X = iris_df.iloc[:, :-1] # Features
y = iris_df.iloc[:, -1] # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Split data into training and testing sets
clf = SVC(kernel='linear') # Create SVM classifier
clf.fit(X_train, y_train) # Fit classifier on training data
y_pred = clf.predict(X_test) # Make predictions on testing data
accuracy = accuracy_score(y_test, y_pred) # Calculate accuracy score

# Display results using Streamlit
st.write('Accuracy: ', accuracy)
st.write('Number of training examples: ', X_train.shape[0])
st.write('Number of testing examples: ', X_test.shape[0])
