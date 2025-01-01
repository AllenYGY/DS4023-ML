import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import pandas as pd


# Create a data set for analysis
file_path = '2.2-Exercise.csv'
df = pd.read_csv(file_path)
x = df['High'].values.reshape(-1, 1)
y = df['Target'].values


# Split the data set into testing and training data
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

# Create a linear regression object
regression = linear_model.LinearRegression()

# Train the model using the training set
regression.fit(x_train, y_train)

# Make predictions using the testing set
y_predictions = regression.predict(x_test)

# Plot the data
sns.set_style("darkgrid")
sns.regplot(x=x_test.flatten(), y=y_test, fit_reg=False)
plt.plot(x_test, y_predictions, color='black')

# Remove ticks from the plot
plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.show()
