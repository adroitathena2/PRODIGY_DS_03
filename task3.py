'''Build a decision tree classifier to predict whether a 
customer will purchase a product or service based on their 
demographic and behavioural data. Use a dataset such as the Bank Marketing 
dataset from the UCI Machine Learning Repository. '''
 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import os 

df = pd.read_csv("C://Users//archi//Documents//INTERNSHIPS//PRODIGY INFOTECH//Task 3//bank-additional.csv", sep=';')
print(df.head())

df.describe()

df.head()

# Define features and target variable
features = df.columns.drop("y")
target = "y"

# Split the data into features and target variable
X = df[features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define categorical columns
categorical_columns = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "poutcome"]

# One-hot encode categorical columns
df_encoded = pd.get_dummies(df, columns=categorical_columns)

# Define features (all columns except the target)
features = df_encoded.drop(columns=["y"]).columns.tolist()

# Define target variable
target = "y"

# Split the data into features (X) and target variable (y), and then into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_encoded[features], df_encoded[target], test_size=0.2, random_state=42)

# Create a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Generate the confusion matrix
confusion = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{confusion}")

# Generate the classification report
classification_rep = classification_report(y_test, y_pred)
print(f"Classification Report:\n{classification_rep}")

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=features, class_names=["no", "yes"])  # Replace with appropriate class names
plt.show()

param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10, 20]
}

grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42),
                           param_grid=param_grid,
                           scoring='accuracy',
                           cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Hyperparameters:")
print(grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"Best Model Accuracy: {accuracy_best}")

plt.figure(figsize=(12, 8))
plot_tree(best_model, filled=True, feature_names=features, class_names=["no", "yes"])
plt.show()







