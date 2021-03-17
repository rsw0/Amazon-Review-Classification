import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix
import xgboost as xgb


# change the name of the import file names below if you want to use it. In the download folder, it has name extension 1000

# Import Saved Pickles
print("Importing Data...")
X_train = pd.read_pickle("./data/X_train.pkl")
X_validation = pd.read_pickle("./data/X_validation.pkl")
Y_train = pd.read_pickle("./data/Y_train.pkl")
Y_validation = pd.read_pickle("./data/Y_validation.pkl")
X_submission = pd.read_pickle("./data/X_submission.pkl")


# Removing String Columns
print("Dropping Unused Columns")
X_train = X_train.drop(columns=['Summary', 'Text'])
X_validation = X_validation.drop(columns=['Summary', 'Text'])
X_submission = X_submission.drop(columns = ['Summary', 'Text'])


# Converting to DMatrix for XGBoost
print("Converting to DMatrix...")
dtrain = xgb.DMatrix(X_train, label=Y_train)
dvalidation = xgb.DMatrix(X_validation, label=Y_validation)
dsubmission = xgb.DMatrix(X_submission)


# Setting Parameters
print("Setting XGBoost Parameters...")
param = {'max_depth':2, 'eta':1, 'objective':'binary:logistic'}
num_round = 999

eval_set = [(X_train, Y_train), (X_validation, Y_validation)]

# early stopping
# regularization
# Cross Validation




# Train
print("Training XGBoost...")
bst = xgb.train(param, dtrain, num_round, evals = eval_set, early_stopping_round = 10)


# Predict
print("Predicting...")
# can you do this here? if you use mean_squard_error, just don't even give dvalidation it's labels?
Y_validation_predictions = bst.predict(dvalidation)
X_submission['Score'] = bst.predict(dsubmission)
print("RMSE on validation set = ", mean_squared_error(Y_validation, Y_validation_predictions))


# Plot a confusion matrix
print("Creating Confusion Matrix...")
cm = confusion_matrix(Y_validation, Y_validation_predictions, normalize='true')
sns.heatmap(cm, annot=True)
plt.title('Confusion matrix of the classifier')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('matrix.png', dpi=300)


# Create the submission file
print('Creating Submission File')
submission = X_submission[['Id', 'Score']]
submission.to_csv("./data/submission.csv", index=False)




'''
# Learn the model
model = KNeighborsClassifier(n_neighbors=10).fit(X_train, Y_train)

# Predict the score using the model
Y_validation_predictions = model.predict(X_validation)
X_submission['Score'] = model.predict(X_submission)

# Evaluate your model on the validation set
print("RMSE on validation set = ", mean_squared_error(Y_validation, Y_validation_predictions))

# Plot a confusion matrix
cm = confusion_matrix(Y_validation, Y_validation_predictions, normalize='true')
sns.heatmap(cm, annot=True)
plt.title('Confusion matrix of the classifier')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('matrix.png', dpi=300)

# Create the submission file
submission = X_submission[['Id', 'Score']]
submission.to_csv("./data/submission.csv", index=False)
'''

# next steps:
# check confusion matrix to tune errors