import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV


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

#clf = RandomForestClassifier(n_estimators = 1000, n_jobs = -1, oob_score = True, class_weight='balanced', random_state=0, max_depth = 8, max_samples = 0.2, max_features = 3, min_samples_leaf = 50, min_samples_split = 200)



clf = SGDClassifier(random_state=22)
parameters = [{ 'loss': ['hinge', 'log', 'perceptron'], 
                'alpha': 10.0**-np.arange(1,7),
                'penalty': ['l1', 'l2', 'elasticnet'],
                }]
clf_refined = GridSearchCV(SGDClassifier(random_state=22), parameters)
clf_refined.fit(X_train, Y_train)

clf_SGD_refined.best_params_

Y_validation_predictions = clf_refined.predict(X_validation)
X_submission['Score'] = clf_refined.predict(X_submission)

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


# next steps:
# check confusion matrix to tune errors