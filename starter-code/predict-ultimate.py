import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix

# logistic
# xgboost with random forest, or just random forest alone
# how can you be nested?

# Import Saved Pickles
print("Importing Data...")
X_train = pd.read_pickle("./data/X_train.pkl")
X_validation = pd.read_pickle("./data/X_validation.pkl")
Y_train = pd.read_pickle("./data/Y_train.pkl")
Y_validation = pd.read_pickle("./data/Y_validation.pkl")
X_submission = pd.read_pickle("./data/X_submission.pkl")


# Removing String Columns
X_train = X_train.drop(columns=['Summary', 'Text'])
X_validation = X_validation.drop(columns=['Summary', 'Text'])
X_submission = X_submission.drop(columns = ['Summary', 'Text'])


# Learn the model
model = KNeighborsClassifier(n_neighbors=10).fit(X_train, Y_train)

# Predict the score using the model
Y_validation_predictions = model.predict(X_validation)
# does the ID column affect my predictions? search online
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








# next steps:
# go back to process, add undersampling and compare results, should you even undersample?
# tune parameters in the process file to see the output
# regularization techniques
# try other models (boosting methods, SVM (use PCA if you do so), logistic)
# aggregate several models, how? linear regression of the output weightings? can each method give probabilistic weightings? search online on how to
# combine boosting and bagging methods. Don't do it in the blind
# kfold cross validation (You can to properly construct CV predictions for each train fold and then build a 2nd level model using the 1st level models predictions as input features. )
# word count
# oversampling with synonyms then undersampling (generate how much?) What's a good amount to undersample to?
# You can use a grid method for parameter tuning (try multiple parameters)
# word embedding
# Other text parameters
#   word count/length of review (do all of these this prior to tokenizing and removals)
#   punctuations count (more = more extreme?)
#   punctions such as !!! and ??? indicating emotions (Excitement vs confusion?)
#   textmojis such as :)