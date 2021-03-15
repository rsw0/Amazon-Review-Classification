# next steps:
# add summary into the equation as well. two vectors tfidf. How? fit two models and linearly weight their outputs? search combine two tfidf together (e.g., title and text) online
# add non-word features
# tune the parameters max_df min_df max_features, especially, adjust max_df to filter out certain words that appears too often but are not predictive
# over and under sampling in the process file
# dont do it in the blind
# regularization technique on random forest?
# you can use a grid method for parameter tuning (try multiple parameters)
# try other models (boosting methods, SVM (use PCA if you do so), logistic)
# aggregate several models, how? linear regression of the output weightings? can each method give probabilistic weightings? search online on how to
# combine boosting and bagging methods. Don't do it in the blind
# word embedding
# oversampling with synonyms then undersampling (generate how much?) What's a good amount to undersample to?
# kfold cross validation (You can to properly construct CV predictions for each train fold and then build a 2nd level model using the 1st level models predictions as input features. )
# Other text parameters
#   word count/length of review (do all of these this prior to tokenizing and removals)
#   punctuations count (more = more extreme?)
#   punctions such as !!! and ??? indicating emotions (Excitement vs confusion?)
#   textmojis such as :)


'''
# try grid search logistic regression with tfidf normalization first
# Learn the model
# check saved bookmarks on random forest
# naive bayes multinomial NB, don't need to be vectorized?
# xgboost with random forest

model = KNeighborsClassifier(n_neighbors=3).fit(X_train_processed, Y_train)

# Predict the score using the model
Y_test_predictions = model.predict(X_test_processed)
X_submission['Score'] = model.predict(X_submission_processed)

# Evaluate your model on the testing set
print("RMSE on testing set = ", mean_squared_error(Y_test, Y_test_predictions))

# Plot a confusion matrix
cm = confusion_matrix(Y_test, Y_test_predictions, normalize='true')
sns.heatmap(cm, annot=True)
plt.title('Confusion matrix of the classifier')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Create the submission file
submission = X_submission[['Id', 'Score']]
submission.to_csv("./data/submission.csv", index=False)
'''