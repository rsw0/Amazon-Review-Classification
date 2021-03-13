import pandas as pd

trainingSet = pd.read_csv("./data/train.csv")
testingSet = pd.read_csv("./data/test.csv")

X_test = pd.merge(trainingSet, testingSet, left_on='Id', right_on='Id')
X_test = X_test.drop(columns=['Score_x'])
X_test = X_test.rename(columns={'Score_y': 'Score'})
small_submission = X_test.sample(frac=0.01, random_state=1)
small_submission.to_csv("./data/small_submission.csv", index=False)

X_train = trainingSet[trainingSet['Score'].notnull()]
small_train = X_train.sample(frac=0.01, random_state=1)
small_train.to_csv("./data/small_train.csv", index=False)