import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix


# Loading
X_train = pd.read_csv("./data/X_train.csv")
X_submission = pd.read_csv("./data/X_submission.csv")


# # Loading (small test set)
# X_train = pd.read_csv("./data/small_train.csv")
# X_submission = pd.read_csv("./data/small_submission.csv")

# Converting objects to strings
X_train['ProductId']= X_train['ProductId'].apply(str)
X_train['UserId']= X_train['UserId'].apply(str)
X_train['Summary']= X_train['Summary'].apply(str)
X_train['Text']= X_train['Text'].apply(str)


# Train/Test split
X_train, X_test, Y_train, Y_test = train_test_split(
        X_train.drop(['Score'], axis=1),
        X_train['Score'],
        test_size=1/4.0,
        random_state=0
    )


# Drop NA
X_train.dropna()
# to get the number of Null: X_train.isna().sum()

# Lowercase
X_train['Summary'] = X_train['Summary'].str.lower()
X_train['Text'] = X_train['Text'].str.lower()

# Tokenization (with Punctuation, Special Character, and Whitespace Removal via Regexp)

# Regexp tokenizer
print(X_train.dtypes)
tokenizer = RegexpTokenizer(r'\w+')
X_train['tokenized_summary'] = X_train['Summary'].apply(word_tokenize)
X_train['tokenized_text'] = X_train['Text'].apply(tokenizer.tokenize)

print(X_train.head()[['tokenized_summary', 'tokenized_text']])
# Stopwords

# remove all words below length 2

# Lemmatization

# checkout what tfidf vectorizer offers

# do you need to reconstruct it to a string or can you pass in a vector? tfidf on tokenized input



# could punctuations be actually useful?


# non-word parameters:
# sentence length


# helpfulness error and convertion
# datetime processing



# do feature extraction on submission file after you're finished with the training set, know how you predicted training data

'''
# Learn the model
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