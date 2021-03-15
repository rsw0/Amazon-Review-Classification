import time
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# # Loading
# X_train = pd.read_csv("./data/X_train.csv")
# X_submission = pd.read_csv("./data/X_submission.csv")


# Loading (small test set)
X_train = pd.read_csv("./data/small_train.csv")
X_submission = pd.read_csv("./data/small_submission.csv")


# A test text to test individual steps
testtext = "He ended up burning his fingers poking someone else's fire."
# testtext = "Nick likes to PLAYful played tried attempted delightful footBall, aren't @#% @ #^ &&%$*!!@#$ however   you're don't he's is not'$ t  os o FOND of ten'nis ab bc cd"


# Converting objects to strings
X_train['ProductId']= X_train['ProductId'].apply(str)
X_train['UserId']= X_train['UserId'].apply(str)
X_train['Summary']= X_train['Summary'].apply(str)
X_train['Text']= X_train['Text'].apply(str)


# Drop NA
X_train.dropna()
# to get the number of Null: X_train.isna().sum()


# Lowercase
X_train['Summary'] = X_train['Summary'].str.lower()
X_train['Text'] = X_train['Text'].str.lower()
testtext = testtext.lower()
print(testtext)


# Punctuation, Special Character & Whitespace (adjusted for stopwords)
def fast_rem(my_string):
    return(re.sub(r'[^a-z \']', '', my_string).replace('\'', ' '))
X_train['Summary'] = X_train['Summary'].apply(fast_rem)
X_train['Text'] = X_train['Text'].apply(fast_rem)
testtext = fast_rem(testtext)
print(testtext)


# Tokenization, Lemmatization & Removing Noise (Tokens below Length 2)
lemmatizer = WordNetLemmatizer()
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None
def lemmatize_sentence(sentence):
    #tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(word_tokenize(sentence))  
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if len(word) <= 2:
            continue
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:        
            #else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)
t1_start = time.perf_counter()
X_train['Summary'] = X_train['Summary'].apply(lemmatize_sentence)
X_train['Text'] = X_train['Text'].apply(lemmatize_sentence)
t1_stop = time.perf_counter()
print("Elapsed time during the whole program in seconds:", t1_stop-t1_start) 
testtext = lemmatize_sentence(testtext)
print(testtext)


# Stopword
cachedStopWords = stopwords.words("english")
def fast_stop(my_string):
    return(' '.join([word for word in my_string.split() if word not in cachedStopWords]))
X_train['Summary'] = X_train['Summary'].apply(fast_stop)
X_train['Text'] = X_train['Text'].apply(fast_stop)
testtext = fast_stop(testtext)
print(testtext)





# does spacy takes in the full sentence when assigning tokens? Does it interpret? will I get things wrong if I do it in the current way?


# non-word parameters:
# word count
# character count
# punctuations count
# average word length
# average sentence length


# do oversampling after completely processing the dataset, apply processing to all datasets







# save the proprocessed data in to a panda file to be imported later in the prediction file. Use two separate files to avoid accumulating runtime

# do feature extraction on submission file after you're finished with the training set, know how you predicted training data





'''
# do you need to reconstruct it to a string or can you pass in a vector? tfidf on tokenized input
# checkout what tfidf vectorizer offers
tfidf_s_time = time.perf_counter()
vectorizer = TfidfVectorizer(strip_accents=ascii, lowercase=True, preprocessor=None, tokenizer=None, analyzer='word', stop_words='english', ngram_range=(1, 1))
#X_train['tokenized_text'] = X_train['Text'].apply(vectorizer.fit_Transform)
features = vectorizer.fit_transform(X_train['Text'])
tfidf_f_time = time.perf_counter()
print('tfidf vectorizer took: ' + str(tfidf_f_time - tfidf_s_time) + ' seconds')
print(features)
'''
# do this after you've done the processing
# Train/Test split
X_train, X_test, Y_train, Y_test = train_test_split(
        X_train.drop(['Score'], axis=1),
        X_train['Score'],
        test_size=1/4.0,
        random_state=0
    )


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

# next steps:
# bigrams (how do you do it with stopwords removed?)
# punctuations indicating emotion, make that into a binary column?
# helpfulness columnn actually useful?
# datetime processing? check review time and align more closely to the review time within that period






# # spaCy Tokenization and Lemmatization with noise removal below length 2. Too slow
# nlp = spacy.load('en_core_web_sm', disable=["parser", "ner"]) 
# def fast_lemma(my_string):
#     spacy_form = nlp(my_string)
#     return(" ".join([word.lemma_ for word in spacy_form if len(word) > 2]))
# testtext = fast_lemma(testtext)
# print(testtext)
# t1_start = time.perf_counter()
# X_train['Summary'] = X_train['Summary'].apply(fast_lemma)
# X_train['Text'] = X_train['Text'].apply(fast_lemma)
# t1_stop = time.perf_counter()
# print("Elapsed time during the whole program in seconds:", t1_stop-t1_start) 