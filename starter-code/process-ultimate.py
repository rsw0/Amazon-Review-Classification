import time
import pandas as pd
import re
import numpy as np
import nltk
from imblearn.under_sampling import RandomUnderSampler
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# # Loading
# print("Loading data...")
# X_train = pd.read_csv("./data/X_train.csv")
# X_submission = pd.read_csv("./data/X_submission.csv")


# Loading (small test set)
print("Loading data...")
X_train = pd.read_csv("./data/small_train.csv")
X_submission = pd.read_csv("./data/small_submission.csv")





# Train/Validation Split
print("Train/Validation Split...")
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train.drop(['Score'], axis=1), X_train['Score'], test_size=0.20, random_state=0, stratify=X_train['Score'])
print(list(Y_validation.columns.values))
print(list(Y_train.columns.values))




# Subsetting Columns
print("Dropping unnecessary columns...")
X_train = X_train.drop(columns=['ProductId', 'UserId', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Time'])
X_submission = X_submission.drop(columns=['ProductId', 'UserId', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Time', 'Score'])


# Handling NA
print("Handling NA...")
X_train.dropna(inplace=True)


# Undersampling
undersample = RandomUnderSampler(sampling_strategy=0.5)





# Converting objects to strings
print("Converting to strings...")
X_train['Summary'] = X_train['Summary'].apply(str)
X_train['Text'] = X_train['Text'].apply(str)
X_submission['Summary'] = X_submission['Summary'].apply(str)
X_submission['Text'] = X_submission['Text'].apply(str)


# Lowercase
print("Converting to lowercase...")
X_train['Summary'] = X_train['Summary'].str.lower()
X_train['Text'] = X_train['Text'].str.lower()
X_submission['Summary'] = X_submission['Summary'].str.lower()
X_submission['Text'] = X_submission['Text'].str.lower()


# Punctuation, Special Character & Whitespace (adjusted for stopwords)
print("Removing punctuations and special characters...")
def fast_rem(my_string):
    return(re.sub(r'[^a-z \']', '', my_string).replace('\'', ' '))
X_train['Summary'] = X_train['Summary'].apply(fast_rem)
X_train['Text'] = X_train['Text'].apply(fast_rem)
X_submission['Summary'] = X_submission['Summary'].apply(fast_rem)
X_submission['Text'] = X_submission['Text'].apply(fast_rem)


# Tokenization, Lemmatization
print("Tokenization and Lemmatization...")
lemmatizer = WordNetLemmatizer()
tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
def fast_lemma(sentence):
    return (" ".join([lemmatizer.lemmatize(key[0], tag_dict.get(key[1][0], wordnet.NOUN)) for key in nltk.pos_tag(word_tokenize(sentence))]))
tk_start = time.perf_counter()
X_train['Summary'] = X_train['Summary'].apply(fast_lemma)
X_train['Text'] = X_train['Text'].apply(fast_lemma)
X_submission['Summary'] = X_submission['Summary'].apply(fast_lemma)
X_submission['Text'] = X_submission['Text'].apply(fast_lemma)
tk_stop = time.perf_counter()
print("Tokenization and Lemmatization took: " + str(tk_stop-tk_start) + ' seconds')


# Stopword & Noise Removal (Token with length below 2)
print("Removing Stopwords...")
cachedStopWords = stopwords.words("english")
def fast_stop(my_string):
    return(' '.join([word for word in my_string.split() if word not in cachedStopWords and len(word) > 2]))
X_train['Summary'] = X_train['Summary'].apply(fast_stop)
X_train['Text'] = X_train['Text'].apply(fast_stop)
X_submission['Summary'] = X_submission['Summary'].apply(fast_stop)
X_submission['Text'] = X_submission['Text'].apply(fast_stop)


# Vectorizer
print("Vectorization - Fitting...")
vectorizer = TfidfVectorizer(lowercase = False, ngram_range= (1,2), min_df = 5, max_df = 0.9, max_features = 5000).fit(X_train['Text'])
vectorizer_s = TfidfVectorizer(lowercase = False, ngram_range= (1,2), min_df = 5, max_df = 0.9, max_features = 1000).fit(X_train['Summary'])
print("Vectorization - Transforming...")
X_train_vect = vectorizer.transform(X_train['Text'])
X_submission_vect = vectorizer.transform(X_submission['Text'])
X_train_vect_s = vectorizer_s.transform(X_train['Summary'])
X_submission_vect_s = vectorizer_s.transform(X_submission['Summary'])
print("Vectorization - Merging Sparse Matrices")
X_train_vect = hstack((X_train_vect, X_train_vect_s))
X_submission_vect = hstack((X_submission_vect, X_submission_vect_s))
print("Vectorization - SVD...")
svd_s_time = time.perf_counter()
svd = TruncatedSVD(n_components=600, random_state=0)
X_train_vect = svd.fit_transform(X_train_vect)
print(svd.explained_variance_ratio_.sum())
X_submission_vect = svd.fit_transform(X_submission_vect)
print(svd.explained_variance_ratio_.sum())
svd_f_time = time.perf_counter()
print('SVD took: ' + str(svd_f_time - svd_s_time) + ' seconds')
print("Vectorization - Creating Pandas df...")
X_train_df = pd.DataFrame(X_train_vect, columns=np.arange(500)).set_index(X_train.index.values)
X_submission_df = pd.DataFrame(X_submission_vect, columns=np.arange(500)).set_index(X_submission.index.values)
# X_train_df = pd.DataFrame(X_train_vect.toarray(), columns=vectorizer.get_feature_names()).set_index(X_train.index.values)
# X_submission_df = pd.DataFrame(X_submission_vect.toarray(), columns=vectorizer.get_feature_names()).set_index(X_submission.index.values)
print("Vectorization - Joining with Original df...")
# X_train and X_submission below contains original columns with tfidf 
X_train = X_train.join(X_train_df)
X_submission = X_submission.join(X_submission_df)




# Saving to Local
print("Saving to Local...")
X_train.to_pickle("./data/X_train.pkl")
X_validation.to_pickle("./data/X_validation.pkl")
Y_train.to_pickle("./data/Y_train.pkl")
Y_validation.to_pickle("./data/Y_validation.pkl")
X_submission.to_pickle("./data/X_submission.pkl")




# Old Functions
'''
# time
tfidf_s_time = time.perf_counter()
tfidf_f_time = time.perf_counter()
print('tfidf vectorizer took: ' + str(tfidf_f_time - tfidf_s_time) + ' seconds')
'''

'''
# old WordNet lemmatizer, used list comprehension instead, performance similar
def lemmatize_sentence(sentence):
    nltk_tagged = nltk.pos_tag(word_tokenize(sentence))  
    wordnet_tagged = map(lambda x: (x[0], tag_dict.get(x[1][0])), nltk_tagged)
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
'''

'''
# spaCy Tokenization and Lemmatization with noise removal below length 2. Too slow
nlp = spacy.load('en_core_web_sm', disable=["parser", "ner"]) 
def fast_lemma(my_string):
    spacy_form = nlp(my_string)
    return(" ".join([word.lemma_ for word in spacy_form if len(word) > 2]))
testtext = fast_lemma(testtext)
print(testtext)
t1_start = time.perf_counter()
X_train['Summary'] = X_train['Summary'].apply(fast_lemma)
X_train['Text'] = X_train['Text'].apply(fast_lemma)
t1_stop = time.perf_counter()
print("Elapsed time during the whole program in seconds:", t1_stop-t1_start) 
'''