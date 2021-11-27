#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import libraries
import nltk
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import pickle
import re
from sklearn.svm import SVC
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix, classification_report, hamming_loss
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])


# In[ ]:


# load data from database
engine = create_engine('sqlite:////content/sample_data/Database.db')
df = pd.read_sql("SELECT * FROM DatabaseTable", engine)
df.related.replace(2,1,inplace=True)
X = df['message']
Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
#/content/sample_data/Database.db


# In[ ]:


def tokenize(text):
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
       
        # Remove stop words
        if tok in stopwords.words("english"):
            continue
        # Reduce words to their stems
        tok = PorterStemmer().stem(tok)
        # Reduce words to their root form
        tok = lemmatizer.lemmatize(tok).lower().strip()

        clean_tokens.append(tok)
        
    clean_tokens = [tok for tok in clean_tokens if tok.isalpha()]
    return clean_tokens

print(X[3])
print(tokenize(X[3]))


# In[ ]:


# Build a machine learning pipeline
pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])


# In[ ]:


# Train pipeline
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
 # train classifier
pipeline.fit(X_train, Y_train)


# In[ ]:


# Test your model
Y_pred = pipeline.predict(X_test)

for ix, col in enumerate(Y.columns):
    print(col)
    print(classification_report(Y_test[col], Y_pred[:,ix]))

avg = (Y_pred == Y_test).mean().mean()
print("Accuracy Overall:\n", avg)


# In[ ]:


# Improve model
pipeline.get_params

parameters = {
        'vect__max_df':[1, 5],
        'clf__estimator__n_estimators': [20, 50]
    }


cv = GridSearchCV(pipeline, param_grid=parameters, verbose=10)
cv.fit(X_train, Y_train)


# In[ ]:


# test model

Y_pred = cv.predict(X_test)
for ix, col in enumerate(Y.columns):
    print(col)
    print(classification_report(Y_test[col], Y_pred[:,ix]))

avg = (Y_pred == Y_test).mean().mean()
print("Accuracy Overall:\n", avg)


# In[ ]:


# improve model
# using SVM instead 
pipe2 = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(SVC()))
])

parameters2 = {'vect__min_df': [5],
              'tfidf__use_idf':[True],
              'clf__estimator__kernel': ['poly'], 
              'clf__estimator__degree': [1, 2, 3],
              'clf__estimator__C':[1, 10, 100]}

cv2 = GridSearchCV(pipe2, param_grid = parameters2, verbose = 10)

# Find best parameters
np.random.seed(77)
T_model2 = cv2.fit(X_train, Y_train)


# In[ ]:


# grid search results
T_model2.cv_results_


# In[ ]:


# Calculate evaluation metrics for test set
tuned_pred_test2 = T_model2.predict(X_test)

eval_metrics2 = eval_metrics(np.array(Y_test), tuned_pred_test2, col_names)

print(eval_metrics2)


# In[ ]:


# Export model as a pickle file
pickle.dump(cv, open("classifier.pkl", 'wb'))

