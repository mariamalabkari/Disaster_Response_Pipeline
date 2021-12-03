# import libraries
import pandas as pd
import numpy as np
import sys
import pickle
import nltk
import re
import warnings
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.externals import joblib
from sklearn.metrics import classification_report, accuracy_score,  precision_score, recall_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
nltk.download(['punkt', 'wordnet'])
warnings.filterwarnings("ignore")


def load_data(database_filepath):
    '''
    input:
        database_filepath = File path of SQL database
    output:
    X = Features dataframe
    Y =  Target dataframe
    category_names list = Target labels 
    '''
    # load data from database
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('CleanTable', engine)
    X = df.message.values
    Y = df[df.columns[4:]].values
    category_names = list(df.columns[4:])
    return X, Y, category_names


def tokenize(text):
    '''
    input:
        text = Messages for tokenization.
    output:
        clean_tokeni = list after tokenization.
    '''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def build_model():
    pipeline = Pipeline([('cvect', CountVectorizer(tokenizer = tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC())))
    ])
    parameters = {'clf__estimator__estimator__C': [1, 2, 5]}
    cv =  GridSearchCV(pipeline, param_grid=parameters, verbose=3, cv = 5)
    return cv


def evaluate_model(model, X_test, y_test, category_names):
    # Predict categories of messages.
    y_pred = model.predict(X_test)
    # Print accuracy score, precision score, recall score and f1_score for each categories
    for i in range(0, len(category_names)):
        print(category_names[i])
        print("\tAccuracy: {:.4f}\t|| Precision: {:.4f}\t|| Recall: {:.4f}\t|| F1_score:                            {:.4f}".format(
            accuracy_score(y_test[:, i], y_pred[:, i]),
            precision_score(y_test[:, i], y_pred[:, i], average='weighted'),
            recall_score(y_test[:, i], y_pred[:, i], average='weighted'),
            f1_score(y_test[:, i], y_pred[:, i], average='weighted')
        ))

def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
