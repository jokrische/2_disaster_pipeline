import sys

import nltk
nltk.download('punkt')
nltk.download('wordnet')

from sqlalchemy import create_engine
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import pickle


def load_data(database_filepath):
    '''
    INPUT: filepath of database
    OUTPUT: X: textmessages from the dataframe, Y: column names, categorie_names: 
    categories names from the dataframe
    '''
    CONNECTION_STRING = f"sqlite:///{database_filepath}"
    engine = create_engine(CONNECTION_STRING)
    df = pd.read_sql_table('disaster_data', engine)

    X = df["message"]
    Y = df.iloc[:,4:]
    categorie_names = Y.columns
    return X, Y, categorie_names


def tokenize(text):
    '''
    INPUT: text
    OUTPUT: clean_tokens: list with cleaned words of text
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(tok)
        
    return clean_tokens


def build_model():
    '''
    INPUT: 
    OUTPUT: cv: GridSearch pipeline and parameters
    '''
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer = tokenize, max_df = 0.5)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultiOutputClassifier(RandomForestClassifier()))                
                      ])

    parameters = {
            #'vect__ngram_range': ((1, 1), (1, 2)),
            #'vect__max_df': (0.5),
            'vect__max_features': (None, 5000, 10000),
            'clf__estimator__n_estimators': [10, 100],
            'clf__estimator__min_samples_leaf': [2, 4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT: model, X_test, Y_test, categorie_names
    OUTPUT: printed classification report of model
    '''
    y_pred = model.predict(X_test)
    for col in category_names:
        print(classification_report(y_test.iloc[:,col], y_pred[:,col]))


def save_model(model, model_filepath):
    '''
    INPUT: model, modelfilepath (with .pkl ending)
    OUTPUT:
    '''
    pickle.dump(model, open(model_filepath, 'wb'))



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