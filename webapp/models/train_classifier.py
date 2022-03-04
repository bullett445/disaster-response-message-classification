import sys
import nltk
nltk.download(['punkt', 'wordnet'])
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):
    """
    Load data from sqlite db and return X, Y and the category names.
    :param database_filepath: path to the sqlite file
    :return: X, Y, category_names of the entire dataset
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages', engine)
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    Tokenize, clean and lemmatize a text
    :param text: The text to process
    :return: array of clean tokens
    """
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum() and word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Asemble model pipeline and grid search to be used for fitting
    :return: GridSearchCV
    """
    pipeline = Pipeline(
        [
            ('vectorize', CountVectorizer(tokenizer=tokenize)),
            ('tfidtransform', TfidfTransformer()),
            ('classify', MultiOutputClassifier(RandomForestClassifier()))
        ])

    parameters = [
        {'classify__estimator': [RandomForestClassifier()], 'classify__estimator__n_estimators': [20, 50]},
        {'classify__estimator': [AdaBoostClassifier()], 'classify__estimator__n_estimators': [20, 50]},
        {'classify__estimator': [MultinomialNB()]}
    ]

    cv = GridSearchCV(pipeline, parameters, verbose=2, cv=3, n_jobs=1)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Print a classification_report for each category to evaluate the model
    :param model: model to be evaluated
    :param X_test: test data features
    :param Y_test: test data categories
    :param category_names: array of category names
    :return: None
    """
    Y_test_pred = model.predict(X_test)
    Y_test_pred = pd.DataFrame(Y_test_pred, columns=category_names)
    for c in category_names:
        print('category: ', c)
        print(classification_report(Y_test[c], Y_test_pred[c], zero_division=0))


def save_model(model, model_filepath):
    """
    Save the model into a serialized object
    :param model: the model to be saved
    :param model_filepath: path to save the model to
    :return: None
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


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

        print('Picking best model from gridsearch...')
        model = model.best_estimator_

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