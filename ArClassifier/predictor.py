import os

import numpy as np
import pandas as pd
from .Preprocessing import extractor
from joblib import dump, load
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer  # to create Bag of words
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score  # to calculate accuracy
from sklearn.model_selection import train_test_split  # for splitting data
from sklearn.naive_bayes import MultinomialNB  # to bulid classifier model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder  # to convert classes to number

from Text_Classification.settings import DATA_DIR, JOBLIB_DIR

count = CountVectorizer()
encoder = LabelEncoder()


def split_data():
    data = pd.read_csv(os.path.join(DATA_DIR, 'nada.csv'), encoding='utf-8')
    data.dropna(inplace=True)
    train, test = train_test_split(data, test_size=0.2)
    return train, test


def pre_processing():
    train, test = split_data()
    if not os.path.isfile(os.path.join(JOBLIB_DIR, 'count_vector.joblib')):
        X = count.fit_transform(train['text'].values.astype('U')).toarray()
        dump(count.fit(train['text']), os.path.join(JOBLIB_DIR, 'count_vector.joblib'))
    else:
        train_df_vectorized = load(os.path.join(JOBLIB_DIR, 'count_vector.joblib'))
        X = train_df_vectorized.fit_transform(train['text'].values.astype('U')).toarray()
    if not os.path.isfile(os.path.join(JOBLIB_DIR, 'label_encoder.joblib')):
        y = encoder.fit_transform(train['classe'])
        dump(encoder.fit(train['classe']), os.path.join(JOBLIB_DIR, 'label_encoder.joblib'))
    else:
        train_encoder = load(os.path.join(JOBLIB_DIR, 'label_encoder.joblib'))
        y = train_encoder.fit_transform(train['classe'])
    return [X, y]


def predict(model, count_vector, label_enc, text_to_predict):
    keywords = {}
    terms = extractor(text_to_predict)
    keywords.append(terms)
    # convert to number
    test_vector = count_vector.transform(keywords).toarray()

    # encodeing predict class
    text_predict_class = label_enc.inverse_transform(model.predict(test_vector))
    return text_predict_class[0], keywords


def train_naive_bayes():
    # load data
    data, test = pre_processing()
    # create model
    clfrNB = MultinomialNB(alpha=0.1)
    clfrNB.fit(data[0], data[1])
    # save model
    dump(clfrNB, os.path.join(JOBLIB_DIR, 'NBmodel.joblib'))


def predict_naive_bayes(text_to_predict):
    if not os.path.isfile(os.path.join(JOBLIB_DIR, 'NBmodel.joblib')) or not os.path.isfile(
            os.path.join(JOBLIB_DIR, 'count_vector.joblib')) or not os.path.isfile(
            os.path.join(JOBLIB_DIR, 'label_encoder.joblib')):
        train_naive_bayes()

    loaded_model = load(os.path.join(JOBLIB_DIR, 'NBmodel.joblib'))
    loaded_count_vector = load(os.path.join(JOBLIB_DIR, 'count_vector.joblib'))
    loaded_label = load(os.path.join(JOBLIB_DIR, 'label_encoder.joblib'))

    return predict(loaded_model, loaded_count_vector, loaded_label, text_to_predict), evaluate(loaded_model,
                                                                                               loaded_count_vector,
                                                                                               loaded_label)


def train_knn():
    data = pre_processing()
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(data[0], data[1])
    dump(model, os.path.join(JOBLIB_DIR, 'KNNmodel.joblib'))


def predict_knn(text_to_predict):
    if not os.path.isfile(os.path.join(JOBLIB_DIR, 'KNNmodel.joblib')) or not os.path.isfile(
            os.path.join(JOBLIB_DIR, 'count_vector.joblib')) or not os.path.isfile(
            os.path.join(JOBLIB_DIR, 'label_encoder.joblib')):
        train_knn()
    loaded_model = load(os.path.join(JOBLIB_DIR, 'KNNmodel.joblib'))
    loaded_count_vector = load(os.path.join(JOBLIB_DIR, 'count_vector.joblib'))
    loaded_label = load(os.path.join(JOBLIB_DIR, 'label_encoder.joblib'))

    return predict(loaded_model, loaded_count_vector, loaded_label, text_to_predict), evaluate(loaded_model,
                                                                                               loaded_count_vector,
                                                                                               loaded_label)


def train_svm():
    data = pre_processing()
    Svm = svm.LinearSVC()
    Svm.fit(data[0], data[1])
    dump(svm, os.path.join(JOBLIB_DIR, 'SVMmodel.joblib'))


def predict_svm(text_to_predict):
    if not os.path.isfile(os.path.join(JOBLIB_DIR, 'SVMmodel.joblib')) or not os.path.isfile(
            os.path.join(JOBLIB_DIR, 'count_vector.joblib')) or not os.path.isfile(
            os.path.join(JOBLIB_DIR, 'label_encoder.joblib')):
        train_svm()

    loaded_model = load(os.path.join(JOBLIB_DIR, 'SVMmodel.joblib'))
    loaded_count_vector = load(os.path.join(JOBLIB_DIR, 'count_vector.joblib'))
    loaded_label = load(os.path.join(JOBLIB_DIR, 'label_encoder.joblib'))

    return predict(loaded_model, loaded_count_vector, loaded_label, text_to_predict), evaluate(loaded_model,
                                                                                               loaded_count_vector,
                                                                                               loaded_label)


def evaluate(loaded_model, loaded_count_vector, loaded_label):
    metrics = {}
    train, test = split_data()
    y_pred = loaded_model.predict(loaded_count_vector.transform(test['text'].values.astype('U')).toarray())
    y_test = loaded_label.transform(test['classe'])
    # getting metrics
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    metrics['recall'] = recall_score(y_test, y_pred, average='weighted')
    metrics['precision'] = precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    metrics['f1_score'] = f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
    return metrics
