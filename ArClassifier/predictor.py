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
from .models import Metric

count = CountVectorizer()
encoder = LabelEncoder()


def pre_processing(dataset_file):
    data = pd.read_csv(os.path.join(DATA_DIR, dataset_file), encoding='utf-8')
    data = data.fillna(' ')
    if not os.path.isfile(os.path.join(JOBLIB_DIR, 'count_vector.joblib')):
        X = count.fit_transform(data['text'].values.astype('U')).toarray()
        file = os.path.join(JOBLIB_DIR, 'count_vector.joblib')
        dump(count.fit(data['text']), open(file, "wb"))
    else:
        train_df_vectorized = load(open(os.path.join(JOBLIB_DIR, 'count_vector.joblib'), 'rb'))
        X = train_df_vectorized.fit_transform(data['text'].values.astype('U')).toarray()
    if not os.path.isfile(os.path.join(JOBLIB_DIR, 'encoder.joblib')):
        y = encoder.fit_transform(data['classe'])
        dump(encoder.fit(data['classe']), open(os.path.join(JOBLIB_DIR, 'encoder.joblib'), 'wb'))
    else:
        train_encoder = load(open(os.path.join(JOBLIB_DIR, 'encoder.joblib'), 'rb'))
        y = train_encoder.fit_transform(data['classe'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, y_train, X_test, y_test


def train_svm(dataset_file):
    X_train, y_train, X_test, y_test = pre_processing(dataset_file)
    model = svm.LinearSVC()
    model.fit(X_train, y_train)
    dump(model, os.path.join(JOBLIB_DIR, 'SVM_model.joblib'))

    svm_metric = Metric.objects.filter(name='SVM')
    if not len(svm_metric) > 0:
        # Predicting the Test set results
        y_pred = model.predict(X_test)
        # getting metrics
        metrics = {'accuracy': accuracy_score(y_test, y_pred),
                   'recall': recall_score(y_test, y_pred, average='weighted'),
                   'precision': precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)),
                   'f1_score': f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))}
        svm_metric = Metric(algorithm='SVM',
                            accuracy=Metric['accuracy'],
                            recall=Metric['recall'],
                            precision=Metric['precision'],
                            f1_score=Metric['f1_score'])
        svm_metric.save()


def train_knn(dataset_file, k):
    X_train, y_train, X_test, y_test = pre_processing(dataset_file)
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    dump(model, os.path.join(JOBLIB_DIR, 'KNN_model.joblib'))

    knn_metric = Metric.objects.filter(name='KNN')
    if not len(knn_metric) > 0:
        # Predicting the Test set results
        y_pred = model.predict(X_test)
        # getting metrics
        metrics = {'accuracy': accuracy_score(y_test, y_pred),
                   'recall': recall_score(y_test, y_pred, average='weighted'),
                   'precision': precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)),
                   'f1_score': f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))}
        knn_metric = Metric(algorithm='KNN',
                            accuracy=Metric['accuracy'],
                            recall=Metric['recall'],
                            precision=Metric['precision'],
                            f1_score=Metric['f1_score'])
        knn_metric.save()


def train_naive_bayes(dataset_file):
    X_train, y_train, X_test, y_test = pre_processing(dataset_file)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    dump(model, os.path.join(JOBLIB_DIR, 'Naive_Bayes_model.joblib'))

    nb_metric = Metric.objects.filter(algorithm='Naive Bayes')
    if not len(nb_metric) > 0:
        # Predicting the Test set results
        y_pred = model.predict(X_test)
        # getting metrics
        metrics = {'accuracy': accuracy_score(y_test, y_pred),
                   'recall': recall_score(y_test, y_pred, average='weighted'),
                   'precision': precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)),
                   'f1_score': f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))}
        nb_metric = Metric(algorithm='Naive Bayes',
                           accuracy=metrics.get('accuracy'),
                           recall=metrics.get('recall'),
                           precision=metrics.get('precision'),
                           f1_score=metrics.get('f1_score'))
        nb_metric.save()


def predict(text_to_predict, algorithm, dataset_file='nada.csv', k=5):
    if not os.path.isfile(os.path.join(JOBLIB_DIR, algorithm + '_model.joblib')) or not os.path.isfile(os.path.join(
            JOBLIB_DIR, 'encoder.joblib')) or not os.path.isfile(os.path.join(JOBLIB_DIR, 'count_vector.joblib')):
        if algorithm == 'SVM':
            train_svm(dataset_file)
        elif algorithm == 'KNN':
            train_knn(dataset_file, k)
        else:
            train_naive_bayes(dataset_file)

    model = load(os.path.join(JOBLIB_DIR, algorithm + '_model.joblib'))
    encoder_vector = load(os.path.join(JOBLIB_DIR, 'encoder.joblib'))
    count_vector = load(os.path.join(JOBLIB_DIR, 'count_vector.joblib'))

    terms = extractor(text_to_predict)
    keywords = [terms]

    # convert to number
    test_vector = count_vector.transform(keywords).toarray()

    # encoding predict class
    category = encoder_vector.inverse_transform(model.predict(test_vector))

    return category[0], keywords