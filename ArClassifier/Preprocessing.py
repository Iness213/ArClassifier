import math
import operator
import os
import re
import string

import nltk
import numpy as np
import networkx as nx
from arabic_reshaper import arabic_reshaper
from bidi.algorithm import get_display
from nltk.tag import pos_tag
from pyarabic.araby import tokenize
import matplotlib.pyplot as plt

from Text_Classification.settings import BASE_DIR

nltk.download('averaged_perceptron_tagger')
from nltk.tag import StanfordPOSTagger as pos_tag
from tashaphyne.stemming import ArabicLightStemmer

arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
english_punctuations = string.punctuation
punctuations_list = arabic_punctuations + english_punctuations

arabic_diacritics = re.compile("""
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)


def clean_text(text):
    text = re.sub("[a-zA-Z]", " ", text)  # remove english letters
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    text = re.sub(arabic_diacritics, '', text)
    # remove_punctuations(text)
    translator = str.maketrans('', '', punctuations_list)
    text = text.translate(translator)
    # remove_repeating_char(text)
    text = re.sub(r'(.)\1+', r'\1', text)
    text = re.sub(r'\n', ' ', text)  # remove empty lines
    text = text.strip()  # remove whitespaces
    return text


def make_tagger():
    java_path = os.path.join(BASE_DIR, os.path.join('jdk', 'bin'))
    os.environ['JAVAHOME'] = java_path
    stanford_dir = os.path.join(BASE_DIR, 'stanford_tagger')
    jar_file = os.path.join(stanford_dir, 'stanford-postagger.jar')
    model_file = os.path.join(stanford_dir, os.path.join('models', 'arabic.tagger'))
    return pos_tag(model_filename=model_file, path_to_jar=jar_file)


# The tokenized text (mainly the nouns and adjectives) is normalized by lemmatization
def stemming(pos_tag):
    Ar_Listem = ArabicLightStemmer()
    adjective_tags = ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS']
    stemmed_text = []

    for word in pos_tag:
        p = word[1].split('/')
        if p[-1] in adjective_tags:
            stemmed_text.append(str(Ar_Listem.light_stem(p[0])))
        else:
            stemmed_text.append(str(Ar_Listem.light_stem(p[0])))
    # print("Text tokens after lemmatization of adjectives and nouns: \n")
    return stemmed_text


# parametres : pos_tags : of stemmed text
#              token : stemmed text

# Any word from the lemmatized text, which isn't a noun, adjective, or gerund
def remove_stopwords(pos_tag, token):
    # stopwords generation using pos tags and arabic stop words list
    stopwords = []
    lots_of_stopwords = []
    stopwords_plus = []
    processed_text = []
    file_path = os.path.join(BASE_DIR, 'stop_words.txt')
    stopword_file = open(file_path, "r", encoding='utf-8')

    wanted_POS = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS', 'VBG', 'FW']

    for word in pos_tag:
        p = word[1].split('/')
        if p[-1] not in wanted_POS:
            stopwords.append(p[0])
    punctuations = list(str(string.punctuation))
    stopwords = stopwords + punctuations

    for line in stopword_file.readlines():
        lots_of_stopwords.append(str(line.strip()))

    stopwords_plus = stopwords + lots_of_stopwords
    stopwords_plus = set(stopwords_plus)

    # Removing stopwords from stemmed_text
    for word in token:
        if word not in stopwords_plus:
            processed_text.append(word)
    return processed_text, stopwords_plus


def finale_preprocess(text):  # returns preprocessed text without stopwords, english lettres
    tagger = make_tagger()
    text = clean_text(text)
    token = tokenize(text)
    POS_tag = tagger.tag(token)
    stemmed = stemming(POS_tag)
    pos = tagger.tag(stemmed)
    preprocessed_text, stopwords = remove_stopwords(pos, stemmed)
    preprocessed_text_string = ' '.join(word for word in preprocessed_text)
    preprocessed_text_string = re.sub("[a-zA-Z]", "", preprocessed_text_string)  # remove english letters
    return preprocessed_text_string


def vocab_creation(processed_text):
    vocabulary = list(set(processed_text))
    return vocabulary


def build_graph(processed_text, vocabulary, window_size=3):
    vocab_len = len(vocabulary)

    weighted_edge = np.zeros((vocab_len, vocab_len), dtype=np.float32)

    score = np.zeros(vocab_len, dtype=np.float32)
    window_size = 3
    covered_coocurrences = []

    for i in range(0, vocab_len):
        score[i] = 1
        for j in range(0, vocab_len):
            if j == i:
                weighted_edge[i][j] = 0
            else:
                for window_start in range(0, (len(processed_text) - window_size)):

                    window_end = window_start + window_size

                    window = processed_text[window_start:window_end]

                    if (vocabulary[i] in window) and (vocabulary[j] in window):

                        index_of_i = window_start + window.index(vocabulary[i])
                        index_of_j = window_start + window.index(vocabulary[j])

                        # index_of_x is the absolute position of the xth term in the window
                        # (counting from 0)
                        # in the processed_text

                        if [index_of_i, index_of_j] not in covered_coocurrences:
                            weighted_edge[i][j] += 1 / math.fabs(index_of_i - index_of_j)
                            covered_coocurrences.append([index_of_i, index_of_j])
    return weighted_edge, score


def summation(vocabulary, weighted_edge):
    vocab_len = len(vocabulary)
    inout = np.zeros(vocab_len, dtype=np.float32)

    for i in range(0, vocab_len):
        for j in range(0, vocab_len):
            inout[i] += weighted_edge[i][j]
    return inout


def scores(inout, weighted_edge, vocabulary, score):
    MAX_ITERATIONS = 50
    d = 0.85
    threshold = 0.0001  # convergence threshold
    vocab_len = len(vocabulary)
    for iter in range(0, MAX_ITERATIONS):
        prev_score = np.copy(score)

        for i in range(0, vocab_len):

            summation = 0
            for j in range(0, vocab_len):
                if weighted_edge[i][j] != 0:
                    summation += (weighted_edge[i][j] / inout[j]) * score[j]

            score[i] = (1 - d) + d * summation

        if np.sum(np.fabs(prev_score - score)) <= threshold:  # convergence condition
            print("Converging at iteration " + str(iter) + "....")
            break
    return score


def extractor(text, user_id, file_name):
    keys = {}
    tagger = make_tagger()
    text = clean_text(text)
    token = tokenize(text)
    POS_tag = tagger.tag(token)
    stemmed = stemming(POS_tag)
    pos = tagger.tag(stemmed)
    preprocessed_text, stopwords = remove_stopwords(pos, stemmed)
    vocabulary = vocab_creation(preprocessed_text)
    weighted_edge, score = build_graph(preprocessed_text, vocabulary)
    inout = summation(vocabulary, weighted_edge)
    scor = scores(inout, weighted_edge, vocabulary, score)
    for i in range(0, len(vocabulary)):
        sc = str(scor[i])
        word = vocabulary[i]
        keys[word] = sc
    sort = dict(sorted(keys.items(), key=operator.itemgetter(1), reverse=True)[:20])
    lis = ''
    for k in sort.keys():
        lis = lis + str(k) + ' '
    terms = lis
    terms = re.sub("[a-zA-Z]", "", terms)  # remove english letters
    keys.clear()
    sort.clear()
    edges_list = edges(weighted_edge, vocabulary)
    draw_graph(edges_list, user_id, file_name)
    return terms


def edges(A, vocab):
    edges = []
    for i in range(len(A)):
        for j in range(i + 1, len(A[0])):
            if A[i][j] > 0:
                edges.append((vocab[i], vocab[j]))
    return edges


def draw_graph(edgeList, user_id, file_name):
    lis = []
    for i in edgeList:
        reshaped_text = arabic_reshaper.reshape(i[0])
        artext = get_display(reshaped_text)
        reshaped_text1 = arabic_reshaper.reshape(i[1])
        artext1 = get_display(reshaped_text1)
        lis.append((artext, artext1))
    G = nx.OrderedMultiDiGraph()
    G.add_edges_from(lis)
    pos = nx.spring_layout(G, k=0.85, iterations=10)
    nx.draw_networkx_nodes(G, pos, node_color='#AED6F1', node_size=2500)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='#95A5A6', arrows=True)
    nx.draw_networkx_labels(G, pos, font_size=20, font_family='Times New Roman')
    graph_path = os.path.join(os.path.join(BASE_DIR, 'static/media'), 'graph')
    user_path = os.path.join(graph_path, str(user_id))
    if not os.path.isdir(graph_path):
        os.mkdir(graph_path)
    if not os.path.isdir(user_path):
        os.mkdir(user_path)
    path = os.path.join(user_path, file_name)
    plt.tight_layout()
    plt.savefig(path, format="PNG")
    plt.show()
