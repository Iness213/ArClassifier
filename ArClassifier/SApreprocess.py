import re  # for pre-processing text
import string  # for pre-processing text
import nltk  # for processing texts
from nltk.corpus import stopwords
from tashaphyne.stemming import ArabicLightStemmer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# define arabic punctuations
arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
english_punctuations = string.punctuation
punctuations_list = arabic_punctuations + english_punctuations
st = nltk.ISRIStemmer()
Ar_Listem = ArabicLightStemmer()


def clean_text(text):
    """
        DESCRIPTION:
        This function to clean text
        INPUT:
        text: string
        OUTPUT:
        text: string after clean it
    """
    # remove english letters
    text = re.sub("[a-zA-Z]", " ", str(text))
    # remove \n from text
    text = re.sub('\n', ' ', text)
    # remove number
    text = re.sub(r'\d+', '', text)
    # remove links
    text = re.sub(r'http\S+', '', text)
    # remove punctuation
    text = text.translate(str.maketrans('', '', punctuations_list))
    # remove stop word
    text = ' '.join([word for word in text.split() if word not in stopwords.words("arabic")])
    # remove extra space
    text = re.sub(' +', ' ', text)
    # remove whitespaces
    text = text.strip()
    # stemming
    stemmed_words = []
    words = nltk.word_tokenize(text)
    for w in words:
        stemmed_words.append(str(Ar_Listem.light_stem(w)))
    stemmed_sentence = " ".join(stemmed_words)
    return stemmed_sentence