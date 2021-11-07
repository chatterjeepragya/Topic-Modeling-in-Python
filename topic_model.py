"""
The following code run LDA topic model on reuters data from nltk
reference: https://radimrehurek.com/gensim/models/ldamodel.html#gensim.models.ldamodel.LdaModel
"""
from nltk.corpus import reuters, stopwords
from nltk import word_tokenize, sent_tokenize
from nltk.stem.snowball import SnowballStemmer
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel

# a set of English stop words in NLTK
stop_words = set(stopwords.words('english'))
# get a stemmer 
stemmer = SnowballStemmer("english")


# define a function to tokenize a raw text of a document and remove stopwords
# input is a raw string
# out put is a list of tokens
def tokenize_stem_stopwords(doc):
    tokens = []  # a list of tokens in this document
    doc = doc.lower()  # normalize to lower case
    for sent in sent_tokenize(doc):  # for each sentence
        for word in word_tokenize(sent):  # for each word in sentence
            if word.isalpha() and (word not in stop_words):  # remove stop words and non-alphabetic words
                tokens.append(stemmer.stem(word))  # add stemmed word to token list
    return tokens


# get the raw text as a list of string, data source from NLTK
documents = [reuters.raw(fileid) for fileid in reuters.fileids()]
# tokenized documents, a list of list of tokens
tokenized_docs = []
for doc in documents:
    tokenized_docs.append(tokenize_stem_stopwords(doc))
# create a dictionary in gensim - a dictionary is a mapping between term id to term string
dictionary = Dictionary(tokenized_docs)
# filter extreme tokens in the dictionary
dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=100000)
# obtain corpus in bag-of-word representation
corpus = [dictionary.doc2bow(words) for words in tokenized_docs]
# train LDA model on corpus
num_topics = 50
lda = LdaModel(corpus, num_topics=num_topics, id2word=dictionary)
# print the most contributing words for each topic
for topic in lda.show_topics(num_topics=num_topics, num_words=15):
    print(topic)
