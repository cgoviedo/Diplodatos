# Importing NLTK
import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import abc

# Data Preparation
data = []
for i in sent_tokenize(abc.raw()):
    temp = []

    # tokenize the sentence into words
    for j in word_tokenize(i):
        temp.append(j.lower())

    data.append(temp)

# Training model
import gensim
model = gensim.models.Word2Vec(data, min_count=1,
                               size=1000,
                               window=5,
                               workers=2,
                               seed=100)

print("Words most similar to 'china':",model.wv.most_similar("china"))
