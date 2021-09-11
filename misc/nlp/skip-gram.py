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
                               window=10,
                               workers=2,
                               seed=100,
                               sg=1)

# Word embeddings for the word "minister" by the trained skip-gram model
print("Length of the word vector:",len(model.wv["minister"]))
print("Word embedding for the word 'minister':",model.wv["minister"])


print("Words most similar to 'minister':",model.wv.most_similar("minister"))
