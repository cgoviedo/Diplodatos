# Importing NLTK
import nltk
from nltk import sent_tokenize, word_tokenize
# Download the abc corpus
nltk.download('punkt')
nltk.download('abc')
# Importing the abc corpus
from nltk.corpus import abc

# Data Preparation
data = []
for i in sent_tokenize(abc.raw()):
    temp = []

    # tokenize the sentence into words
    for j in word_tokenize(i):
        temp.append(j.lower())

    data.append(temp)

# First 5 sentences of data prepared for Word2Vec
print("Data sample:")
print(data[:5])
print("_________________________________________________")



import gensim

# Training model
model = gensim.models.Word2Vec(data, min_count=1,
                               size=100,
                               window=5,
                               workers=2,
                               seed=100)

# Word embeddings for the word "prime" by the trained model
print("Length of the word vector:",len(model.wv["prime"]))
print("Word embedding for the word 'prime':",model.wv["prime"])
