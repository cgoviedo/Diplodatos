# Importing TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Using two documents
text = [
    "Two roads diverged in a wood and I took the one less traveled by, and that has made all the difference.",
    "The best way out is always through.",
    "The best things and best people rise out of their separateness."
]

# A TfidfVectorizer object
vectorizer = TfidfVectorizer()

# This method tokenizes text and generates vocabulary
vectorizer.fit(text)

print("Generated Vocabulary:")
print(vectorizer.vocabulary_)

print("\nNumber of words in the document:")
print(sum([len(i.split()) for i in text]))

print("\nNumber of words in the vocabulary:")
print(len(vectorizer.vocabulary_))

print("\nInverse document frequency:")
print(vectorizer.idf_)

# Transforming document into a vector based on vocabulary
vector = vectorizer.transform(text)

print("\nShape of the transformed vectors:")
print(vector.shape)

print("\nVector representation of the documents:")
print(vector.toarray())
