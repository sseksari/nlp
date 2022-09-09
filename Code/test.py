from collections import Counter
from unittest.util import _MAX_LENGTH
import nltk
import numpy as np
import PyPDF2 as pdf
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from rake_nltk import Rake

#1. EXTRACTION
#extracting text from pdf
mypdf = open("Seksaria_Shrishti.pdf", mode = "rb")
doc = pdf.PdfFileReader(mypdf)
corpus = ""
pages = doc.numPages
for i in range(0, pages):
    page = doc.getPage(i)
    text = page.extract_text()
    corpus += text.lower()


#2. TOKENIZATION
#tokenizing text
#tokens = nltk.word_tokenize(corpus) 
tokenizer = RegexpTokenizer(r'[A-Za-z]+')
tokens = tokenizer.tokenize(corpus)


#3. FINDING ENTITIES
#part-of-speech tagging
tags = nltk.pos_tag(tokens) #associates each word with adjective/noun/proper noun etc


#4. FREQUENCY
#frequency of each word
freq = nltk.FreqDist(tokens)


#5. KEYWORDS
#finding keywords using tf-idf (term frequency - inverse document frequency) to identify weightage - flitering 10 most important keywords
#drawbacks - not time and space efficient

unique_words = np.unique(tokens)
tf_idf = {}
tf_denom = len(tokens)

#finds frequency of each word
def find_freq(word):
    return tokens.count(word)

#checks the number of pages/documents in which the word exists
def word_exists(word):
    count = 0
    for i in range(0, pages):
        page = doc.getPage(i)
        text = page.extract_text().lower()
        if word in text:
            count+=1
    return count

for word in unique_words:
    tf = find_freq(word) / tf_denom
    idf = np.log(pages /  word_exists(word))
    tf_idf[word] =  tf * idf

tfidf = pd.DataFrame([tf_idf])
keywords = list(tfidf.T.sort_values(by = 0, ascending = False)[:10].index)

#another approach 
#finding key phrases & key words

r = Rake(include_repeated_phrases=False, min_length = 1, max_length = 3)
s = Rake(include_repeated_phrases=False, min_length = 1, max_length = 1)

r.extract_keywords_from_text(corpus)
s.extract_keywords_from_text(corpus)

phrases = r.get_ranked_phrases_with_scores()
words = s.get_ranked_phrases_with_scores()

table1 = pd.DataFrame(phrases,columns=['score','Phrase']).sort_values('score',ascending=False)
table2 = pd.DataFrame(words,columns=['score','Word']).sort_values('score',ascending=False)

key_phrases = list(table1["Phrase"][:10])
key_words = list(table2["Word"][:10])


