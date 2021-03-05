import re, collections
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import wordpunct_tokenize, pos_tag, ne_chunk, word_tokenize
from nltk.util import ngrams

#opening out file
outfile = open('input.txt', 'w')
#setting website variable = to the webpage request
website = requests.get("https://en.wikipedia.org/wiki/Google")
#using beautiful soup to parse the webpage content
soup = BeautifulSoup(website.content,"html.parser")
#finding all p tags and setting to variable
paragraphs = soup.select("p")
#outputting all content in paragraphs to outfile
for paragraph in paragraphs:
    outfile.write(paragraph.text)
outfile.close()

###Tokenizing ------------------------------
textfromfile = (open("input.txt").read())
WORDS = nltk.word_tokenize(textfromfile)
#print ("Tokenized ---------------------------------------------------------")
#print (WORDS)

###PoS Tagging ------------------------------
#print ("Parts of Speach -----------------------------------------------------")
#print (nltk.pos_tag(WORDS))

###Stemming (Porter) ------------------------
#print ("Stemming ------------------------------------------------------------")
#pstem = PorterStemmer()
#for i in WORDS:
#    print (pstem.stem(i))

###Lemmatization ----------------------------
#print ("Lemmatization -------------------------------------------------------")
#lemm = WordNetLemmatizer()
#for i in WORDS:
#    print (lemm.lemmatize(i))

###Trigram ----------------------------------
#print ("Trigram -------------------------------------------------------------")
#trigram = ngrams(WORDS,3)
#for i in trigram:
#    print (i)

###Named Entity Recognition -----------------
print ("NER -----------------------------------------------------------------")
print (ne_chunk(pos_tag(wordpunct_tokenize(textfromfile))))
