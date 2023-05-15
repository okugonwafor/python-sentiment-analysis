#SIMPLE SENTIENT TEXT ANALYSIS IN PYTHON

from textblob import TextBlob
from newspaper import Article
import nltk
nltk.download('punkt')
from nltk import word_tokenize

url = 'https://metro.co.uk/2023/05/15/double-murder-inquiry-launched-after-man-and-woman-found-dead-in-house-18788155/'
article = Article(url)

article.download()
article.parse()
article.nlp()

text = article.text
#we could use article.summary and shows a summary of what is written in the article.
print(text)

blob = TextBlob(text)
sentiment = blob.sentiment.polarity # -1 to 1
print(sentiment) 
