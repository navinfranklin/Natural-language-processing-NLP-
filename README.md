# Natural-language-processing-NLP-
most important features which are important in data science are only practiced. these techniques will be very useful in converting the sentence or words into numerical values into order to make the machine understand.it would be more easy to perform machine learning models.

understanding each program is very important

tokenization is the process of converting the paragragh into sentence and sentence into words.
for that we use
import nltk
which is very much used for natural language processing 
nltk.sent_tokenize(paragraph)
this statement will give you paragraph into sentence
nltk.word_tokenize(paragraph)
this will give you the sentence into words
this is all about the tokenization

stemming
it is a process of reducing infect words to thier word stem(it will not give correct word or meaning in some cases

libaray
import nltk
it is the porterstemmer for the purpose of stemming
from nltk.stem import PorterStemmer
stopwords is used for removing the not important words like example is,are,us,they,them and extra
from nltk.corpus import stopwords

tokenize the paragraph so it will be usefull
sentences = nltk.sent_tokenize(paragraph)
assigning a variable  and having stemming function init
stemmer = PorterStemmer()
performing a for loop lengh of the sentence
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)   
 then create a variable for words and store the word tokenization and perform list comperhension
 then join the sentence.
 
 lemmatization
 it is the same process as of stemming but it will give you a proper meaningfor the words.
 
 bag of words 
 there are 2 types
 binary and bag of words
 step 1 is lowering the sentecne 
 setp 2 is we can perform stemming or lemmatization acoording the to the size of the data set
 step 3 is stopwords
 
 disadvange is we dont the the importance of the good used.in order to know the importance of the words we use TF-IDF instead of bag of words
we use re -which is the regualar expression
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
wordnet=WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)
corpus = []
for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()

we use corpus because finally we can comapare both the senntence and the final sentence.
review = re.sub('[^a-zA-Z]', ' ', sentences[i])
this is used to remove all the extra spaces regualar expressions instead of a-zA-Zand spaces in the sentence
review = review.lower()
it changes all the sentecne into the lower case
    review = review.split()
    it just splits the sentecne into the words
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    this is the list comperhensions
from sklearn.feature_extraction.text import CountVectorizer
in this count vectorizer all the processing happens which is the conversion of categorical variables into the numerical variables which is zeros and ones.

TF-IDF
it is same as the bag os words but it gives the importance of the words.
in bag of words we use countvectorizer instead of that we use tfidfVectorisor which we convert the categorical values into numerical with the importance of the word.
bag of words will give 1s and 0s inth TF-IDF we get values which are in decimals. so we can easliy identify the importance of the words.
