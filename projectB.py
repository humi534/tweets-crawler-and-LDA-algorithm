#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import re
import nltk
import json
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
nltk.download('punkt')
import tweepy
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
nltk.download('wordnet')
import gensim
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

consumer_key = 't7JkWdi85BVHiyRETXxKhAZVw'
consumer_secret = 'dd4CV3evkURrThiqDOGD3QCaY2BhUXboqONfDHcTtzrSGJ9ebb'
access_token = '1272816278319792128-vJcUJMq188VV61bEYMU1wP2hn9DlN8'
access_secret = 'cWFbORJfT7eh8lYxQ2tuIG6DWVMELjUtJA9Mfi87iF3ZM'


#This is a basic listener that just prints received tweets to stdout.
class StdOutListener(StreamListener):
    
    f = open('collectedTweets.txt', 'a')
    def __init__(self):
      StreamListener.__init__(self)
      self.max_tweets = 5000
      self.tweet_count = 0

    def on_data(self, data):
        if (self.tweet_count == self.max_tweets):
            print("Terminated")
            return False
        else:
            self.tweet_count += 1
            #print(data)
            f.write(data)
            return True

    def on_error(self, status):
        print(status)


def downloadTweets():
    
    
    
    #This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    stream = Stream(auth, l)

    #This line filter Twitter Streams to capture data by the keywords: 
    stream.filter(track=['science'], languages=["en"])


def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

def remove_stopwordsAndTokenize(text): 
    all_stopwords = stopwords.words('english')
    #sw_to_add = ['""',"''",":",'``','’','-',",",'|',".",'”', '“',"?","&","!","'"]
    #all_stopwords.extend(sw_to_add)
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in all_stopwords]
    return tokens_without_sw

def clean_text(text):
    text = re.sub(r"\S*https?:\S*", "", text) #delete web links
    text = remove_emoji(text)
    text = re.sub("\n"," ",text) #delete back to the line
    text = text.lower()
    text = re.sub("rt\s*@[^:]*: ","",text) #delete rt and contact
    #text = re.sub("@[a-zA-Z0-9-]\S* ","",text) #remove all contact
    text = re.sub("@[a-zA-Z0-9-]\S*","",text) #remove all contact
    #text = re.sub("#[a-zA-Z0-9-]\S*","",text) #remove all hashtags
    #text = re.sub("#","",text) #remove all hashtags
    text = re.sub("r&amp;d","research and development",text)
    text = re.sub("&amp;","&",text)
    text = re.sub('[^a-zA-Z0-9 \n\.]', ' ', text)
    text = re.sub('[.]', ' ', text)
    return text



def lemmatize(token_list):
    wordnet_lemmatizer = WordNetLemmatizer()
    token_list = [wordnet_lemmatizer.lemmatize(word, pos="v") for word in token_list]
    return token_list


def prepare_data_for_LDA(text):
    text = clean_text(text)
    tokens = remove_stopwordsAndTokenize(text)
    tokens = lemmatize(tokens)
    return tokens


def loadTweets():
    with open("collectedTweets.txt", 'r', encoding="utf-8") as jsonfile:
        data = [json.loads(l) for l in jsonfile.readlines() if len(l) > 5]
    return data


def storeTextTweet():
    data = loadTweets()

    tweetTextFile = open('tweetsText.txt', 'a')
    text_data = []
    for tweet in data:
        if(tweet["truncated"]):
            text = tweet["extended_tweet"]["full_text"]
        else:
            text = tweet["text"]
        tweetTextFile.write(text)
        tweetTextFile.write("\n")
    
    tweetTextFile.close()


def getText_data():
    data = loadTweets()
    text_data = []
    for tweet in data:
        if(tweet["truncated"]):
            text = tweet["extended_tweet"]["full_text"]
        else:
            text = tweet["text"]
        token = prepare_data_for_LDA(text)
        text_data.append(token)
    return text_data
    #print(text_data)



if __name__ == '__main__':
    
    downloadTweets()
    
    """
    text_data = getText_data()

    
    NUM_TOPICS = 5
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
    ldamodel.save('model5.gensim')
    topics = ldamodel.print_topics(num_words=4)
    for topic in topics:
        print(topic)
    
    
    pyLDAvis.enable_notebook()
    
    # feed the LDA model into the pyLDAvis instance
    lda_viz = gensimvis.prepare(ldamodel, corpus, dictionary)
    lda_viz
    
    
    # Compute Model Perplexity and Coherence Score
    
    
    from gensim.models import CoherenceModel
    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=ldamodel, texts=text_data, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score for 5 (=K) topics: ', coherence_lda)
        
    """



