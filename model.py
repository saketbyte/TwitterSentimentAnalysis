#!/usr/bin/env python
# coding: utf-8

# # Twitter Sentiment Analysis
# 
# This project deals with classification of tweet-texts into three classes as - positive, neutral or negative sentiment.I have used dataset from [here](https://www.kaggle.com/arkhoshghalb/twitter-sentiment-analysis-hatred-speech). The dataset has 31900 rows of data with some good class imbalance of about 86% data being either positive or neutral and 14% only being a negative label.
# 
# I have tried various techniques to analyse these texts, work on balancing the dataset and then build a tree based model for predicting the sentiment given text based inputs.

# #### Importing needed libraries and loading data

import pandas as pd
import seaborn as sns
import numpy as np
from textblob import TextBlob
import nltk
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import stopwords

from nltk import PorterStemmer


from sklearn.model_selection import train_test_split

from imblearn.under_sampling import TomekLinks
from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pickle


tweet_df=pd.read_csv('twitter.csv')
tweet_df


tweet_df.drop(["id","label"], axis = 1, inplace = True)
tweet_df


# #### Data Cleaning pipeline
# 
# 1. Removal of symbols, emoticons, flags, lots of @user tags etc.
# 2. Removal of non ASCII characters
# 3. Removal of stop words which provide no essential meaning to our model and our redundantly present.


def cleans(txt):
    txt = re.sub(r'A[A-Za-z0-9_]+','',txt)
    txt = re.sub(r'#','',txt)
    txt = re.sub(r'@user','',txt)
    txt = re.sub(r'RT :','',txt)
    txt = re.sub(r'https?:\/\/A[A-Za-z0-9_]+','',txt)
    txt = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    
    return txt




tweet_df['tweet'].apply(cleans)
tweet_df





def remove_non_ascii(txt): 
    return ''.join(i for i in txt if ord(i)<128)

def removesmall(text):
    return ' '.join(word for word in text.split() if len(word)>3)

tweet_df['tweet'] = tweet_df['tweet'].apply(remove_non_ascii)
tweet_df['tweet'] = tweet_df['tweet'].apply(removesmall)
tweet_df



tweet_df['tweet'] = tweet_df['tweet'].str.replace('@user','')
tweet_df




stopwords = stopwords.words('english')




tweet_df['tweet'] = tweet_df['tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))




tweet_df['length']=tweet_df['tweet'].apply(len)
tweet_df


# For labelling purposes we make use of sentiment polarity and subjectivity to label as positive, neutral or negative as the default labeling is very ineffective and only binary classification is provided with high imbalance. I tried to go for multi class classification as :
# 
# 
# 1: positive
# 0: neutral
# -1: negative




def get_Subjectivity(txt):
    return TextBlob(txt).sentiment.subjectivity

def get_Polarity(txt):
    return TextBlob(txt).sentiment.polarity





tweet_df['Subjectivity'] = tweet_df['tweet'].apply(get_Subjectivity)
tweet_df['Polarity'] = tweet_df['tweet'].apply(get_Polarity)
tweet_df.head()




def applylabel(plr):
    if plr < -0.08:
        return -1
    elif plr>0.08:
        return 1
    else:
        return 0





tweet_df["label"] = tweet_df["Polarity"].apply(applylabel)
tweet_df


# Here we can see the class imbalance present. To deal with this I used majorly two techniques-
# 
# 1. Using class weights for XG Boost tree based model, in order to penalise the errors caused in minority class predictions.
# 
# 2. Using undersampling, by reducing Tomek Links in the data. 
# 
# 
# 
# PS: I would have used oversampling and undersampling combination however neither does JupyterNotebook nor my PC at the moment allowed processing of a larger dataset, even after I had tried freeing memory by deleting the variables I no longer needed.



sns.countplot(tweet_df.label)




tweet_df.label.value_counts()


# ##### Basic Natural Language Text Processing techniques to make the data even more suitable for models:
# 
# I applied tokenization, stemming, created a wordcloud to visualise most common words and at the end finally did count vectorization which creates a position based one hot encoded array for the words we have in our vocabulary, with atleast a minimum filter frequency of 2.



#Appliyng tokenization
def tokenization(text):
    text = re.split('\W+', text)
    return text

tweet_df['tokenized'] = tweet_df['tweet'].apply(lambda x: tokenization(x.lower()))




from nltk import PorterStemmer

ps = nltk.PorterStemmer()

def stemming(text):
    text = [ps.stem(word) for word in text]
    return text



tweet_df['stems'] = tweet_df['tokenized'].apply(lambda x: stemming(x))



tweet_df



sentences = tweet_df['tweet']
sentences_string = " ".join(sentences)


get_ipython().system('pip install WordCloud')
from wordcloud import WordCloud

plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(sentences_string))





tweet_df.drop(['Subjectivity','Polarity','tokenized','length'],axis = 1, inplace =True)




from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=2)
tweets_countvectorizer = vectorizer.fit_transform(tweet_df['tweet'])




x = pd.DataFrame(tweets_countvectorizer.toarray())




y = tweet_df["label"]
print(x.shape , y.shape)



del tweets_countvectorizer


# In[39]:


from imblearn.under_sampling import TomekLinks

tl = TomekLinks(sampling_strategy = "not minority")
X_tl, y_tl = tl.fit_resample(x, y)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_tl, y_tl, test_size=0.30)





del x, y, X_tl,y_tl




! pip install xgboost

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

XGBT = XGBClassifier(scale_pos_weight = 7.5)
XGBT.fit(X_train, y_train)

#rf = RandomForestClassifier(max_depth= 16 , random_state=0, class_weight="balanced",n_estimators=200)
#rf.fit(X_train, y_train)




# Predicting the Test set results
y_pred = XGBT.predict(X_test)



from sklearn.metrics import classification_report, confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)


# # Classification report for XGBoost with weighted classes



print(classification_report(y_test, y_pred))


# # Classification report for Naive Bayes as used earlier 




from sklearn.naive_bayes import MultinomialNB  
NBmodel = MultinomialNB()
NBmodel.fit(X_train, y_train)
y_NB =NBmodel.predict(X_test)

print(classification_report(y_test, y_NB))





# Saving model to disk
pickle.dump(XGBT, open('XGBTmodel.pkl','wb'))
pickle.dump(vectorizer,open('vectorizer.pkl','wb'))

# Loading model to compare the results







