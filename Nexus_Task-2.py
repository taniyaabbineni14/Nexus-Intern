#!/usr/bin/env python
# coding: utf-8

# In[66]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # vizulization
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import re


# In[67]:


df=pd.read_csv("tweets.csv",encoding = 'latin',header=None)


# In[68]:


df


# In[69]:


df.shape


# In[70]:


df=df.rename(columns={0: 'sentiment',1:"id",2:"Date",3:"flag",4:"user",5:"text"})


# In[71]:


df.sample(5)


# In[72]:


df = df.drop(['id', 'Date', 'flag', 'user'], axis=1)


# In[73]:


lab_to_sentiment = {0:"Negative", 4:"Positive"}
def label_decoder(label):
  return lab_to_sentiment[label]
df.sentiment = df.sentiment.apply(lambda x: label_decoder(x))


# In[74]:


df["text"][0]


# In[75]:


df.info()


# In[76]:


df.duplicated().sum()


# In[77]:


df = df.sample(50000)


# In[78]:


df["text"]=df["text"].str.lower()


# In[79]:


df


# In[80]:


# Removed whitespace
df["text"]=df["text"].str.strip()


# In[81]:


# remove html 
df['text'] = df['text'].str.replace(r'<.*?>','')


# In[82]:


df


# In[83]:


df[df['text'].str.contains(r"https?://\S+|www\.\S+",'')].iloc[3].values


# In[84]:


# Code to remove url
df['text'] = df['text'].str.replace(r"https?://\S+|www\.\S+",'')


# In[85]:


df


# In[86]:


# expanding abbvr

# expand 

def remove_abb(data):
    data = re.sub(r"he's", "he is", data)
    data = re.sub(r"there's", "there is", data)
    data = re.sub(r"We're", "We are", data)
    data = re.sub(r"That's", "That is", data)
    data = re.sub(r"won't", "will not", data)
    data = re.sub(r"they're", "they are", data)
    data = re.sub(r"Can't", "Cannot", data)
    data = re.sub(r"wasn't", "was not", data)
    data = re.sub(r"don\x89Ûªt", "do not", data)
    data= re.sub(r"aren't", "are not", data)
    data = re.sub(r"isn't", "is not", data)
    data = re.sub(r"What's", "What is", data)
    data = re.sub(r"haven't", "have not", data)
    data = re.sub(r"hasn't", "has not", data)
    data = re.sub(r"There's", "There is", data)
    data = re.sub(r"He's", "He is", data)
    data = re.sub(r"It's", "It is", data)
    data = re.sub(r"You're", "You are", data)
    data = re.sub(r"I'M", "I am", data)
    data = re.sub(r"shouldn't", "should not", data)
    data = re.sub(r"wouldn't", "would not", data)
    data = re.sub(r"i'm", "I am", data)
    data = re.sub(r"I\x89Ûªm", "I am", data)
    data = re.sub(r"I'm", "I am", data)
    data = re.sub(r"Isn't", "is not", data)
    data = re.sub(r"Here's", "Here is", data)
    data = re.sub(r"you've", "you have", data)
    data = re.sub(r"you\x89Ûªve", "you have", data)
    data = re.sub(r"we're", "we are", data)
    data = re.sub(r"what's", "what is", data)
    data = re.sub(r"couldn't", "could not", data)
    data = re.sub(r"we've", "we have", data)
    data = re.sub(r"it\x89Ûªs", "it is", data)
    data = re.sub(r"doesn\x89Ûªt", "does not", data)
    data = re.sub(r"It\x89Ûªs", "It is", data)
    data = re.sub(r"Here\x89Ûªs", "Here is", data)
    data = re.sub(r"who's", "who is", data)
    data = re.sub(r"I\x89Ûªve", "I have", data)
    data = re.sub(r"y'all", "you all", data)
    data = re.sub(r"can\x89Ûªt", "cannot", data)
    data = re.sub(r"would've", "would have", data)
    data = re.sub(r"it'll", "it will", data)
    data = re.sub(r"we'll", "we will", data)
    data = re.sub(r"wouldn\x89Ûªt", "would not", data)
    data = re.sub(r"We've", "We have", data)
    data = re.sub(r"he'll", "he will", data)
    data = re.sub(r"Y'all", "You all", data)
    data = re.sub(r"Weren't", "Were not", data)
    data = re.sub(r"Didn't", "Did not", data)
    data = re.sub(r"they'll", "they will", data)
    data = re.sub(r"they'd", "they would", data)
    data = re.sub(r"DON'T", "DO NOT", data)
    data = re.sub(r"That\x89Ûªs", "That is", data)
    data = re.sub(r"they've", "they have", data)
    data = re.sub(r"i'd", "I would", data)
    data = re.sub(r"should've", "should have", data)
    data = re.sub(r"You\x89Ûªre", "You are", data)
    data = re.sub(r"where's", "where is", data)
    data = re.sub(r"Don\x89Ûªt", "Do not", data)
    data = re.sub(r"we'd", "we would", data)
    data = re.sub(r"i'll", "I will", data)
    data = re.sub(r"weren't", "were not", data)
    data = re.sub(r"They're", "They are", data)
    data = re.sub(r"Can\x89Ûªt", "Cannot", data)
    data = re.sub(r"you\x89Ûªll", "you will", data)
    data = re.sub(r"I\x89Ûªd", "I would", data)
    data = re.sub(r"let's", "let us", data)
    data = re.sub(r"it's", "it is", data)
    data = re.sub(r"can't", "cannot", data)
    data = re.sub(r"don't", "do not", data)
    data = re.sub(r"you're", "you are", data)
    data = re.sub(r"i've", "I have", data)
    data = re.sub(r"that's", "that is", data)
    data = re.sub(r"i'll", "I will", data)
    data = re.sub(r"doesn't", "does not",data)
    data = re.sub(r"i'd", "I would", data)
    data = re.sub(r"didn't", "did not", data)
    data = re.sub(r"ain't", "am not", data)
    data = re.sub(r"you'll", "you will", data)
    data = re.sub(r"I've", "I have", data)
    data = re.sub(r"Don't", "do not", data)
    data = re.sub(r"I'll", "I will", data)
    data = re.sub(r"I'd", "I would", data)
    data = re.sub(r"Let's", "Let us", data)
    data = re.sub(r"you'd", "You would", data)
    data = re.sub(r"It's", "It is", data)
    data = re.sub(r"Ain't", "am not", data)
    data = re.sub(r"Haven't", "Have not", data)
    data = re.sub(r"Could've", "Could have", data)
    data = re.sub(r"youve", "you have", data)  
    data = re.sub(r"donå«t", "do not", data)
    
    return data


# In[87]:


df['text'] = df['text'].apply(remove_abb)


# In[88]:


df["text"]


# In[89]:


pip install textblob


# In[92]:


#Punctuation
import string


# In[93]:


string.punctuation


# In[94]:


def remove_puctuation(text):
    
    for i in string.punctuation:
        if i in text:
            text = text.replace(i,'')
            
    return text


# In[95]:


df['text'] = df['text'].apply(remove_puctuation)
df['text'].head()


# In[96]:


pip install tokenizer


# In[97]:


pip install nltk


# In[98]:


import nltk

# Download the Punkt tokenizer
nltk.download('punkt')


# In[99]:


from nltk.tokenize import word_tokenize


# In[100]:


df['tokenized_text'] = df['text'].apply(word_tokenize)


# In[101]:


df.head()


# In[102]:


import nltk

# Download the stopwords resource
nltk.download('stopwords')


# In[103]:


from nltk.corpus import stopwords
import nltk

# Download the stopwords data if not already downloaded
nltk.download('stopwords')

stopwords.words("english")


# In[104]:


def remove_stopwords(text):
    
    L = []
    for word in text:
        if word not in stopwords.words('english'):
            L.append(word)
            
    return L


# In[105]:


df['tokenized_text'] = df['tokenized_text'].apply(remove_stopwords)


# In[106]:


df.head()


# In[107]:


df['text'] = df['tokenized_text'].apply(lambda x:" ".join(x))


# In[108]:


df.head()


# In[109]:


df['char_length'] = df['text'].str.len()


# In[110]:


df.head()


# In[111]:


df['word_length'] = df['tokenized_text'].apply(len)


# In[112]:


df.head()


# In[113]:


import seaborn as sns


# In[114]:


sns.displot(df["word_length"])


# In[115]:


sns.distplot(df[df['sentiment'] == 'Positive']['word_length'])
sns.distplot(df[df['sentiment'] == 'Negative']['word_length'])
sns.plot


# In[116]:


sns.distplot(df[df['sentiment'] == 'Positive']['char_length'])
sns.distplot(df[df['sentiment'] == 'Negative']['char_length'])


# In[122]:


df['tokenized_text'].sum()


# In[123]:


from nltk import ngrams

pd.Series(ngrams(df['tokenized_text'].sum(),2)).value_counts()


# In[124]:


pd.Series(ngrams(df['tokenized_text'].sum(),3)).value_counts()


# In[117]:


pip install wordcloud


# In[118]:


import matplotlib.pyplot as plt


# In[119]:


from wordcloud import WordCloud


# In[120]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

plt.figure(figsize = (20,20)) # Positive Review Text
wc = WordCloud(width = 1000 , height = 400).generate(" ".join(df[df['sentiment'] == 'Negative']['text']))
plt.imshow(wc)


# In[121]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

plt.figure(figsize = (20,20)) # Positive Review Text
wc = WordCloud(width = 1000 , height = 400).generate(" ".join(df[df['sentiment'] == 'Positive']['text']))
plt.imshow(wc)


# In[141]:


# Count the occurrences of each sentiment
sentiment_counts = df['sentiment'].value_counts()

# Create a bar chart
plt.bar(sentiment_counts.index, sentiment_counts.values, color=['skyblue', 'salmon'])
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Sentiment Distribution')
plt.show()

