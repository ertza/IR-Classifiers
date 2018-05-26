
# coding: utf-8

# ### Library imports for the code

# In[1]:


import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import warnings
import os, re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.metrics import precision_score, recall_score, classification_report
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from datetime import datetime as dt
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# ### Classifier Tester

# Will test and show important metrics for any classifier passed to it

# In[2]:


def testClassifier(x_train, y_train, x_test, y_test, clf):
    metrics = []
    start = dt.now()
    clf.fit(x_train, y_train)
    end = dt.now()
    print ('training time: ', (end - start))
    metrics.append(end-start)
    start = dt.now()
    yhat = clf.predict(x_test)
    end = dt.now()
    print ('testing time: ', (end - start))
    metrics.append(end-start)
    print ('classification report: ')
    print(classification_report(y_test, yhat))
    print ('f1 score')
    print (f1_score(y_test, yhat, average='macro'))
    print ('accuracy score')
    print (accuracy_score(y_test, yhat))
    precision = precision_score(y_test, yhat, average=None)
    recall = recall_score(y_test, yhat, average=None)
    for p, r in zip(precision, recall):
        metrics.append(p)
        metrics.append(r)
    metrics.append(f1_score(y_test, yhat, average='macro'))
    print ('confusion matrix:')
    print (confusion_matrix(y_test, yhat))
    plt.imshow(confusion_matrix(y_test, yhat), interpolation='nearest')
    plt.show()
    return metrics
metrics_dict = []


# ## DBWorld Emails Data

# The data for this database is already pre-processed and in bag of words format. We just read and split the data into train test sets

# In[3]:


dbworld = 'Datasets/dbworld/MATLAB/dbworld_bodies_stemmed.mat'
db_world = sio.loadmat(dbworld)
db_world_inputs = db_world['inputs']
db_world_labels = db_world['labels'].reshape(len(db_world['labels']),)
X_train, X_test, y_train, y_test = train_test_split(db_world_inputs, 
                    db_world_labels, test_size=0.33, random_state=42)


# #### Naive Bayes

# We have chosen MultinomialNB as it gives the best results for Naive Bayes in case of text classification.

# In[4]:


mnb = MultinomialNB()
mnb_me = testClassifier(X_train, y_train, X_test, y_test, mnb)
metrics_dict.append({'name':'NaiveBayes', 'metrics':mnb_me})


# #### Rocchio Classification

# For this we will be using the NearestCentroid classifier as when it is used for text classification with tf-idf vectors, this classifier is also known as the Rocchio classifier.

# In[5]:


tfidf = TfidfTransformer()
tfidf.fit(X_train)
train_tf = tfidf.transform(X_train)
test_tf = tfidf.transform(X_test)
ncr = NearestCentroid()
ncr_me = testClassifier(train_tf, y_train, test_tf, y_test, ncr)
metrics_dict.append({'name':'Rocchio', 'metrics':ncr_me})


# #### kNN Classification

# We'll use kNearestNeighbor for classification now. We tried different values for k and 4 came out to be the best for this.

# In[6]:


knn = KNeighborsClassifier(n_neighbors = 4)
knn_me = testClassifier(train_tf, y_train, test_tf, y_test, knn)
metrics_dict.append({'name':'kNN', 'metrics':knn_me})


# #### Conclusion

# As we can see the kNN classifier with k=4 gives the highest accuracy and f1 score for this document. The training time is negligible but the testing time is the highest among all as it is a known trait of kNN.

# ## Health Tweets

# First of all we need to do pre-processing on the data as it is in raw text format. We split the tweets according to the delimeter '|' and clean-up the text.

# #### Pre-processing

# We divide the documents into different classes according to the news agency accounts. The documents are then converted to tf-idf vectors. Further they are split into train test sets.

# In[7]:


health_tweet = os.listdir('Datasets/Health-News-Tweets/Health-Tweets/')
X_data = []
y_data = []
for files in health_tweet:
    file = open('Datasets/Health-News-Tweets/Health-Tweets/'+files, encoding="utf8")
    data = file.readlines()
    for line in data:
        try:
            line = re.sub(r"http\S+", "", line.split('|')[2]).lower()
            X_data.append(line.strip())
            y_data.append(files.rstrip('.txt'))
        except: pass
    file.close()
vectorizer = CountVectorizer()
vectorizer.fit(X_data)
train_mat = vectorizer.transform(X_data)
tfidf = TfidfTransformer()
tfidf.fit(train_mat)
train_tfmat = tfidf.transform(train_mat)
X_train, X_test, y_train, y_test = train_test_split(train_tfmat, 
                    y_data, test_size=0.33, random_state=42)


# #### Naive Bayes

# We have chosen MultinomialNB as it gives the best results for Naive Bayes in case of text classification.

# In[8]:


mnb = MultinomialNB()
mnb_me = testClassifier(X_train, y_train, X_test, y_test, mnb)
metrics_dict.append({'name':'NaiveBayes', 'metrics':mnb_me})


# #### Rocchio Classification

# For this we will be using the NearestCentroid classifier as when it is used for text classification with tf-idf vectors, this classifier is also known as the Rocchio classifier.

# In[9]:


tfidf = TfidfTransformer()
tfidf.fit(X_train)
train_tf = tfidf.transform(X_train)
test_tf = tfidf.transform(X_test)
ncr = NearestCentroid()
ncr_me = testClassifier(train_tf, y_train, test_tf, y_test, ncr)
metrics_dict.append({'name':'Rocchio', 'metrics':ncr_me})


# #### kNN Classification

# We'll use kNearestNeighbor for classification now. We tried different values for k and 4 came out to be the best for this.

# In[10]:


knn = KNeighborsClassifier(n_neighbors = 5)
knn_me = testClassifier(train_tf, y_train, test_tf, y_test, knn)
metrics_dict.append({'name':'kNN', 'metrics':knn_me})


# #### Conclusion

# For this dataset the Rocchio outperformed the rest of the two classification algorithms and it also was the one that took the least amount of time for training as well testing of the data.

# ## Sentence Corpus

# First of all we need to do pre-processing on the data as it is in raw text format. We split the dataset according to the Argumentative Zones annotation scheme and clean-up the text.

# #### Pre-processing

# We divide the documents into different classes according to the Argumentative Zones annotation scheme. The documents are then converted to tf-idf vectors. Further they are split into train test sets.

# In[11]:


sentence_corpus = os.listdir('Datasets/SentenceCorpus/SentenceCorpus/labeled_articles')
X_data = []
y_data = []
for files in sentence_corpus:
    file = open ('Datasets/SentenceCorpus/SentenceCorpus/labeled_articles/'+files)
    data = file.readlines()
    for lines in data:
        if '###' not in lines:
            lines = lines.split("\t")
            try:
                X_data.append(lines[1].lower().replace(' citation',''))
                y_data.append(lines[0].lower().strip())
            except: pass
    file.close()
vectorizer = CountVectorizer()
vectorizer.fit(X_data)
train_mat = vectorizer.transform(X_data)
tfidf = TfidfTransformer()
tfidf.fit(train_mat)
train_tfmat = tfidf.transform(train_mat)
X_train, X_test, y_train, y_test = train_test_split(train_tfmat, 
                    y_data, test_size=0.33, random_state=42)


# #### Naive Bayes

# We have chosen MultinomialNB as it gives the best results for Naive Bayes in case of text classification.

# In[12]:


mnb = MultinomialNB()
mnb_me = testClassifier(X_train, y_train, X_test, y_test, mnb)
metrics_dict.append({'name':'NaiveBayes', 'metrics':mnb_me})


# #### Rocchio Classification

# For this we will be using the NearestCentroid classifier as when it is used for text classification with tf-idf vectors, this classifier is also known as the Rocchio classifier.

# In[13]:


tfidf = TfidfTransformer()
tfidf.fit(X_train)
train_tf = tfidf.transform(X_train)
test_tf = tfidf.transform(X_test)
ncr = NearestCentroid()
ncr_me = testClassifier(train_tf, y_train, test_tf, y_test, ncr)
metrics_dict.append({'name':'Rocchio', 'metrics':ncr_me})


# #### kNN Classification

# We'll use kNearestNeighbor for classification now. We tried different values for k and 4 came out to be the best for this.

# In[14]:


knn = KNeighborsClassifier(n_neighbors = 5)
knn_me = testClassifier(train_tf, y_train, test_tf, y_test, knn)
metrics_dict.append({'name':'kNN', 'metrics':knn_me})


# #### Conclusion

# For this dataset again the Rocchio outperformed the rest of the two classification algorithms and it also was the one that took the least amount of time for training as well testing of the data. The Naive Bayes performed the worst in this case
