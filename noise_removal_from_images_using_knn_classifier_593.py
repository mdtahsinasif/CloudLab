#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import gzip


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib


# In[5]:


import matplotlib.pyplot as plt


# In[11]:


def showImage(data):
    some_article = data   # Selecting the image.
    some_article_image = some_article.reshape(28, 28)
    plt.imshow(some_article_image, cmap = matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()


# In[15]:


filePath_train_set = '/cxldata/datasets/project/mnist/train-images-idx3-ubyte.gz'
filePath_train_label = '/cxldata/datasets/project/mnist/train-labels-idx1-ubyte.gz'
filePath_test_set = '/cxldata/datasets/project/mnist/t10k-images-idx3-ubyte.gz'
filePath_test_label = '/cxldata/datasets/project/mnist/t10k-labels-idx1-ubyte.gz'


# In[18]:


with gzip.open(filePath_train_label, 'rb') as trainLbpath:
     trainLabel = np.frombuffer(trainLbpath.read(), dtype=np.uint8,
                               offset=8)
with gzip.open(filePath_train_set, 'rb') as trainSetpath:
     trainSet = np.frombuffer(trainSetpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(trainLabel), 784)

with gzip.open(filePath_test_label, 'rb') as testLbpath:
     testLabel = np.frombuffer(testLbpath.read(), dtype=np.uint8,
                               offset=8)

with gzip.open(filePath_test_set, 'rb') as testSetpath:
     testSet = np.frombuffer(testSetpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(testLabel), 784)


# In[19]:


X_train,X_test, y_train, y_test = trainSet, testSet, trainLabel,testLabel


# In[25]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[26]:


showImage(X_train[0])
X_train[0]


# In[27]:


plt.figure(figsize=(10,10))
for i in range(15):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    array_image = X_train[i].reshape(28, 28)
    plt.imshow(array_image, cmap = matplotlib.cm.binary, interpolation="nearest")
plt.show()


# In[32]:


np.random.seed(42)


# In[33]:


shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index],y_train[shuffle_index]


# In[37]:


import numpy.random as rnd


# In[38]:


noise_train = rnd.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise_train
noise_test = rnd.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise_test
y_train_mod = X_train
y_test_mod = X_test


# In[42]:


showImage(X_test_mod[4000])

showImage(y_test_mod[4000])


# In[45]:


from sklearn.neighbors import KNeighborsClassifier


# In[46]:


knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_mod, y_train_mod)


# In[50]:


clean_digit = knn_clf.predict([X_test_mod[5000]])


# In[51]:


showImage(clean_digit)
showImage(y_test_mod[5000])


# In[ ]:




