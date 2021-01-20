#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


# In[4]:


np.random.seed(42)
tf.random.set_seed(42)


# In[10]:


from tensorflow.keras.preprocessing import image


# In[16]:


from  tensorflow.keras.applications.resnet50 import ResNet50 as myModel


# In[17]:


from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions 


# In[18]:


model = myModel(weights="imagenet")


# In[25]:


img_path = '/cxldata/dlcourse/lion.jpg'


# In[26]:


img = image.load_img(img_path, target_size=(224, 224))


# In[27]:


plt.imshow(np.asarray(img))


# In[30]:


x = image.img_to_array(img)
x= np.array([x])


# In[31]:


x =preprocess_input(x)


# In[32]:


preds = model.predict(x)


# In[33]:


print('Predicted:', decode_predictions(preds, top=3)[0])


# In[ ]:




