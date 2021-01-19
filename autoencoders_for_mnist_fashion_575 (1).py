#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
import sklearn
import matplotlib.pyplot as plt
import matplotlib as mpl


# In[3]:


np.random.seed(42)
tf.random.set_seed(42)


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


# In[13]:


(X_train_full, y_train_full), (X_test, y_test) =keras.datasets.fashion_mnist.load_data()


# In[14]:


X_train_full = X_train_full.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255


# In[15]:


X_train, X_valid = X_train_full[:-5000],X_train_full[-5000:]


# In[16]:


y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]


# In[23]:


def rounded_accuracy(y_true, y_pred):
    return keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))


# In[24]:


def plot_image(image):
    plt.imshow(image, cmap="binary")
    plt.axis("off")


# In[25]:


def show_reconstructions(model, images=X_valid, n_images=5):
    reconstructions = model.predict(images[:n_images])
    fig = plt.figure(figsize=(n_images * 1.5, 3))
    for image_index in range(n_images):
        plt.subplot(2, n_images, 1 + image_index)
        plot_image(images[image_index])
        plt.subplot(2, n_images, 1 + n_images + image_index)
        plot_image(reconstructions[image_index])


# In[30]:


stacked_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(30, activation="selu"),
])


# In[31]:


stacked_decoder = keras.models.Sequential([
    keras.layers.Dense(100, activation="selu", input_shape=[30]),
    keras.layers.Dense(28 * 28, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])


# In[32]:


stacked_ae = keras.models.Sequential([stacked_encoder, stacked_decoder])


# In[33]:


stacked_ae.compile(loss="binary_crossentropy",
           optimizer=keras.optimizers.SGD(lr=1.5), metrics=[rounded_accuracy])


# In[38]:


import time


# In[39]:


start = time.time()


# In[40]:


history = stacked_ae.fit(X_train, X_train, epochs=20,
                 validation_data=(X_valid, X_valid))


# In[41]:


end = time.time()


# In[42]:


print("Time of execution:", round(end-start,2),"seconds")


# In[45]:


show_reconstructions(stacked_ae, X_test)


# In[46]:


stacked_ae.evaluate(X_test, X_test)


# In[47]:


np.random.seed(42)

from sklearn.manifold import TSNE

start = time.time()

X_valid_compressed = stacked_encoder.predict(X_valid)
tsne = TSNE()
X_valid_2D = tsne.fit_transform(X_valid_compressed)
X_valid_2D = (X_valid_2D - X_valid_2D.min()) / (X_valid_2D.max() - X_valid_2D.min())

end = time.time()

print("Time of execution:", round(end-start,2),"seconds")


# In[48]:


plt.figure(figsize=(10, 8))
cmap = plt.cm.tab10
plt.scatter(X_valid_2D[:, 0], X_valid_2D[:, 1], c=y_valid, s=10, cmap=cmap)
image_positions = np.array([[1., 1.]])
for index, position in enumerate(X_valid_2D):
    dist = np.sum((position - image_positions) ** 2, axis=1)
    if np.min(dist) > 0.02: # if far enough from other images
        image_positions = np.r_[image_positions, [position]]
        imagebox = mpl.offsetbox.AnnotationBbox(
            mpl.offsetbox.OffsetImage(X_valid[index], cmap="binary"),
            position, bboxprops={"edgecolor": cmap(y_valid[index]), "lw": 2})
        plt.gca().add_artist(imagebox)
plt.axis("off")
plt.show()


# In[ ]:




