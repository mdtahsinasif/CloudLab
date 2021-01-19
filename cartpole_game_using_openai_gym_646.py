#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
assert sys.version_info >= (3, 5)
import numpy as np


# In[2]:


import tensorflow as tf
assert tf.__version__ >= "2.0"


# In[3]:


from tensorflow import keras
import sklearn
assert sklearn.__version__ >= "0.20"


# In[4]:


np.random.seed(42)
tf.random.set_seed(42)


# In[5]:


import matplotlib.pyplot as plt
import matplotlib as mpl


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


# In[14]:


import gym


# In[15]:


gym.envs.registry.all()


# In[18]:


env = gym.make('CartPole-v1')


# In[19]:


env.seed(42)


# In[20]:


obs = env.reset()


# In[21]:


print(obs)


# In[22]:


print(env.action_space)


# In[23]:


help(env)


# In[26]:


action=1


# In[27]:


obs, reward, done, info = env.step(action)


# In[28]:


print(obs)


# In[29]:


print(reward)
print(done)
print(info)


# In[32]:


env.seed(42)

def basic_policy(obs):
    angle = obs[2]
    if angle < 0:
        return 0
    else:
        return 1


# In[33]:


totals = []
for episode in range(500):
    episode_rewards = 0
    obs = env.reset()
    for step in range(200):
        action = basic_policy(obs)
        obs, reward, done, info = env.step(action)
        episode_rewards += reward
        if done:
            break
    totals.append(episode_rewards)


# In[34]:


print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals))


# In[43]:


keras.backend.clear_session()


# In[44]:


tf.random.set_seed(42)
np.random.seed(42)


# In[45]:


n_inputs = 4


# In[46]:


model = keras.models.Sequential([
    keras.layers.Dense(5, activation="elu", input_shape=[n_inputs]),
    keras.layers.Dense(1, activation="sigmoid"),
])


# In[47]:


model.summary()


# In[50]:


env.seed(42)
def basic_policy_untrained(obs):
    left_proba = model.predict(obs.reshape(1, -1))
    action = int(np.random.rand() > left_proba)
    return action


# In[51]:


totals = []
for episode in range(50):
    episode_rewards = 0
    obs = env.reset()
    for step in range(200):
        action = basic_policy_untrained(obs)
        obs, reward, done, info = env.step(action)
        episode_rewards += reward
        if done:
            break
    totals.append(episode_rewards)

np.mean(totals), np.std(totals), np.min(totals), np.max(totals)


# In[54]:


np.random.seed(42)


# In[55]:


n_environments = 50


# In[56]:


n_iterations = 5000


# In[57]:


envs = [gym.make("CartPole-v1") for _ in range(n_environments)]


# In[58]:


for index, env in enumerate(envs):
    env.seed(index)


# In[59]:


observations = [env.reset() for env in envs]


# In[60]:


optimizer = keras.optimizers.RMSprop()
loss_fn =  keras.losses.binary_crossentropy


# In[61]:


for iteration in range(n_iterations):
    # if angle < 0, we want proba(left) = 1., or else proba(left) = 0.
    target_probas = np.array([([1.] if obs[2] < 0 else [0.])
                              for obs in observations])

    with tf.GradientTape() as tape:
        left_probas = model(np.array(observations))
        loss = tf.reduce_mean(loss_fn(target_probas, left_probas))
    print("\rIteration: {}, Loss: {:.3f}".format(iteration, loss.numpy()), end="")
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    actions = (np.random.rand(n_environments, 1) > left_probas.numpy()).astype(np.int32)
    for env_index, env in enumerate(envs):
        obs, reward, done, info = env.step(actions[env_index][0])
        observations[env_index] = obs if not done else env.reset()


# In[66]:


def play_one_step(env, obs, model, loss_fn):
    with tf.GradientTape() as tape:
        left_proba = model(obs[np.newaxis])
        action = (tf.random.uniform([1, 1]) > left_proba)
        y_target = tf.constant([[1.]]) - tf.cast(action, tf.float32)
        loss = tf.reduce_mean(loss_fn(y_target, left_proba))
    grads = tape.gradient(loss, model.trainable_variables)
    obs, reward, done, info = env.step(int(action[0, 0].numpy()))
    return obs, reward, done, grads


# In[71]:


def play_multiple_episodes(env, n_episodes, n_max_steps, model, loss_fn):
    all_rewards = []
    all_grads = []
    for episode in range(n_episodes):
        current_rewards = []
        current_grads = []
        obs = env.reset()
        for step in range(n_max_steps):
            obs, reward, done, grads = play_one_step(env, obs, model, loss_fn)
            current_rewards.append(reward)
            current_grads.append(grads)
            if done:
                break
        all_rewards.append(current_rewards)
        all_grads.append(current_grads)
    return all_rewards, all_grads


# In[74]:


def discount_rewards(rewards, discount_rate):
    discounted = np.array(rewards)
    for step in range(len(rewards) - 2, -1, -1):
        discounted[step] += discounted[step + 1] * discount_rate
    return discounted


# In[75]:


def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate)
                            for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean) / reward_std
            for discounted_rewards in all_discounted_rewards]


# In[82]:


keras.backend.clear_session()


# In[83]:


tf.random.set_seed(42)
np.random.seed(42)


# In[84]:


n_episodes_per_update = 10


# In[85]:


n_iterations = 150


# In[86]:


n_max_steps = 200


# In[87]:


discount_rate = 0.95


# In[88]:


n_inputs = 4


# In[89]:


optimizer = keras.optimizers.Adam(lr=0.01)


# In[90]:


loss_fn = keras.losses.binary_crossentropy


# In[91]:


def nn_policy_gradient(model, n_iterations, n_episodes_per_update, n_max_steps, loss_fn):
    env = gym.make("CartPole-v1")
    env.seed(42);

    for iteration in range(n_iterations):
        all_rewards, all_grads = play_multiple_episodes(
            env, n_episodes_per_update, n_max_steps, model, loss_fn)
        total_rewards = sum(map(sum, all_rewards))                     # Not shown in the book
        print("\rIteration: {}, mean rewards: {:.1f}".format(          # Not shown
            iteration, total_rewards / n_episodes_per_update), end="") # Not shown
        all_final_rewards = discount_and_normalize_rewards(all_rewards,
                                                        discount_rate)
        all_mean_grads = []
        for var_index in range(len(model.trainable_variables)):
            mean_grads = tf.reduce_mean(
                [final_reward * all_grads[episode_index][step][var_index]
                for episode_index, final_rewards in enumerate(all_final_rewards)
                    for step, final_reward in enumerate(final_rewards)], axis=0)
            all_mean_grads.append(mean_grads)
        optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))

    return model

    env.close()


# In[92]:


model = keras.models.Sequential([
    keras.layers.Dense(5, activation="elu", input_shape=[n_inputs]),
    keras.layers.Dense(1, activation="sigmoid"),
])


# In[ ]:


model = nn_policy_gradient(model, n_iterations, n_episodes_per_update, n_max_steps, loss_fn)


# In[ ]:


totals = []
for episode in range(20):
    print("Episode:",episode)
    episode_rewards = 0
    obs = env.reset()
    for step in range(200):
        action = basic_policy_untrained(obs)
        obs, reward, done, info = env.step(action)
        episode_rewards += reward
        if done:
            break
    totals.append(episode_rewards)

np.mean(totals), np.std(totals), np.min(totals), np.max(totals)


# In[ ]:




