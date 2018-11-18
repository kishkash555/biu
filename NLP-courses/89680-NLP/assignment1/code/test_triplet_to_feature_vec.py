#!/usr/bin/env python
# coding: utf-8

# In[21]:


from importlib import reload

import GreedyMaxEntTag as gmet
reload(gmet)
generate_greedily_tagged_triplets = gmet.generate_greedily_tagged_triplets 

import extractFeatures as ef
import pickle


# In[30]:


sentence_str = 'The ancient financial executive'
model = pickle.load(open('../data/memm_model.pickle','rb'))


# In[22]:


tag_dict, feature_dict = gmet.load_map_file('../data/map_file')
registered_words, registered_tags, registered_tag_pairs = gmet.get_registered_features(feature_dict.keys())
registered_features = ef.registered_features +        [ef.is_word(w) for w in registered_words] +         [ef.prev_tag(t) for t in registered_tags] +         [ef.previous_2_tags(*tp) for tp in registered_tag_pairs]
   


# In[23]:


len(registered_words), len(registered_features), len(registered_tags), len(registered_tag_pairs)


# In[31]:


tt = iter(generate_greedily_tagged_triplets(sentence_str, model, registered_features, feature_dict, tag_dict))


# In[32]:


next(tt)


# In[ ]:




