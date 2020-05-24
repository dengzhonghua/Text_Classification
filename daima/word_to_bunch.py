#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import pickle
from sklearn.datasets.base import Bunch


# In[8]:


#读取文件
def read_file(file_path):
    with open(file_path,'r',encoding='utf-8',errors='ignore') as fp:
        contents=fp.readlines()
        return str(contents)


# In[13]:


#将文件转换成bunch数据结构
def word_to_bunch(train_save_path,train_bunch_path):
    bunch=Bunch(label=[],filepath=[],contents=[])
    labels=os.listdir(train_save_path)
    
    for label in labels:
        file_path=train_save_path + label + '/'
        detail_paths=os.listdir(file_path)
        
        for detail_path in detail_paths:
            full_path=file_path + detail_path
            contents=read_file(full_path)
            
            bunch.label.append(label)
            bunch.filepath.append(full_path)
            bunch.contents.append(contents)
            
    with open(train_bunch_path,'wb+') as fp:
        pickle.dump(bunch,fp)
    print('创建完成')


# In[14]:


if __name__=='__main__':
    train_save_path='./train_segments/'
    train_bunch_path='train_bunch_bag.dat'
    word_to_bunch(train_save_path,train_bunch_path)
    
    test_save_path='./test_segments/'
    test_bunch_path='test_bunch_bag.dat'
    word_to_bunch(test_save_path,test_bunch_path)


# In[ ]:




