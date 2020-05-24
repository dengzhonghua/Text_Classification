#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.datasets.base import Bunch


# In[2]:


#读取bunch对象
def read_bunch(path):
    with open(path,'rb') as fp:
        bunch=pickle.load(fp)
    return bunch


# In[3]:


#读取文件
def read_file(path):
    with open(path,'rb') as fp:
        bunch=fp.read()
    return bunch


# In[4]:


#保存bunch对象
def save_bunch(path,bunch):
    with open(path,'wb') as fp:
        pickle.dump(bunch,fp)


# In[10]:


#训练集
def train_tfidf_space(stopword_path,train_bunch_path,train_tfidf_data):
    bunch=read_bunch(train_bunch_path)                 #读取训练集bunch结构
    stopwords=read_file(stopword_path).splitlines()    #读取停用词
    tfidf_space=Bunch(label=bunch.label,filepath=bunch.filepath,contents=bunch.contents,tdm=[],space={})
    
    vectorizer=TfidfVectorizer(stop_words=stopwords,sublinear_tf=True,max_df=0.5)
    tfidf_space.tdm=vectorizer.fit_transform(bunch.contents)
    tfidf_space.space=vectorizer.vocabulary_           #训练集的词向量空间坐标
    
    save_bunch(train_tfidf_data,tfidf_space)


# In[11]:


#测试集
def test_tfidf_space(stopword_path,test_bunch_path,test_tfidf_data,train_tfidf_data):
    bunch=read_bunch(test_bunch_path)               #读取测试集bunch结构
    stopwords=read_file(stopword_path).splitlines() #读取停用词
    tfidf_space=Bunch(label=bunch.label,filepath=bunch.filepath,contents=bunch.contents,tdm=[],space={})
    
    train_bunch=read_bunch(train_tfidf_data)        #读取训练集tfidf数据
    tfidf_space.space=train_bunch.space             #将训练集的词向量空间坐标赋值给测试集
    
    vectorizer=TfidfVectorizer(stop_words=stopwords,sublinear_tf=True,max_df=0.5,vocabulary=train_bunch.space)
    
    tfidf_space.tdm=vectorizer.fit_transform(bunch.contents)
    save_bunch(test_tfidf_data,tfidf_space)


# In[12]:


if __name__=='__main__':
    stopword_path='./chinese_stop_words.txt'    #训练集数据处理
    
    train_bunch_path='./train_bunch_bag.dat'
    train_tfidf_data='./train_tfidfspace.dat'
    train_tfidf_space(stopword_path,train_bunch_path,train_tfidf_data)
    
    test_bunch_path='./test_bunch_bag.dat'      #测试集数据处理
    test_tfidf_data='./test_tfidfspace.dat'
    test_tfidf_space(stopword_path,test_bunch_path,test_tfidf_data,train_tfidf_data)


# In[ ]:




