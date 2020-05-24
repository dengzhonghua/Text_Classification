#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import warnings
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
warnings.filterwarnings("ignore")


# In[2]:


#读取bunch对象
def read_bunch(path):
    with open(path,"rb") as fp:
        bunch=pickle.load(fp)
    return bunch


# In[3]:


#保存文件对象
def save_file(path,content):
    with open(path,'a',encoding="utf-8",errors='ignore') as fp:
        fp.write(content)


# In[4]:


#朴素贝叶斯分类
def nbayes_classify(train_set,test_set):
    clf=MultinomialNB(alpha=0.5)
    clf.fit(train_set.tdm,train_set.label)
    predict=clf.predict(test_set.tdm)
    return predict


# In[5]:


#模型评价
def classification_result(actual,predict):
    print('精度:{0:.3f}'.format(metrics.precision_score(actual,predict,average='weighted')))
    print('召回:{0:.3f}'.format(metrics.recall_score(actual,predict,average='weighted')))
    print('f1-score:{0:.3f}'.format((metrics.f1_score(actual,predict,average='weighted'))))


# In[6]:


if __name__=='__main__':
    train_path='./train_tfidfspace.dat'    #导入训练集数据
    train_set=read_bunch(train_path)
    
    test_path='./test_tfidfspace.dat'      #导入测试集数据
    test_set=read_bunch(test_path)
    
    predict=nbayes_classify(train_set,test_set)
    classification_result(test_set.label,predict)
    print('-'*100)
    
    save_path='./classify_result_file.txt'  #保存结果路径
    for label,filename,predict in zip(test_set.label,test_set.filepath,predict):
        print(filename,'\t实际类别：',label,'\t-->预测类别：',predict)
        save_content=filename + '\t实际类别：' + label + '\t-->预测类别：' + predict + '\n'
        save_file(save_path,save_content)   #将分类结果保存


# In[ ]:




