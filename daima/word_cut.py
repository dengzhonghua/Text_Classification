#!/usr/bin/env python
# coding: utf-8

# In[49]:


import os
import jieba
import jieba.analyse


# In[50]:


#保存至文件
def save_file(file_path,content):
    with open(file_path,'a',encoding='utf-8',errors='ignore') as fp:
        fp.write(content)


# In[51]:


#读取文件
def read_file(file_path):
    with open(file_path,'r',encoding='utf-8',errors='ignore') as fp:
        content=fp.readlines()
        return str(content)


# In[52]:


#提取文章主题词
def extract_theme(content):
    themes=[]
    tags=jieba.analyse.extract_tags(content,topK=3,withWeight=True,allowPOS=['n','ns','v','vn'],withFlag=True)
    for i in tags:
        themes.append(i[0].word)
    return str(themes)


# In[70]:


#对文本进行分词
def cast_words(origin_path,save_path,theme_tag):
    file_lists=os.listdir(origin_path)   #原文档所在路径
    for dir_1 in file_lists:             #找到原文档中的文件夹
        file_path=origin_path + dir_1 + '/'
        seg_path=save_path + dir_1 + '/'
        
        if not os.path.exists(seg_path):
            os.mkdir(seg_path)
        
        detail_paths=os.listdir(file_path)
        for detail_path in detail_paths:
            full_path=file_path + detail_path
            file_content=read_file(full_path)
            file_content=file_content.strip()
            file_content=file_content.replace('\'',"")
            file_content=file_content.replace('\n',"")
            
            content_seg1=jieba.cut(file_content)    #为文件内容分词
            save_file(seg_path + detail_path," ".join(content_seg1))
            
            if theme_tag is not None:
                print('文件路径：{}'.format(theme_tag + detail_path))
                content_seg2=jieba.cut(file_content)
                theme=extract_theme(" ".join(content_seg2))    #theme为文章的主题关键词
                print('文章主题关键词：{}'.format(theme))
                save_file(theme_tag + detail_path,theme)    #将测试集文章主题关键字保存到指定路径     


# In[71]:


#对训练集进行分词
if __name__=='__main__':
    train_words_path='./train_words/'
    train_save_path='./train_segments/'
    cast_words(train_words_path,train_save_path,theme_tag=None)


# In[72]:


#对测试集进行分词，并抽取文章主题关键词
if __name__=='__main__':
    train_words_path='./test_words/'
    train_save_path='./test_segments/'
    theme_tag_path='./theme_tag1/'   #存放测试集文章主题关键词路径
    cast_words(train_words_path,train_save_path,theme_tag=theme_tag_path)

