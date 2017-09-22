#! -*- coding:utf-8 -*-
'''
@author: Kai Zheng
'''
import config
import numpy as np
import jieba
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences

def word_seg(training_file, seg_file):
    """
    possible exceptions:
    sentence too long
    no question mark
    no period
    file coding: Unicode --encode--> utf-8
    
    other functionality
    question index
    normalization
    co&idf at next level
    """
    raw_text = open(training_file, 'r')
    seg_text = open(seg_file, 'w')
    prev_q = ""
    index = 0
    for line in raw_text:
        q = line.split('\t')[0]
        a = line.split('\t')[1]
        score = line.split()[-1]
        
        if prev_q != q:
            index += 1 # new question set
            prev_q = q

        q_seg_list = list(jieba.cut(q))
        a_seg_list = list(jieba.cut(a))
        q_seg_utf8_list = []
        a_seg_utf8_list = []
        for s in q_seg_list:
            # q_seg_utf8_list.append(s.encode("utf-8"))
            q_seg_utf8_list.append(s)
        for s in a_seg_list:
            # a_seg_utf8_list.append(s.encode("utf-8"))
            a_seg_utf8_list.append(s)
        #print q_seg_utf8_list
        
        seg_text.write(str(index))
        seg_text.write('\t') # separator
        seg_text.write(' '.join(q_seg_utf8_list))
        seg_text.write('\t')
        seg_text.write(' '.join(a_seg_utf8_list))
        seg_text.write('\t')
        seg_text.write(str(score))
        seg_text.write('\n')
    raw_text.close()
    seg_text.close()
    
def word_to_syn(seg_file, syntrans_file):
    # Raw text dividing
    raw_text = open(seg_file, 'r')
    syntrans_text = open(syntrans_file, 'w')
    opp_dict = {} # Record known words' code
    for line in raw_text:
        index, q, a, score = line.split('\t') # index, q, a, match result
        score = score[0]
        # Translate sentence to synonym coding
        syn_sentence_q = [] # sentence after encoded to synonym coding  
        for word in q.split():
            if word in opp_dict:
                syn_sentence_q.append(opp_dict[word])
            else:
                opp_dict[word] = find_synonym(word)
                syn_sentence_q.append(opp_dict[word])
        
        syn_sentence_a = []
        for word in a.split():
            if word in opp_dict:
                syn_sentence_a.append(opp_dict[word])
            else:
                opp_dict[word] = find_synonym(word)
                syn_sentence_a.append(opp_dict[word])
                
        syntrans_text.write(str(index))
        syntrans_text.write('\t')
        syntrans_text.write(" ".join(syn_sentence_q))
        syntrans_text.write('\t')
        syntrans_text.write(" ".join(syn_sentence_a))
        syntrans_text.write('\t')
        syntrans_text.write(str(score))
        syntrans_text.write('\t')
        
        # find co-occuring word
        for co_w in syn_sentence_a:
            if co_w in syn_sentence_q:
                syntrans_text.write(co_w)
                syntrans_text.write(' ')
        syntrans_text.write('\n')
        
    syntrans_text.close()
    raw_text.close()
                    
def find_synonym(word):
    syn_dict = open('Synonym_sets.txt', 'r')
    for line in syn_dict:
        word_code = line.split()[0][:-1] # First 'word' in line
        for dict_word in line.split():
            if word == dict_word:
                syn_dict.close()
                return word_code # Word code found
    syn_dict.close()
    return str(hash(word)) # Word code not found, return a hash value

def calc_idf(syntrans_file, idf_file):
    syntrans_text = open(syntrans_file, 'r')
    n = 0 # number of pairs per question
    prev_index = 0
    co_dict_list = [] # list of dicts for each index
    co_dict = {}
    for line in syntrans_text:
        index, q, a, score, co_words = line.split('\t') # index is not number; co_words is a string w/ all co_words, '\n' included
        if int(index) != prev_index:
            #print index
            co_dict['n'] = n
            co_dict_list.append(co_dict)
            prev_index += 1
            n = 0
            co_dict = {}
        for w in co_words.split():
            if w in co_dict:
                co_dict[w] += 1
            else:
                co_dict[w] = 1
        n += 1
    co_dict['n'] = n
    co_dict_list.append(co_dict)
    print(co_dict_list)
    syntrans_text.close()
    
    syntrans_text = open(syntrans_file, 'r')
    idf_text = open(idf_file, 'w')
    for line in syntrans_text:
        index, q, a, score, co_words = line.split('\t')
        idf_list = [] # per line
        for co_word in co_words.split():
            n_apr = co_dict_list[int(index)][co_word]
            n = co_dict_list[int(index)]['n']
            div = n*1.0 / n_apr
            idf = np.log(div)
            if idf > 0:
                idf_list.append(str(idf))
        idf_list.sort(reverse=True) # large first
        idf_text.write(' '.join(idf_list))
        idf_text.write('\n')
            
    idf_text.close()
    syntrans_text.close()

def gen_word_vec():
    syntrans_text = open('training.syntrans', 'r')
    sentences2vec = [] # corpus
    for line in syntrans_text:
        q = line.split('\t')[1].split()
        a = line.split('\t')[2].split()
        #score = line.split('?')[1].split('.')[1]
        sentences2vec.append(q)
        sentences2vec.append(a)
    model = Word2Vec(sentences2vec,
                     size=config.TrainConfig.WORD_EMBEDDING_DIM,
                     min_count=config.TrainConfig.W2V_MIN_COUNT)
    #print model.wv.vocab
    model.save('training.w2v')
    #new_model = Word2Vec.load(word_vec_file)
    #print new_model.wv.vocab
    syntrans_text.close()

def load_word_vec(word_vec_file):
    model = Word2Vec.load(word_vec_file)
    return model

def gen_input(w2v_model_file, syntrans_file, idf_file): ###########################################3
    sample_text = open(syntrans_file, 'r')
    idf_text = open(idf_file, 'r')
    idf_num = config.TrainConfig.IDF_NUM # number of the biggest idf of co-occurring words accounted for
    #[(batch, None, dim), (...)]
    q_sample = [] # (batch, None, dim), input unit
    a_sample = []
    idf_sample = []
    Y_sample = []
    prev_index = 0
    dic = load_word_vec(w2v_model_file).wv
    dim = config.TrainConfig.WORD_EMBEDDING_DIM
    #i = 0 # count for random encoded words
    #j = 0 # count for all words
    for line in sample_text:
        index, q, a, score, co_words = line.split('\t')
        idf_line = idf_text.readline() # idf file should have the same length as syntrans file | need split
        if int(index) != prev_index: # new batch
            prev_index += 1
            #q_batch_list.append()
            #a_batch_list.append()
            #idf_batch_list.append(pad(idf_sample, padding='post', dtype='float64'))
            if len(Y_sample) > 1:
                yield (pad(q_sample, padding='post', dtype='float64'),
                       pad(a_sample, padding='post', dtype='float64'),
                       pad(idf_sample, padding='post', dtype='float64'),
                       Y_sample)
            else:
                yield (pad(q_sample, padding='post', dtype='float64'),
                       pad(a_sample, padding='post', dtype='float64'),
                       pad(idf_sample, padding='post', dtype='float64'),
                       np.array(Y_sample))
            q_sample = []
            a_sample = []
            idf_sample = []
            Y_sample = []
        word_ebd_list = [dic[word] if word in dic else np.random.rand(dim) for word in q.split()]
        q_sample.append(word_ebd_list)
        word_ebd_list = [dic[word] if word in dic else np.random.rand(dim) for word in a.split()]
        #print len(word_ebd_list)
        a_sample.append(word_ebd_list)
        #for word in q.split():
        Y_sample.append(int(score))
        idf_cnt = 0
        line_idf = []
        for idf in idf_line.split():
            idf_cnt += 1
            if idf_cnt > idf_num:
                break
            line_idf.append(float(idf))
        for i in range(idf_num - len(line_idf)):
            line_idf.append(0)
        idf_sample.append(line_idf)
    # last batch
    if len(Y_sample) > 1:
        yield (pad(q_sample, padding='post', dtype='float64'),
               pad(a_sample, padding='post', dtype='float64'),
               pad(idf_sample, padding='post', dtype='float64'),
               Y_sample)
    else:
        yield (pad(q_sample, padding='post', dtype='float64'),
               pad(a_sample, padding='post', dtype='float64'),
               pad(idf_sample, padding='post', dtype='float64'),
               np.array(Y_sample))
    print("End of sample iterations.")
    
def pad(x, **kwargs):
    if x == []:
        return []
    else:
        return pad_sequences(x, **kwargs)
    
#word_seg("testing.data", "testing.seg")
#word_to_syn("testing.seg", "testing.syntrans")
#calc_idf("testing.syntrans", "testing.idf")
#gen_word_vec("testing.syntrans", "testing.w2v")
#print load_word_vec('testing.w2v')
#(qin, ain, idfin, yin) = gen_input("debug.w2v", "debug.syntrans", "debug.idf")
"""print (len(qin), len(ain), len(idfin), len(yin))
print ain[0]
print ain[1]"""