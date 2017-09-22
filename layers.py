#! -*- coding:utf-8 -*-
'''
@author: Kai Zheng
'''
from keras.layers import Dense, Activation, Conv2D, Input, Lambda
from keras.activations import softmax
from keras.layers.pooling import GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers.merge import Concatenate, Dot
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras import backend as K
from keras.engine.topology import Layer
#from keras.models import Sequential
# from theano.tensor import max as mx, sum, mean, shape
#import numpy as np
import config

"""

One perplexing problem is:
It seems not possible to build a layer taking
X(None, 2, dim)
[[X[0]:...vec...]
[X[1]:...vec...]]
as input and compute X[0] dot M(dim x dim) dot X[1].T = a scalar
and output the scalar.

Probable explanation is : X dot X.T is scalar; X.T dot X is d-dim matrix

-------------------------------------

Layers passing tensor variables not tensors

-------------------------------------

Whenever an error is raised, it appears that it's in the core, but actually in the code logic:

-------------------------------------

keras doesn't support different size of input in the same batch.
feeding 1 sample and add a dimension np.array([sample]) will work for that:but gradient descent will be painful
using max pooling can ignore all padding time steps

-------------------------------------

(1,2,3)'s transpose is (3,2,1), yielding problems for batch_dot, and not dot

"""

class DL4AQS_model(object):
    def __init__(self):
        dim = config.TrainConfig.WORD_EMBEDDING_DIM
        idf_num = config.TrainConfig.IDF_NUM
        q = Input(batch_shape=(1, None, None, dim)) # expected (batch(1), sentence_num, word_num(max), dim)
        a = Input(batch_shape=(1, None, None, dim))
        idf = Input(batch_shape=(1, None, idf_num))
        # Convolution
        cnv1_layer = Conv2D(batch_input_shape=(1, None, None, dim),
                             filters=dim, 
                             kernel_size=(1, 2), 
                             activation='tanh', 
                             use_bias=True, 
                             bias_initializer='zeros') # output: (batch(1), sentence_num, word_num(max)-1, dim)
        q_c = cnv1_layer(q) # expected (sequence_len-1, dim)
        a_c = cnv1_layer(a)
        
        # Pooling
        #pooling_layer = max_pool_layer(wdim=dim)
        pooling_layer = TimeDistributed(GlobalAveragePooling1D())
        q_s = pooling_layer(q_c) # (batch(1), sentence_num, 50)
        a_s = pooling_layer(a_c)
        
        # Compare similarity
        # q dot M
        qM_ly = TimeDistributed(qM_layer(output_dim=dim)) # q dot M layer, halfway done for similarity
        #qM_neg_ly = qM_layer(output_dim=dim) # for pr(not matched)
        q_M_dot = qM_ly(q_s) # TimeDist should never be added here
        #q_M_neg_dot = qM_neg_ly(q_s)
        
        # Dot Q and A
        # Concatenate
        q_a_c = Concatenate(axis=-1)([q_M_dot, a_s])
        #sc_ly = Lambda(lambda x: K.batch_dot(x[1][:], x[0][:]))
        sc_ly = TimeDistributed(score_layer(wdim=dim))
        sc = sc_ly(q_a_c)
        
        
        sc_i_pair = [sc, idf]
        idf_sc_ly = idf_score_layer(idf_dim=idf_num)
        idf_sc = idf_sc_ly(sc_i_pair)
        
        pr = Activation('softmax')(idf_sc)
        
        """print sc.ndim
        print shape(sc)"""
        """sc_raw_ly = Dot(axes=2) # score we want lies in the diagonal
        sc_r = sc_raw_ly([q_M_dot, a_s]) # TimeDistributed doesn't apply to list of tensors
        
        diag_ly = diag_layer()
        sc = diag_ly(sc_r)"""
        
        #pr_ly = Activation(lambda x: K.softmax(x, axis=1)) # axis kw not understood
        #pr = softmax(sc, axis=1)

        # To (0, 1)
        #ac_ly = Activation('sigmoid')
        #pr = ac_ly(sc_r)
        
        qa_idf_pair = (q, a, idf)
        self.model = Model(inputs = qa_idf_pair, outputs = pr)
        #lf = config.TrainConfig.LOSS_FUNC
        #optmz = config.TrainConfig.OPTIMIZER
        #self.model.compile(optimizer=optmz, loss=lf)

        print(self.model.summary())
        
        """# Concatenate product
        q_a_pair = Concatenate(axis=1)([q_M_dot, a_s]) # from here, Model input should be q, a
        #q_a_neg_pair = Concatenate(axis=1)([q_M_neg_dot, a_s])
        sc_ly = score_layer(output_dim=1, wdim=dim) # q and a similarity result layer
        #sc_neg_ly = score_layer(output_dim=1)
        sc = sc_ly(q_a_pair)
        #sc_neg = sc_neg_ly(q_a_neg_pair)"""
        
        
        """
        # Modifying
        
        pooling_layer = Average() # output: (1, dim)
        q_s = pooling_layer(q_c) # expected (1, dim)
        a_s = pooling_layer(a_c)
        
        s = Sequential()
        s.add(cnv1_layer)
        s.add(pooling_layer)
        print s.summary()
        self.model = Model(inputs = (q, a), outputs = q_s)
        print self.model.summary()
        
        #
        #self.model.add(TimeDistributed(cnv1_layer, input_shape=(2, 10, dim)))
        #filters=1, kernal_size=2, activation='tanh', batch_input_shape=(64, 10, config.TrainConfig.WORD_EMBEDDING_DIM)

        """
class qM_layer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim # qM is expected to be the same size of q
        super(qM_layer, self).__init__(**kwargs)
        
    def build(self, input_shape): # input q: (batch(1), dim)
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim), # M is dim x dim; keepdim then shape[2]
                                      initializer='uniform',
                                      trainable=True)
        #self.bias = self.add_weight(name='bias', shape=(1,), initializer='zeros', trainable=True)
        super(qM_layer, self).build(input_shape)
        
    def call(self, X): # input (batch, dim)
        ret = K.dot(X, self.kernel) # (batch, dim)
        #ret = K.dot(tmp.T, X[:, 1, :]) #(batch, 1)
        #b = self.bias
        return ret #... attribute:_keras_shape???
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)# 1, self.output_dim)
    
class score_layer(Layer):
    def __init__(self, wdim, **kwargs): # input (batch(1), 2*dim)
        self.output_dim = (1,)
        self.wdim = wdim
        super(score_layer, self).__init__(**kwargs)
        
    def build(self, input_shape): # do
        self.bias = self.add_weight(name='bias', 
                                    shape=self.output_dim,
                                    initializer='zeros',
                                    trainable=True)
        super(score_layer, self).build(input_shape)
        
    def call(self, X): # input (batch(1), 2*dim)
        # the whole batch will be regarded as X
        #X_1 = X[0].dimshuffle((0, 2, 1))
        #X_2 = X[1]#
        #X_2 = K.transpose(X[1])
        
        # concatenated
        X_1 = X[:, 0:self.wdim]
        X_2 = X[:, self.wdim:2*self.wdim]
        prod = X_1 * X_2 # X dot X.T is scalar; X.T dot X is d-dim matrix
        b = self.bias
        return prod[:, 0] + b # output (1, ) ; softmax should never be used here for this is a timedist layer
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],)# self.output_dim)
    
class idf_score_layer(Layer):
    def __init__(self, idf_dim, **kwargs): # input (batch(1), 2*dim)
        self.output_dim = 1
        self.idf_dim = idf_dim
        super(idf_score_layer, self).__init__(**kwargs)
    def build(self, input_shape): # do 
        """self.kernel = self.add_weight(name='kernel', 
                                    shape=(self.output_dim, 1),
                                    initializer='uniform',
                                    trainable=True)"""
        super(idf_score_layer, self).build(input_shape)
    def call(self, X): # input : [(1, sen_num), (1, sen_num, idf_num) ]
        ori_score = X[0]
        idf_score = X[1]
        idf_sum = K.sum(idf_score, axis=2) / self.idf_dim
        #ret = ori_score + self.kernel * idf_sum
        ret = ori_score + idf_sum
        return ret
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1])# self.output_dim) (1, sentence_num)

"""
# batch dot won't work. the res is a matrix with diagnonal elements being the needed output. 
class score_layer(Layer):
    def __init__(self, **kwargs): # input [(batch(1), s_num, dim), (batch(1), s_num, dim)]
        self.output_dim = 1
        super(score_layer, self).__init__(**kwargs)
        
    def build(self, input_shape): # do 
        self.bias = self.add_weight(name='bias', 
                                    shape=(self.output_dim),
                                    initializer='zeros',
                                    trainable=True)
        super(score_layer, self).build(input_shape)
        
    def call(self, X): # input [(batch(1), s_num, dim), (batch(1), s_num, dim)]
        # the whole batch will be regarded as X
        X_1 = X[0]
        X_2 = X[1].dimshuffle((0, 2, 1))
        prod = K.batch_dot(X_1, X_2.T) # X dot X.T is scalar; X.T dot X is d-dim matrix
        b = self.bias
        return prod + b # output (batch, dim)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)"""
    
"""class diag_layer(Layer):
    def __init__(self, **kwargs): # input (batch(1), s_num, s_num)
        self.output_dim = 1
        super(diag_layer, self).__init__(**kwargs)
        
    def build(self, input_shape): # do 
        self.bias = self.add_weight(name='bias', 
                                    shape=(self.output_dim),
                                    initializer='zeros',
                                    trainable=True)
        super(diag_layer, self).build(input_shape)
        
    def call(self, X): # input (batch(1), s_num, s_num)
        b = self.bias
        return diag(X[0]) + b # output (batch(1), s_num, 1) AFTER diag, ndim = 1
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 1)"""
        
"""class ave_pool_layer(Layer):
    def __init__(self, wdim, **kwargs): # input (batch, None, dim)
        self.output_dim = wdim
        super(ave_pool_layer, self).__init__(**kwargs)
        
    def build(self, input_shape): # do 
        super(ave_pool_layer, self).build(input_shape)
        
    def call(self, X): # input (batch, None, dim)
        return mean(X, axis=1, keepdims=True) # output (batch, dim)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1, self.output_dim)
    
class max_pool_layer(Layer):
    def __init__(self, wdim, **kwargs): # input (batch, None, dim)
        self.output_dim = wdim
        super(max_pool_layer, self).__init__(**kwargs)
        
    def build(self, input_shape): # do 
        super(max_pool_layer, self).build(input_shape)
        
    def call(self, X): # input (batch, None, dim)
        return mx(X, axis=1)#, keepdims=True) # output (batch, dim)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)# (1, sentence_num, self.output_dim)"""
    


#DL4AQS_model() # no code here, otherwise it will be executed