from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from utils import weight_variable, bias_variable
from config import Config
config = Config()


class Extract_word(object):
  def __init__(self, num_words, window_size):
      self.num_words = tf.cast(tf.reshape(num_words, [-1]),tf.int32)
      self.window_size = window_size

  def f1(self,index,num_word):
      return tf.range(2 * self.window_size)

  def f2(self, index,num_word):
      return tf.range(num_word - 2 * self.window_size, num_word)

  def f3(self,index,num_word):
      mask = tf.range(-self.window_size, self.window_size)
      return index + mask

  def extract_fn(self,index,doc_num):
      r = tf.case({tf.less(index, self.window_size): lambda :self.f1(index,doc_num),
                   tf.greater(index, doc_num - self.window_size): lambda :self.f2(index,doc_num)},
                  default= lambda :self.f3(index,doc_num), exclusive=True)
      return r

  def words_extracter(self, indexs):
      fn = lambda x:self.extract_fn(x[0], x[1])
      indexs = tf.reshape(indexs,[-1])
      elems = (indexs,self.num_words)
      result = tf.map_fn(fn,elems,dtype=tf.int32)
      result = tf.expand_dims(result,axis=2)

      replicated_result = tf.expand_dims(tf.tile(tf.expand_dims(tf.range(tf.shape(result)[0]), dim=1),
                                                 [1, tf.shape(result)[1]]), axis=2)
      output = tf.concat([replicated_result, result], axis=2)

      return output

  def __call__(self,indexs):
      index_for_words = self.words_extracter(indexs)
      return index_for_words




class GlimpseNet(object):
  """Glimpse network.

  Take glimpse location input and output features for RNN.

  """

  def __init__(self, config, document_ph,num_word_ph,embedding_matrix):
    self.word_vector = config.word_vector
    self.hl_size = config.hl_size
    self.g_size = config.g_size
    self.loc_dim = config.loc_dim
    self.batch_size = config.batch_size
    self.document_ph = document_ph
    self.num_word_ph = num_word_ph
    #self.dorpout_prob = dropout
    self.embedding_mat = embedding_matrix  #glove矩阵
    self.filter_nums = config.filter_nums
    self.kernel_size1 = config.kernel_size1
    self.kernel_size2 = config.kernel_size2
    self.kernel_size3 = config.kernel_size3
    self.cell_size = config.cell_size
    self.max_pool_size1 = config.maxpool_size1
    self.max_pool_size2 = config.maxpool_size2
    self.max_pool_size3 = config.maxpool_size3
    self.init_weights()
    self.exwords = Extract_word(num_word_ph, config.window_size_exw)
    self.extract_context=Extract_word(num_word_ph,config.random_ws_exw)
  def init_weights(self):
    """ Initialize all the trainable weights."""
    self.w_l0 = weight_variable((self.loc_dim, self.hl_size))
    self.b_l0 = bias_variable((self.hl_size,))
    self.w_l1 = weight_variable((self.hl_size, self.g_size))
    self.b_l1 = weight_variable((self.g_size,))
    self.w_g = weight_variable((self.filter_nums*3,self.g_size))
    self.b_g = bias_variable((self.g_size,))
    self.filter1 = weight_variable((self.kernel_size1, self.word_vector, self.filter_nums))
    self.filter2 = weight_variable((self.kernel_size2, self.word_vector, self.filter_nums))
    self.filter3 = weight_variable((self.kernel_size3, self.word_vector, self.filter_nums))
    self.bias = tf.Variable(tf.constant(0.1,shape=[self.filter_nums]))

  def lookup_embedding(self,doc_idx):
    results=tf.nn.embedding_lookup(self.embedding_mat,doc_idx)
    return results

  def conv_net(self,words):
    cnn_f1 = tf.nn.conv1d(words, filters=self.filter1, stride=1, padding='VALID')
    h1=tf.nn.relu(tf.nn.bias_add(cnn_f1,self.bias))
    max_p1 = tf.layers.max_pooling1d(h1, pool_size=self.max_pool_size1, strides=1)
    cnn_f2 = tf.nn.conv1d(words, filters=self.filter2, stride=1, padding='VALID')
    h2 = tf.nn.relu(tf.nn.bias_add(cnn_f2, self.bias))
    max_p2 = tf.layers.max_pooling1d(h2, pool_size=self.max_pool_size2, strides=1)
    cnn_f3 = tf.nn.conv1d(words, filters=self.filter3, stride=1, padding='VALID')
    h3 = tf.nn.relu(tf.nn.bias_add(cnn_f3, self.bias))
    max_p3 = tf.layers.max_pooling1d(h3, pool_size=self.max_pool_size3, strides=1)
    h_pool = tf.concat([max_p1,max_p2,max_p3],2)
    h_pool_flat = tf.reshape(h_pool,[-1, self.filter_nums*3])
    #h_drop = tf.nn.dropout(h_pool_flat,self.dorpout_prob)
    h_fc = tf.nn.xw_plus_b(h_pool_flat,self.w_g,self.b_g)
    return h_fc

  def context_info(self,index):
    #每次取连续的10个词
    index_ext0 = self.extract_context(index[:,0])
    index_ext1 = self.extract_context(index[:,1])
    context_idx0 = tf.gather_nd(params=self.document_ph,indices=index_ext0)
    context_idx1 = tf.gather_nd(params=self.document_ph,indices=index_ext1)
    context_idx = tf.concat([context_idx0,context_idx1],axis=1)
    context_words = self.lookup_embedding(context_idx)
    feature = self.conv_net(context_words)
    feature = tf.nn.relu(feature)
    return feature


  def get_glimpse(self, loc):
    """Take glimpse on the original document."""
    loc = tf.reshape(loc,[-1])
    index = tf.cast((loc+1)/2 * tf.cast(self.num_word_ph,tf.float32),tf.int32)
    #(batch,num_word)

#取出与位置index相邻的20个词
    extract_words_index = self.exwords(index)
    glimpse_docs = tf.gather_nd(params=self.document_ph, indices=extract_words_index)
    doc_words = self.lookup_embedding(glimpse_docs)
# CNN部分 (batch,20,300)
    feature_ew = self.conv_net(doc_words)
    return feature_ew

  def __call__(self, loc):
    glimpse_input1 = self.get_glimpse(loc)
    l = tf.nn.relu(tf.nn.xw_plus_b(loc, self.w_l0, self.b_l0))
    l = tf.nn.xw_plus_b(l, self.w_l1, self.b_l1)
    # 不包含随机抽样
    g = tf.nn.relu(glimpse_input1+l)
    return g


class LocNet(object):
  """ Location network.
  Take output from other network and produce and sample the next location.
  """

  def __init__(self, config):
    self.loc_dim = config.loc_dim
    self.input_dim = config.cell_output_size
    self.loc_std = config.loc_std
    self._sampling = True

    self.init_weights()

  def init_weights(self):
    self.w = weight_variable((self.input_dim, self.loc_dim))
    self.b = bias_variable((self.loc_dim,))
    self.wh = weight_variable((self.input_dim,self.input_dim))
    self.bh = bias_variable((self.input_dim,))
  def __call__(self, input):
    input = tf.stop_gradient(input)
    h = tf.nn.tanh(tf.nn.xw_plus_b(input,self.wh,self.bh))
    mean = tf.clip_by_value(tf.nn.xw_plus_b(h,self.w,self.b),-1.,1.)
    #mean = tf.clip_by_value(tf.nn.xw_plus_b(input, self.w, self.b), -1., 1.)
    #mean = tf.stop_gradient(mean)
    if self._sampling:
      loc = mean + tf.random_normal(
          (tf.shape(input)[0], self.loc_dim), stddev=self.loc_std)
      loc = tf.clip_by_value(loc, -1., 1.)
    else:
      loc = mean
    loc = tf.stop_gradient(loc)
    return loc, mean

  @property
  def sampling(self):
    return self._sampling

  @sampling.setter
  def sampling(self, sampling):
    self._sampling = sampling
