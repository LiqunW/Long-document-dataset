from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import tensorflow as tf
import numpy as np

import re
import pandas as pd
import csv
from glimpse import GlimpseNet, LocNet
from utils import weight_variable, bias_variable, loglikelihood
from config import Config
from dataloader import Dataloader

config = Config()
dataloader = Dataloader(path=config.ds_path,batchsize=config.batch_size)

logger=logging.getLogger('RL_NLP')
logger.setLevel(logging.INFO)

if config.logging_file:
	hdlr = logging.FileHandler('res.log')
else:
	hdlr = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)


# load_glove_dict
glove = pd.read_table('glove.6B.100d.txt'
					   ,sep=" ",index_col=0,header=None,quoting = csv.QUOTE_NONE)
glove_mat = glove.as_matrix().astype(np.float32)
# 创建一个新的glove字典，把没有的词加进去  4类arxiv 207478
#new_mat = np.random.normal(scale=0.38,size=[600000,300])
#glove_mat = np.concatenate([glove_mat,new_mat],axis=0)
vocab_size, embedding_dim = np.shape(glove_mat)


n_steps = config.step
number_examples = dataloader.nTrain

loc_mean_arr = []
sampled_loc_arr = []


def get_next_input(output, i):
	loc, loc_mean = loc_net(output)
	gl_next = gl(loc)
	loc_mean_arr.append(loc_mean)
	sampled_loc_arr.append(loc) #把每一步的loc存下来，然后再去计算真实的loc
	return gl_next

# placeholders
# 文档大小[batch,词数量*词向量维度],分词程序
document_ph = tf.placeholder(tf.int32,
						[None, config.num_word],name='document_ph')
labels_ph = tf.placeholder(tf.int64, [None],name='labels_ph')
num_doc_ph = tf.placeholder(tf.int64,[None],name='num_doc_ph')

#把glove导如tf图中
embedding_ph = tf.placeholder(tf.float32,[vocab_size,embedding_dim],name='glove_mat')
W = tf.Variable(tf.constant(0.0,shape=[vocab_size,embedding_dim]),trainable=False,name='W')
W2 = tf.Variable(tf.random_uniform([1129256,config.word_vector],-1.,1.,name='W2'))
embedding_matrix = W.assign(embedding_ph)
W = tf.concat([W,W2],axis=0)
# Build the aux nets.
with tf.variable_scope('glimpse_net'):
	gl = GlimpseNet(config, document_ph,num_doc_ph,W)
with tf.variable_scope('loc_net'):
	loc_net = LocNet(config)

# number of examples
N = tf.shape(document_ph)[0]
#init_loc = tf.random_uniform((N, 1), minval=0, maxval=1)
#init_glimpse = gl(init_loc)
# 第一步的时候获取全文信息
init_feature = []
number_of_coarse = 10
for i in range(number_of_coarse):
	init_loc = tf.random_uniform([N,2],minval=0.,maxval=1.)
	init_index = tf.cast(init_loc * tf.cast(tf.tile(tf.reshape(num_doc_ph,[-1,1]),
													[1,2]), tf.float32), tf.int32)
	init_feature.append(gl.context_info(init_index))
init_glimpse = tf.zeros(tf.shape(init_feature[0]),dtype=tf.float32)
for i in range(number_of_coarse):
	init_glimpse+=init_feature[i]
init_glimpse = tf.nn.relu(init_glimpse)

# Core network.
lstm_cell = tf.contrib.rnn.LSTMCell(config.cell_size, initializer=tf.orthogonal_initializer(),
							  activation=tf.tanh, state_is_tuple=True)
init_state = lstm_cell.zero_state(N, tf.float32)
inputs = [init_glimpse]
inputs.extend([0] * (config.num_glimpses))
outputs, _ = tf.contrib.legacy_seq2seq.rnn_decoder(
	inputs, init_state, lstm_cell, loop_function=get_next_input)

sample_loc = tf.stack(sampled_loc_arr,name='sample_loc')
# Time independent baselines
with tf.variable_scope('baseline'):
	w_baseline = weight_variable((config.cell_output_size, 1))
	b_baseline = bias_variable((1,))
baselines = []
for t, output in enumerate(outputs[1:]):
	baseline_t = tf.nn.xw_plus_b(output, w_baseline, b_baseline)
	baseline_t = tf.squeeze(baseline_t)
	baselines.append(baseline_t)
baselines = tf.stack(baselines)  # [timesteps, batch_sz]
# baselines = tf.transpose(baselines)  # [batch_sz, timesteps]
baselines = tf.reshape(baselines,(config.batch_size, config.num_glimpses))

# Take the last step only.
output = outputs[-1]
# Build classification network.
with tf.variable_scope('cls'):
	w_logit = weight_variable((config.cell_output_size, config.num_classes))
	b_logit = bias_variable((config.num_classes,))

logits = tf.nn.xw_plus_b(output, w_logit, b_logit)
softmax = tf.nn.softmax(logits,name='softmax')

# cross-entropy.
xent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_ph, logits=logits)
xent = tf.reduce_mean(xent,name='xent')
pred_labels = tf.argmax(logits, 1)
# 0/1 reward.
reward = tf.cast(tf.equal(pred_labels, labels_ph), tf.float32)
rewards = tf.expand_dims(reward, 1)  # [batch_sz, 1]
rewards = tf.tile(rewards, (1, config.num_glimpses))  # [batch_sz, timesteps]
logll = loglikelihood(loc_mean_arr, sampled_loc_arr, config.loc_std)
advs = rewards - tf.stop_gradient(baselines)
logllratio = tf.reduce_mean(logll * advs)
reward = tf.reduce_mean(reward)

baselines_mse = tf.reduce_mean(tf.square((rewards - baselines)))
var_list = tf.trainable_variables()
# hybrid loss
loss = -logllratio + xent + baselines_mse  # `-` for minimize
grads = tf.gradients(loss, var_list)
grads, _ = tf.clip_by_global_norm(grads, config.max_grad_norm)

# learning rate
global_step = tf.get_variable(
	'global_step', [], initializer=tf.constant_initializer(0), trainable=False)

# 需要根据样本数量的大小进行修正
training_steps_per_epoch = number_examples // config.batch_size

starter_learning_rate = config.lr_start
# decay per training epoch
learning_rate = tf.train.exponential_decay(
	starter_learning_rate,
	global_step,
	training_steps_per_epoch,
	0.98,
	staircase=True)
learning_rate = tf.maximum(learning_rate, config.lr_min)
opt = tf.train.AdamOptimizer(learning_rate)
train_op = opt.apply_gradients(zip(grads, var_list), global_step=global_step)


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	t0=time.time()
	words_num = np.load(config.doc_totalword_file)
	#先运行glove op
	sess.run(embedding_matrix,feed_dict={embedding_ph:glove_mat})
	for i in range(n_steps):
		train_docs, train_labels, train_flag, out_filename_train = dataloader.next_train_batch()
		doc_num_train = np.zeros([config.batch_size], dtype='int32')

		if train_flag == False:
			for count_train, file_name_train in enumerate(out_filename_train):
				temp_train = int(re.findall(r"\d+?\d*", file_name_train)[-1])
				doc_num_train[count_train] = words_num[temp_train]
			doc_num_train = np.clip(doc_num_train, 0, config.num_word)
			train_docs = np.reshape(train_docs,[config.batch_size,-1])
			loc_net.samping = True
			fetches = [advs, baselines_mse, xent, logllratio,
					   reward, loss, learning_rate, train_op,sample_loc]
			feed_dict = {document_ph: train_docs,
						 labels_ph: train_labels,
						 num_doc_ph:doc_num_train
						}
			results = sess.run(fetches, feed_dict=feed_dict)

			adv_val, baselines_mse_val, xent_val, logllratio_val,\
			reward_val, loss_val, lr_val, _, sample_loc_val = results


		if i and i % 50 == 0:
			t_run = time.time() - t0
			t0=time.time()
			logger.info('step {} cost {:3.2f} secs: lr = {:3.6f}'.format(i, t_run,lr_val))
			logger.info(
			  'step {}: reward = {:3.4f}\tloss = {:3.4f}\txent = {:3.4f}'.format(
				  i, reward_val, loss_val, xent_val))
			logger.info('llratio = {:3.4f}\tbaselines_mse = {:3.4f}'.format(
				logllratio_val, baselines_mse_val))
#验证
		if i and i % config.val_freq ==0:
			t_val=0
			step_per_epoch = dataloader.nVal // config.batch_size
			correct_cnt = 0
			num_samples = 0
			loc_net.samping = False
			for val_step in range(step_per_epoch):
				t_batch_val = time.time()-t_val
				t_val=time.time()
				print('val time per batch',t_batch_val)
				doc_num_eval = np.zeros([config.batch_size], dtype='int32')
				eval_doc,eval_labels,eval_flag,out_filename_eval = dataloader.next_val_batch()

				for count_eval, file_name_eval in enumerate(out_filename_eval):
					temp_eval = int(re.findall(r"\d+?\d*", file_name_eval)[-1])
					doc_num_eval[count_eval] += words_num[temp_eval]

				doc_num_eval = np.clip(doc_num_eval, 0, config.num_word)

				eval_doc = np.reshape(eval_doc, [config.batch_size, -1])
				if eval_flag is False:
					softmax_val = sess.run(softmax,feed_dict={
										document_ph:eval_doc,
										labels_ph:eval_labels,
										num_doc_ph: doc_num_eval
										})
					pred_labels_val = np.argmax(softmax_val,1)
					pred_labels_val = pred_labels_val.flatten()
					correct_cnt += np.sum(pred_labels_val == eval_labels)
					num_samples += config.batch_size
			acc = correct_cnt / num_samples
			logger.info('valid accuracy = {:3.4f}: {}/{}'.format(acc, correct_cnt, num_samples))

	step_per_epoch = dataloader.nTest // config.batch_size
	correct_cnt = 0
	num_samples = 0
	loc_net.samping = False
	for test_step in range(step_per_epoch):
		doc_num_test = np.zeros([config.batch_size], dtype='int32')
		test_doc, test_labels, test_flag, out_filename_test = dataloader.next_val_batch()

		for count_test, file_name_test in enumerate(out_filename_test):
			temp_test = int(re.findall(r"\d+?\d*", file_name_test)[-1])
			doc_num_test[count_test] += words_num[temp_test]

		doc_num_test = np.clip(doc_num_test, 0, config.num_word)

		test_doc = np.reshape(test_doc, [config.batch_size, -1])
		if test_flag is False:
			softmax_test = sess.run(softmax, feed_dict={
				document_ph: test_doc,
				labels_ph: test_labels,
				num_doc_ph: doc_num_test
				})
			pred_labels_test = np.argmax(softmax_test, 1)
			pred_labels_test = pred_labels_test.flatten()
			correct_cnt += np.sum(pred_labels_test == test_labels)
			num_samples += config.batch_size
	acc = correct_cnt / num_samples
	logger.info('test accuracy = {:3.4f}: {}/{}'.format(acc, correct_cnt, num_samples))
