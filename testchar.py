# python3

# coding: utf-8

# ELMo usage example with character inputs.

import os
import sys
import numpy as np
import tensorflow as tf
from gensim.matutils import unitvec
from smart_open import open
from bilm import Batcher, BidirectionalLanguageModel, weight_layers

corpusfile = sys.argv[1]
raw_sentences = []

with open(corpusfile, 'r') as f:
    for line in f:
        res = line.strip()
        raw_sentences.append(res)

raw_sentences = raw_sentences

tokenized_sentences = [sentence.split() for sentence in raw_sentences]

print('Sentences:', len(tokenized_sentences))

datadir = sys.argv[2]
vocab_file = os.path.join(datadir, 'vocab.txt.gz')
options_file = os.path.join(datadir, 'options.json')
weight_file = os.path.join(datadir, 'model.hdf5')

# Create a Batcher to map text to character ids.
batcher = Batcher(vocab_file, 50)

# Input placeholders to the biLM.
sentence_character_ids = tf.placeholder('int32', shape=(None, None, 50))

# Build the biLM graph.
bilm = BidirectionalLanguageModel(options_file, weight_file, max_batch_size=200)

# Get ops to compute the LM embeddings.
sentence_embeddings_op = bilm(sentence_character_ids)

# Get an op to compute ELMo (weighted average of the internal biLM layers)
# Our model includes ELMo at both the input and output layers
# of the task GRU, so we need 2x ELMo representations at each of the input and output.

elmo_sentence_input = weight_layers('input', sentence_embeddings_op, use_top_only=True)

# elmo_sentence_output = weight_layers('output', sentence_embeddings_op, l2_coef=0.0)

with tf.Session() as sess:
    # It is necessary to initialize variables once before running inference.
    sess.run(tf.global_variables_initializer())

    # model_vars = tf.global_variables()
    # a = slim.model_analyzer.analyze_vars(model_vars, print_info=True)
    # print(a)

    # Create batches of data.
    sentence_ids = batcher.batch_sentences(tokenized_sentences)

    # Compute ELMo representations (here for the input only, for simplicity).
    elmo_sentence_input_ = sess.run(elmo_sentence_input['weighted_op'],
                                    feed_dict={sentence_character_ids: sentence_ids})
    print(elmo_sentence_input_.shape)

    query_nr = 5

    query_word = tokenized_sentences[0][query_nr]
    print('Query:', query_word)
    query_vec = elmo_sentence_input_[0][query_nr, :]
    query_vec = unitvec(query_vec)
    print(query_vec.shape)

    for sent_nr, sent in enumerate(tokenized_sentences):
        if sent_nr == 0:
            continue
        print('======')
        print(sent)
        sims = {}
        for nr, word in enumerate(sent):
            w_vec = elmo_sentence_input_[sent_nr][nr, :]
            w_vec = unitvec(w_vec)
            sims[word] = np.dot(query_vec, w_vec)

        for k in sorted(sims, key=sims.get, reverse=True):
            print(k, sims[k])
