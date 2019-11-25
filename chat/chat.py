#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import tensorflow as tf
import numpy as np
import tokenization
from processor import DataProcessor


FROZEN_MODEL = 'frozen_chat.pb'
INPUT1_NODE = 'input1_ids:0'
MASK1_NODE = 'input1_mask:0'
OUTPUT_NODE = 'norm1:0'
VOCAB_FILE = 'vocab.txt'
DB_FILE = 'corpora.data'


def get_tensor_graph(model_pb_file):
    with tf.gfile.GFile(model_pb_file, "rb") as f:
        graph_o = tf.GraphDef()
        graph_o.ParseFromString(f.read())
    with tf.Graph().as_default() as G:
        tf.import_graph_def(graph_o,
                            input_map=None,
                            return_elements=None,
                            name='',
                            op_dict=None,
                            producer_op_list=None)
    return G


class QASeverModel(object):

    def __init__(self):
        self.graph = get_tensor_graph(FROZEN_MODEL)
        self.sess = tf.Session(graph=self.graph)
        self.input1_node = self.sess.graph.get_tensor_by_name(
            INPUT1_NODE)
        self.mask1 = self.sess.graph.get_tensor_by_name(
            MASK1_NODE)
        self.output_node = self.sess.graph.get_tensor_by_name(
            OUTPUT_NODE)
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=VOCAB_FILE, do_lower_case=True)
        self.processor = DataProcessor(DB_FILE)
        self.norm2 = np.load('n2.npy')

    def __call__(self, sentence, top_k=1):
        feed_val = self.processor.create_infer_data(''.join(sentence.split(';')), self.tokenizer)
        norm1 = self.sess.run(self.output_node,
                              feed_dict={
                                  self.input1_node: feed_val[0],
                                  self.mask1: feed_val[1],
                              })
        sim = (norm1*self.norm2).sum(axis=-1)

        index = np.argsort(sim)[-top_k:][::-1]

        res = []
        for idx in index:
            res.append([self.processor.data[idx][0], sim[idx]])

        return res
