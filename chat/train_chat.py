# encoding=utf-8

from modeling import BertConfig as ModelConfig, BertModel as Model

import os
import codecs
from tqdm import tqdm
from hp_chat import *
import tensorflow as tf
# import jieba_tokenization as tokenization
import tokenization
import optimization
import jieba

from random import shuffle, random, choice, randint, triangular
from modeling import get_assignment_map_from_checkpoint
import numpy as np
from textrank4zh import TextRank4Keyword


class QAModel(object):

    def __init__(self, model_config, num_labels, batch_size, num_train_steps=None, is_training=True):
        self.input1_ids = tf.placeholder(shape=(None, max_seq_length), dtype=tf.int32, name='input1_ids')
        self.input1_mask = tf.placeholder(shape=(None, max_seq_length), dtype=tf.int32, name='input1_mask')

        self.input2_ids = tf.placeholder(shape=(None, max_seq_length), dtype=tf.int32, name='input2_ids')
        self.input2_mask = tf.placeholder(shape=(None, max_seq_length), dtype=tf.int32, name='input2_mask')

        self.label_id = tf.placeholder(shape=(None,), dtype=tf.int32, name='label_id')

        with tf.variable_scope(''):
            model1 = Model(
                config=model_config,
                is_training=is_training,
                input_ids=self.input1_ids,
                input_mask=self.input1_mask,
                scope='bert'
            )

        with tf.variable_scope('', reuse=True):
            model2 = Model(
                config=model_config,
                is_training=is_training,
                input_ids=self.input2_ids,
                input_mask=self.input2_mask,
                scope='bert'
            )

        # output1 = model1.get_pooled_output()
        # output2 = model2.get_pooled_output()

        output1 = model1.get_sequence_output()
        output2 = model2.get_sequence_output()

        self._build_graph_sequence_cosine(num_labels, output1, output2, num_train_steps, is_training)

        if is_training:
            self.loss = tf.losses.mean_squared_error(self.label_id, self.similarity)

            num_warmup_steps = int(num_train_steps * num_warmup_proportion)
            self.train_op, self.global_step, self.global_norm = optimization.create_optimizer(self.loss,
                                                                                              learning_rate,
                                                                                              num_train_steps,
                                                                                              num_warmup_steps,
                                                                                              use_tpu=False)

            self._make_saver()
        else:
            self.saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=3)

    def _make_saver(self):
        self.saver = tf.train.Saver(var_list=tf.trainable_variables() + [self.global_step],
                                    max_to_keep=3)
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('global_norm', self.global_norm)
        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(os.path.join(output_dir, 'train_summary'),
                                                    tf.get_default_graph())

    def _build_graph_sequence_cosine(self, num_labels, output1, output2, num_train_steps, is_training):
        hidden_size = output1.shape[-1].value * output1.shape[-2].value

        output1 = tf.reshape(output1, (-1, hidden_size))
        output2 = tf.reshape(output2, (-1, hidden_size))

        with tf.variable_scope('output_post_process'):
            output_weights_1 = tf.get_variable(
                "output_weights_1", [256, hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02))

            output_bias_1 = tf.get_variable(
                "output_bias_1", [256], initializer=tf.zeros_initializer())

            # output_weights_2 = tf.get_variable(
            #     "output_weights_2", [256, hidden_size],
            #     initializer=tf.truncated_normal_initializer(stddev=0.02))
            #
            # output_bias_2 = tf.get_variable(
            #     "output_bias_2", [256], initializer=tf.zeros_initializer())

        output1 = tf.matmul(output1, output_weights_1, transpose_b=True)
        output1 = tf.nn.bias_add(output1, output_bias_1)
        output2 = tf.matmul(output2, output_weights_1, transpose_b=True)
        output2 = tf.nn.bias_add(output2, output_bias_1)

        self.norm1 = tf.nn.l2_normalize(output1, -1, name='norm1')
        self.norm2 = tf.nn.l2_normalize(output2, -1, name='norm2')
        self.similarity = tf.reduce_sum(tf.multiply(self.norm1, self.norm2), axis=-1, name='similarity')

    def train(self, sess, feed_values, summary=False, saver=False):
        sess.run(self.train_op, feed_dict=self.make_feed_dict(*feed_values))

        if summary:
            summary = sess.run(self.summary_op,
                               feed_dict=self.make_feed_dict(*feed_values))
            self.summary_writer.add_summary(summary, self.global_step.eval(session=sess))

        if saver:
            self.saver.save(sess, output_dir + 'ckpt', global_step=self.global_step)

    def save_n2(self, sess, feed_values):
        n2 = sess.run(self.norm2, feed_dict=self.make_feed_dict(*feed_values))
        np.save('result/n2.npy', n2)

    def infer(self, sess, feed_values):
        # n1, n2, sim = sess.run([self.norm1, self.norm2, self.similarity], feed_dict=self.make_feed_dict(*feed_values))
        return sess.run(self.norm1, feed_dict={self.input1_ids: feed_values[0], self.input1_mask: feed_values[1]})

    def restore_model(self, sess):
        sess.run(tf.global_variables_initializer())
        try:
            latest_ckpt = tf.train.latest_checkpoint(output_dir)
            self.saver.restore(sess, latest_ckpt)
        except Exception, e:
            print(e)
            # self.restore_ckpt_global_step(init_checkpoint='al_model/albert_model.ckpt',
            #                               include_global_step=False,
            #                               prefix='bert/embeddings')
            # tf.train.init_from_checkpoint('al_model/albert_model.ckpt',
            #                               {
            #                                   'bert/embeddings/LayerNorm/beta': 'reuse_model/Sent/embeddings/LayerNorm/beta',
            #                                   'bert/embeddings/LayerNorm/gamma': 'reuse_model/Sent/embeddings/LayerNorm/gamma',
            #                                   'bert/embeddings/word_embeddings': 'reuse_model/Sent/embeddings/word_embeddings',
            #                                   'bert/embeddings/word_embeddings_2': 'reuse_model/Sent/embeddings/word_embeddings_2'})

    def restore_ckpt_global_step(self, init_checkpoint=None, include_global_step=True, prefix=None):
        tf.global_variables_initializer().run()

        tvars = tf.trainable_variables()
        if include_global_step:
            tvars += [self.global_step]
        initialized_variable_names = {}

        if init_checkpoint:
            assignment_map, initialized_variable_names = \
                get_assignment_map_from_checkpoint(tvars, init_checkpoint)

            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        print("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            print("  name = %s, shape = %s%s", var.name, var.shape,
                  init_string)

    def make_feed_dict(self, id1, mask1, id2, mask2, labels):
        return {
            self.input1_ids: id1,
            self.input1_mask: mask1,
            self.input2_ids: id2,
            self.input2_mask: mask2,
            self.label_id: labels}


class DataProcessor(object):

    def __init__(self, file_dir, separator=u'=', is_training=False, index_file=None):
        self.next_idx = 0
        # self.desc = []
        self.data = []
        self.file_dir = file_dir
        self.separator = separator

        if not index_file:
            # train
            with codecs.open(file_dir, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    try:
                        orig = line.strip()
                        chat, reply = orig.split(separator)
                        assert reply
                        if is_training:
                            # training format
                            self.data.append([chat, reply])
                        else:
                            # infer format
                            self.data.append([chat, reply])

                    except ValueError:
                        print(u'badly formatted line: ' + line)
                    except AssertionError:
                        print('reply is empty')

        else:
            # infer
            with codecs.open(index_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.data.append([line])

    def create_train_data(self, batch_size, tokenizer):
        input1 = []
        mask1 = []
        input2 = []
        mask2 = []
        label = []

        for _ in range(batch_size):
            if self.next_idx >= len(self.data):
                shuffle(self.data)
                self.next_idx = 0

            conv = self.data[self.next_idx]

            if random() < 0.4:
                # label 1, similar pair
                label.append(1)
                q1 = conv[0]
                q2 = conv[1]
                # q2 = q1

            else:
                # label 0, dissimilar pair
                label.append(-1)
                q1 = conv[0]
                q2 = choice(choice(self.data))

            if random() < 0.4:
                q1_tokens = list(jieba.cut(q1))
                q2_tokens = list(jieba.cut(q2))

                # triangular distribution
                len1 = len(q1_tokens)
                len2 = len(q2_tokens)
                mode1 = randint(3, 7)
                mode2 = randint(3, 7)
                rand_len1 = triangular(low=1, high=len1, mode=mode1)
                rand_len2 = triangular(low=1, high=len2, mode=mode2)
                keep_prob1 = rand_len1 / len1
                keep_prob2 = rand_len2 / len2

                mask_q1 = [random() < keep_prob1 for _ in range(len(q1_tokens))]
                mask_q2 = [random() < keep_prob2 for _ in range(len(q2_tokens))]
                q1_new = ''.join([q1_tokens[i] if mask_q1[i] else '' for i in range(len(q1_tokens))])
                q2_new = ''.join([q2_tokens[i] if mask_q2[i] else '' for i in range(len(q2_tokens))])
            else:
                tr4w = TextRank4Keyword()
                tr4w.analyze(text=q1, lower=True, window=5, pagerank_config={'alpha': 0.85})
                q1_tokens = tr4w.get_keywords(randint(3, 7), word_min_len=2)
                q1_new = ''.join([i.word for i in q1_tokens])

                tr4w.analyze(text=q2, lower=True, window=5, pagerank_config={'alpha': 0.85})
                q2_tokens = tr4w.get_keywords(randint(3, 7), word_min_len=2)
                q2_new = ''.join([i.word for i in q2_tokens])

            i1, m1, _ = self._text_to_id(q1_new, max_seq_length, tokenizer)
            i2, m2, _ = self._text_to_id(q2_new, max_seq_length, tokenizer)
            input1.append(i1)
            mask1.append(m1)
            input2.append(i2)
            mask2.append(m2)

            self.next_idx += 1

        return input1, mask1, input2, mask2, label

    # def create_web_infer_data(self, infer_data, tokenizer):
    #     input1 = []
    #     mask1 = []
    #
    #     ids, mask, _ = self._text_to_id(infer_data, max_seq_length, tokenizer)
    #     input1.append(ids)
    #     mask1.append(mask)
    #
    #     return input1, mask1

    def create_infer_data(self, infer_data, tokenizer):
        input1 = []
        mask1 = []

        ids, mask, _ = self._text_to_id(infer_data, max_seq_length, tokenizer)
        input1.append(ids)
        mask1.append(mask)

        return input1, mask1

    def create_save_n2_data(self, tokenizer, index, batch_size):
        input2 = []
        mask2 = []

        for d in self.data[index * batch_size: (index + 1) * batch_size]:
            inp2, m2, _ = self._text_to_id(d[1], max_seq_length, tokenizer)
            input2.append(inp2)
            mask2.append(m2)

        return input2, mask2


    @staticmethod
    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def _text_to_id(self, text_a, max_seq_length, tokenizer, text_b=None):
        text_a = tokenization.convert_to_unicode(text_a)
        token1 = tokenizer.tokenize(text_a)

        token2 = None
        if text_b:
            text_b = tokenization.convert_to_unicode(text_b)
            token2 = tokenizer.tokenize(text_b)

            self._truncate_seq_pair(token1, token2, max_seq_length - 3)
        else:
            token1 = token1[:max_seq_length - 2]

        # format input data
        tokens = ['[CLS]'] + token1 + ['[SEP]']
        segment_ids = [0] * len(tokens)

        if token2:
            tokens += token2 + ['[SEP]']
            segment_ids += [1] * (len(token2) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # padding
        padding_length = max_seq_length - len(input_ids)
        padding_value = [0] * padding_length
        input_ids += padding_value
        input_mask += padding_value
        segment_ids += padding_value

        return input_ids, input_mask, segment_ids


def main():
    model_config = ModelConfig.from_json_file(model_config_file)

    if max_seq_length > model_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the model "
            "was only trained up to sequence length %d" %
            (max_seq_length, model_config.max_position_embeddings))

    tf.gfile.MakeDirs(output_dir)
    tf.gfile.MakeDirs(infer_output_dir)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)

    if train:
        processor = DataProcessor(file_dir=file_dir,
                                  is_training=True)

        num_train_steps = 100000

        with tf.Graph().as_default() as global_graph:
            model = QAModel(model_config, num_labels=2, batch_size=batch_size,
                            num_train_steps=num_train_steps, is_training=True)

            config = tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))
            with tf.Session(graph=global_graph, config=config) as sess:
                model.restore_model(sess)
                # model.restore_ckpt_global_step(init_checkpoint, False)

                # tf.train.write_graph(sess.graph_def, 'result/', 'qa.pbtxt')

                tq = tqdm(range(1, num_train_steps + 1))
                for step in tq:
                    feed_val = processor.create_train_data(batch_size, tokenizer)

                    model.train(sess, feed_val, summary=not step % 100, saver=not step % 1000)

    elif infer:
        processor = DataProcessor(file_dir=file_dir, index_file=index_file)

        with tf.Graph().as_default() as global_graph:
            model = QAModel(model_config, num_labels=2, batch_size=batch_size, is_training=False)

            config = tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))
            with tf.Session(graph=global_graph, config=config) as sess:
                model.restore_model(sess)
                # tf.train.write_graph(sess.graph_def, 'result/', 'qa.pbtxt')

                with codecs.open('data/333.txt', 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                    # lines = [u'我不太明白', u'你能吗', u'我承认我很自私', u'没有会员啦']

                # while True:
                #     infer_data = raw_input('input:').decode('utf-8')
                for l in lines:
                    infer_data = l.strip()
                    feed_val = processor.create_infer_data(infer_data, tokenizer)

                    norm1 = model.infer(sess, feed_val)
                    if save_n2:
                        norm2 = []
                        b_size = 768

                        for idx in range(len(processor.data) // b_size + 1):
                            feed_val = processor.create_save_n2_data(tokenizer, idx, b_size)
                            res = sess.run(model.norm2, feed_dict={model.input2_ids: feed_val[0],
                                                                   model.input2_mask: feed_val[1]})
                            norm2.extend(res)
                        norm2 = np.array(norm2)
                        np.save(output_dir + 'n2.npy', norm2)
                    else:
                        norm2 = np.load(output_dir + 'n2.npy')

                    sim = (norm1 * norm2).sum(axis=-1)

                    index = np.argsort(sim)[::-1]
                    res = infer_data + u'    最相似问题(相似度值域[-1(最不相似),1(最相似)]): \n\n'
                    # res = ''
                    for i, idx in enumerate(index):
                        # if sim[idx] < 0:
                        #     break

                        if i > 0:
                            break

                        # res += str(i) + ': ' + processor.data[idx][1] + u', 相似度: ' \
                        #        + str(sim[idx]) + '\t' + u'原句: ' + processor.data[idx][0] + '\n'

                        res += infer_data + ': ' + processor.data[idx][1] + u', 相似度: ' \
                               + str(sim[idx]) + '\t' + u'原句: ' + processor.data[idx][0]
                    print res




if __name__ == "__main__":
    main()
