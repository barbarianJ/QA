# encoding=utf-8

from modeling import BertConfig as ModelConfig, BertModel as Model

import codecs
# from hp import *
import tensorflow as tf
import tokenization

import numpy as np


####### hp #########

model_config_file = 'base_model/albert_config_base.json'
max_seq_length = 30
output_dir = 'result/'

vocab_file = 'result/vocab.txt'
do_lower_case = True
file_dir = 'data/issues.data'
generated_file_dir = 'data/generated_issues.data'

#####################


class QAModel(object):

    def __init__(self, model_config, is_training=True):
        self.input1_ids = tf.placeholder(shape=(None, max_seq_length), dtype=tf.int32, name='input1_ids')
        self.input1_mask = tf.placeholder(shape=(None, max_seq_length), dtype=tf.int32, name='input1_mask')

        self.input2_ids = tf.placeholder(shape=(None, max_seq_length), dtype=tf.int32, name='input2_ids')
        self.input2_mask = tf.placeholder(shape=(None, max_seq_length), dtype=tf.int32, name='input2_mask')

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

        output1 = model1.get_sequence_output()
        output2 = model2.get_sequence_output()

        self._build_graph_sequence_cosine(output1, output2)

        self.saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=3)

    def _build_graph_sequence_cosine(self, output1, output2):
        hidden_size = output1.shape[-1].value * output1.shape[-2].value

        output1 = tf.reshape(output1, (-1, hidden_size))
        output2 = tf.reshape(output2, (-1, hidden_size))

        with tf.variable_scope('output_post_process'):
            output_weights = tf.get_variable(
                "output_weights", [256, hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02))

            output_bias = tf.get_variable(
                "output_bias", [256], initializer=tf.zeros_initializer())

        output1 = tf.matmul(output1, output_weights, transpose_b=True)
        output1 = tf.nn.bias_add(output1, output_bias)
        output2 = tf.matmul(output2, output_weights, transpose_b=True)
        output2 = tf.nn.bias_add(output2, output_bias)

        self.norm1 = tf.nn.l2_normalize(output1, -1, name='norm1')
        self.norm2 = tf.nn.l2_normalize(output2, -1, name='norm2')
        self.similarity = tf.reduce_sum(tf.multiply(self.norm1, self.norm2), axis=-1, name='similarity')

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


class DataProcessor(object):

    def __init__(self, file_dir, generated_file_dir=None, separator=u' #### ', is_training=False):
        self.next_idx = 0
        self.data = []
        self.file_dir = file_dir
        self.generated_file_dir = generated_file_dir
        self.separator = separator

        if is_training and not generated_file_dir:
            raise ValueError('"generated_file_dir" must be set when training')

        generated = []
        if generated_file_dir:
            with codecs.open(generated_file_dir, 'r', encoding='utf-8') as f:
                for line in f:
                    generated.append(line.rstrip().split(' && ')[1:])

        with codecs.open(file_dir, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                try:
                    orig = line.strip()
                    desc, cause, solution = orig.split(separator)
                    if generated:
                        # training format
                        self.data.append([desc] + generated[idx])
                    else:
                        # infer format
                        self.data.append([desc, cause, solution])

                except ValueError:
                    print(u'badly formatted line: ' + line)

    def create_infer_data(self, infer_data, tokenizer):
        input1 = []
        mask1 = []

        ids, mask, _ = self._text_to_id(infer_data, max_seq_length, tokenizer)
        input1.append(ids)
        mask1.append(mask)

        return input1, mask1

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

    # tf.gfile.MakeDirs(output_dir)
    # tf.gfile.MakeDirs(infer_output_dir)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)

    processor = DataProcessor(file_dir=file_dir)

    with tf.Graph().as_default() as global_graph:
        model = QAModel(model_config, is_training=False)

        config = tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))
        with tf.Session(graph=global_graph, config=config) as sess:
            model.restore_model(sess)
            # tf.train.write_graph(sess.graph_def, 'result/', 'qa.pbtxt')

            while True:
                infer_data = raw_input('input:').decode('utf-8')
                feed_val = processor.create_infer_data(infer_data, tokenizer)

                norm1 = model.infer(sess, feed_val)

                norm2 = np.load(output_dir + 'n2.npy')

                sim = (norm1 * norm2).sum(axis=-1)

                index = np.argsort(sim)[::-1]
                res = infer_data + u'    最相似问题(相似度值域[-1(最不相似),1(最相似)]): \n\n'
                for i, idx in enumerate(index):
                    if sim[idx] < 0:
                        break
                    res += str(i) + ': ' + processor.data[idx][0] + u', 相似度: ' \
                           + str(sim[idx]) + '\n'
                print res


if __name__ == "__main__":
    main()
