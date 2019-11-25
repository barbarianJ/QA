#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import tensorflow as tf
import logging
import numpy as np
import sys
import math
import config as cf
from datetime import datetime
import ner_helpers as ner_h
import nmt_helpers as nmt_h
import cls_scene_helpers as cls_scene_h
import cls_helpers as cls_h
import scene_config

# sys.path.append('..')
import tokenization
from DataProcessor import DataProcessor
import requests
import json


logger = logging.getLogger('save')


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


class TextClassifyWebModel(object):

    def __init__(self, path):
        self.graph = get_tensor_graph(path + cf.CF_FROZEN_MODEL)
        self.text_input = self.graph.get_tensor_by_name(
            cf.CF_INPUT_NODE_1)
        self.result = self.graph.get_tensor_by_name(
            cf.CF_OUTPUT_NODE_1)
        self.voc, _ = cls_h.read_vocabulary(path + cf.CF_VOC_PATH)
        self.sess = tf.Session(graph=self.graph)

    def __call__(self, str_data):
        cf_t_start = datetime.now()
        data_input = [str_data]
        resultdata = []
        x_data = cls_h.read_data_from_strs(data_input, 32)
        x_test = np.array(cls_h.sentence2matrix(x_data, 32, self.voc))
        get_batches = cls_h.batch_iter(
            list(x_test), 1, 1, shuffle=False)
        for x_test_batch in get_batches:
            batch_predictions = self.sess.run(
                self.result, {self.text_input: x_test_batch})
            predict_datas = batch_predictions.tolist()
            for predict_data in predict_datas:
                resultdata.append(
                    predict_data.index(max(predict_data)))
        cf_t_end = datetime.now()
        logger.info("TCT CLS cost:%f seconds" %
                    (cf_t_end - cf_t_start).total_seconds())
        return str(resultdata[0])


class SceneCLSCNN(object):

    def __init__(self, path):
        self.graph = get_tensor_graph(path + cf.SCENECNN_PATH)
        self.scenecnn_input0 = self.graph.get_tensor_by_name(
            cf.SCENECNN_INPUT_NODE0)
        self.scenecnn_input1 = self.graph.get_tensor_by_name(
            cf.SCENECNN_INPUT_NODE1)
        self.resultcnn = self.graph.get_tensor_by_name(
            cf.SCENECNN_OUTPUT_NODE)
        self.voc = cls_scene_h.read_vocabulary(
            path + cf.SCENECNN_VOCAB_PATH)
        self.sess = tf.Session(graph=self.graph)

    def __call__(self, scene_data):
        name_t_start = datetime.now()
        batches = cls_scene_h.get_batch_data(
            scene_data, self.voc, cf.SCENECNN_MAX_LENGTH)
        res = []
        for x_test_batch in batches:
            batch_predictions = self.sess.run(self.resultcnn, {
                self.scenecnn_input0: x_test_batch, self.scenecnn_input1: 1.0})
            res.extend(batch_predictions.tolist())
        res_f = []
        res_top1 = []
        prob_top1 = []
        res_top2 = []
        prob_top2 = []
        for item in res:
            top2_result, top1_label, top1_prob, top2_label, top2_prob = cls_scene_h.translate_readable_logit(
                item, 2, scene_config.Data_tup_scene)
            logger.info("level 1 top2_result is :%s" % (top2_result))
            # print("top1_label is :%s"%(top1_label))
            # print("top1_prob is :%s" % (top1_prob))
            # print type(top1_prob)
            # print("top2_label is :%s"%(top2_label))
            # print("top2_prob is :%s" % (top2_prob))
            # print type(top2_prob)
            res_f.append(top2_result)
            res_top1.append(top1_label)
            prob_top1.append(top1_prob)
            res_top2.append(top2_label)
            prob_top2.append(top2_prob)
        name_t_end = datetime.now()
        logger.info("SceneCLSCNN cost: %f seconds" %
                    (name_t_end - name_t_start).total_seconds())
        return res_f[0], res_top1[0], prob_top1[0], res_top2[0], prob_top2[0]


class SceneCLS(object):

    def __init__(self, path, flag):
        if flag == 'map':
            model = cf.map_SCENE_PATH
            vocab = cf.map_SCENE_VOCAB_PATH
            self.config_list = scene_config.Data_tup_map
        elif flag == 'ticket':
            model = cf.ticket_SCENE_PATH
            vocab = cf.ticket_SCENE_VOCAB_PATH
            self.config_list = scene_config.Data_tup_ticket
        elif flag == 'order':
            model = cf.order_SCENE_PATH
            vocab = cf.order_SCENE_VOCAB_PATH
            self.config_list = scene_config.Data_tup_order
        elif flag == 'audio':
            model = cf.audio_SCENE_PATH
            vocab = cf.audio_SCENE_VOCAB_PATH
            self.config_list = scene_config.Data_tup_audio
        elif flag == 'baike':
            model = cf.baike_SCENE_PATH
            vocab = cf.baike_SCENE_VOCAB_PATH
            self.config_list = scene_config.Data_tup_baike

        self.graph = get_tensor_graph(path + model)
        self.scene_input = self.graph.get_tensor_by_name(
            cf.SCENE_INPUT_NODE)
        self.result = self.graph.get_tensor_by_name(
            cf.SCENE_OUTPUT_NODE)
        self.voc = cls_scene_h.read_vocabulary(path + vocab)
        self.sess = tf.Session(graph=self.graph)

    def __call__(self, scene_data):
        name_t_start = datetime.now()
        batches = cls_scene_h.get_batch_data(
            scene_data, self.voc, cf.SCENE_MAX_LENGTH)
        res = []
        for x_test_batch in batches:
            batch_predictions = self.sess.run(
                self.result, {self.scene_input: x_test_batch})
            res.extend(batch_predictions.tolist())
        res_f = []
        res_top1 = []
        prob_top1 = []
        res_top2 = []
        prob_top2 = []
        for item in res:
            top2_result, top1_label, top1_prob, top2_label, top2_prob = cls_scene_h.translate_readable_logit(
                item, 2, self.config_list)
            logger.info("level 2 top2_result is :%s" % (top2_result))
            # print("top1_label is :%s" % (top1_label))
            # print("top2_label is :%s" % (top2_label))
            res_f.append(top2_result)
            res_top1.append(top1_label)
            prob_top1.append(top1_prob)
            res_top2.append(top2_label)
            prob_top2.append(top2_prob)
        name_t_end = datetime.now()
        logger.info("SceneCLS cost: %f seconds" %
                    (name_t_end - name_t_start).total_seconds())
        return res_f[0], res_top1[0], prob_top1[0], res_top2[0], prob_top2[0]


class NgtvCLSCNN(object):

    def __init__(self, path):
        self.graph = get_tensor_graph(path + cf.SCENECNN_PATH)
        self.scenecnn_input0 = self.graph.get_tensor_by_name(
            cf.SCENECNN_INPUT_NODE0)
        self.scenecnn_input1 = self.graph.get_tensor_by_name(
            cf.SCENECNN_INPUT_NODE1)
        self.resultcnn = self.graph.get_tensor_by_name(
            cf.SCENECNN_OUTPUT_NODE)
        self.voc = cls_scene_h.read_vocabulary(
            path + cf.SCENECNN_VOCAB_PATH)
        self.sess = tf.Session(graph=self.graph)

    def __call__(self, scene_data):
        name_t_start = datetime.now()
        batches = cls_scene_h.get_batch_data(
            scene_data, self.voc, cf.SCENECNN_MAX_LENGTH)
        res = []
        for x_test_batch in batches:
            batch_predictions = self.sess.run(self.resultcnn, {
                self.scenecnn_input0: x_test_batch, self.scenecnn_input1: 1.0})
            res.extend(batch_predictions.tolist())
        res_f = []
        res_top1 = []
        prob_top1 = []
        res_top2 = []
        prob_top2 = []
        for item in res:
            top2_result, top1_label, top1_prob, top2_label, top2_prob = cls_scene_h.translate_readable_logit(
                item, 2, scene_config.Data_tup_ngtv)
            print("level 1 top2_result is :%s" % (top2_result))
            # print("top1_label is :%s"%(top1_label))
            # print("top1_prob is :%s" % (top1_prob))
            # print type(top1_prob)
            # print("top2_label is :%s"%(top2_label))
            # print("top2_prob is :%s" % (top2_prob))
            # print type(top2_prob)
            res_f.append(top2_result)
            res_top1.append(top1_label)
            prob_top1.append(top1_prob)
            res_top2.append(top2_label)
            prob_top2.append(top2_prob)
        name_t_end = datetime.now()
        logger.info("NgtvCLSCNN cost: %f seconds" %
                    (name_t_end - name_t_start).total_seconds())
        return res_f[0], res_top1[0], prob_top1[0], res_top2[0], prob_top2[0]


class NERServerModel(object):

    def __init__(self, path, flag):
        # load ner model data
        self.flag = flag
        if flag == 'main':
            self.NER_FROZEN_MODEL = 'ner/main/ner.pb'
            self.NER_VOC_DATA = 'ner/main/vocab.na.data'
            self.NER_VOC_LABEL = 'ner/main/vocab.lf.data'
        elif flag == 'obj':
            self.NER_FROZEN_MODEL = 'ner/obj/ner_obj.pb'
            self.NER_VOC_DATA = 'ner/obj/vocab.na.data'
            self.NER_VOC_LABEL = 'ner/obj/vocab.lf.data'

        self.graph = get_tensor_graph(path + self.NER_FROZEN_MODEL)
        self.sess = tf.Session(graph=self.graph)
        self.x1 = self.sess.graph.get_tensor_by_name(
            cf.NER_INPUT_NODE_1)
        self.y = self.sess.graph.get_tensor_by_name(
            cf.NER_OUTPUT_NODE_1)
        self.src_datas_vocab, self.tgt_datas_vocab, self.tgt_datas_rev_vocab = \
            ner_h.init_ner_vocab(path + self.NER_VOC_DATA,
                                 path + self.NER_VOC_LABEL)

    def __call__(self, sentence, special_codes=None):
        t_start = datetime.now()
        input_len = cf.OBJNER_INPUT_LEN
        if self.flag == 'main':
            input_len = cf.NER_INPUT_LEN
        sen = ner_h.ner_convert_to_ids_by_single(
            sentence, self.src_datas_vocab, input_len)
        res_arr = self.sess.run(self.y, feed_dict={self.x1: sen})
        res = ner_h.get_batch_string_by_index(
            res_arr, self.tgt_datas_rev_vocab)
        seq = ner_h.split_sentences([sentence], input_len)
        if self.flag == 'main':
            ner_info, ner_locs = ner_h.extractNameEntity(
                res[0], seq[0])
            nmt_sentence, ssp_sentence, ner_info = ner_h.transform2NMTstyle(special_codes,
                                                                            ner_info, ner_locs, seq)
            t_end = datetime.now()
            logger.info("NER cost: %f seconds" %
                        (t_end - t_start).total_seconds())
            return ner_info, nmt_sentence, ssp_sentence
        else:
            ner_obj_location = ner_h.findObjectLocation(
                res[0], seq[0])
            t_end = datetime.now()
            logger.info("NER_OBJ cost: %f seconds" %
                        (t_end - t_start).total_seconds())
            return ner_obj_location


class NMTServerModel(object):

    def __init__(self, path):
        self.home = False
        self.tv = False
        self.dialogue = False
        if ('home' in path):
            self.home = True
        elif ('dialogue' in path):
            self.dialogue = True
        else:
            self.tv = True
        self.graph = get_tensor_graph(path + cf.NMT_FROZEN_MODEL)
        self.sess = tf.Session(graph=self.graph)
        self.x1 = self.sess.graph.get_tensor_by_name(
            cf.NMT_INPUT_NODE_1)
        self.x2 = self.sess.graph.get_tensor_by_name(
            cf.NMT_INPUT_NODE_2)
        self.y = self.sess.graph.get_tensor_by_name(
            cf.NMT_OUTPUT_NODE_1)
        self.logit = self.sess.graph.get_tensor_by_name(
            'RNN/decoder/model_logits:0')
        self.cls_predict = self.sess.graph.get_tensor_by_name(
            'CLS/cls_predict:0')
        self.src_datas_vocab, self.tgt_datas_vocab, self.tgt_datas_rev_vocab, self.cls_datas_vocab = \
            nmt_h.init_nmt_vocab(path + cf.DATA_VOCAB_FILE,
                                 path + cf.LABEL_VOCAB_FILE,
                                 path + cf.CLS_VOCAB_FILE)

    def __call__(self, sentence):
        nmt_t_start = datetime.now()
        sen, slen = nmt_h.nmt_convert_to_ids_by_single(
            sentence, self.src_datas_vocab)
        res_arr, cls_predict, logit = self.sess.run(
            [self.y, self.cls_predict, self.logit], feed_dict={self.x1: sen, self.x2: slen})
        res = nmt_h.nmt_get_string_by_index(
            res_arr, self.tgt_datas_rev_vocab)
        cls_p = nmt_h.get_cls_by_index(
            cls_predict, self.cls_datas_vocab)

        if (cls_p == u'unknow'):
            cls_p = 0
        else:
            cls_p = 1

        preplexity, _ = nmt_h.get_preplexity(logit)
        ppl = preplexity[0]
        if (self.home):
            ppl_relative = preplexity[0] / 4.295370732252
        elif (self.dialogue):
            ppl_relative = preplexity[0] / 3.701109112154
        else:
            ppl_relative = preplexity[0] / 4.249011685590
        nmt_t_end = datetime.now()
        logger.info("NMT cost: %f seconds" %
                    (nmt_t_end - nmt_t_start).total_seconds())

        return res, ppl, ppl_relative, cls_p




class QASeverModel(object):

    def __init__(self, path):
        self.graph = get_tensor_graph(path + cf.QA_FROZEN_MODEL)
        self.sess = tf.Session(graph=self.graph)
        self.input1_node = self.sess.graph.get_tensor_by_name(
            cf.QA_INPUT1_NODE)
        self.mask1 = self.sess.graph.get_tensor_by_name(
            cf.QA_MASK1_NODE)
        self.output_node = self.sess.graph.get_tensor_by_name(
            cf.QA_OUTOUT_NODE)
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=path + cf.QA_VOCAB_FILE, do_lower_case=cf.QA_DO_LOWER_CASE)
        # self.processor = DataProcessor(file_dir=cf.QA_FILE_DIR)
        self.processor = DataProcessor(path + cf.QA_FILE_DIR)
        self.norm2 = np.load(path + 'n2.npy')

    def __call__(self, sentence):
        qa_t_start = datetime.now()

        # key word
        query_url = cf.KEYWORD_URL.format(sentence)
        headers = {"Content-Type": "text/plain"}
        response = requests.get(query_url, headers=headers)
        keyword_display = json.loads(response.text)
        # num_display = len(keyword_answer)

        # qa_model
        top_k = 200
        feed_val = self.processor.create_infer_data(''.join(sentence.split(';')), self.tokenizer)
        norm1 = self.sess.run(self.output_node,
                              feed_dict={
                                  self.input1_node: feed_val[0],
                                  self.mask1: feed_val[1],
                              })
        prob = (norm1*self.norm2).sum(axis=-1)

        model_display_index = np.argsort(prob)[::-1]

        # test
        # res = sentence + u'    最相似问题(相似度值域[-1(最不相似),1(最相似)]): <br><br>'
        # for i, idx in enumerate(index):
        #     if prob[idx] < 0:
        #         break
        #     res += str(i) + ': ' + self.processor.data[idx][0] + ', 相似度: ' \
        #            + str(prob[idx]) + '<br>'
        #

        # keyword_display = np.random.choice(keyword_answer, num_display / 2, replace=False)

        # model_display = []
        # for i, idx in enumerate(index):
        #     dup = False
        #     for k_temp in keyword_display:
        #         if self.processor.data[idx][0] == k_temp['phenomenon']:
        #             dup = True
        #             break
        #     if not dup:
        #         model_display.append(self.processor.data[idx])
        #         if len(model_display) >= num_display / 2:
        #             break


        '''
        # return str
        res = u'与 "' + sentence + u'" 最相似的问题 (语义模型[S]相似度值域[-1(最不相似),1(最相似)])：<br><br>'
        idx = 1
        displayed = []
        for i in range(top_k):
            if not self.processor.data[model_display_index[i]][0] in displayed:

                res += str(idx) + ' [S], 相似度：' + str(prob[model_display_index[i]]) + ':'
                res += '<pre>' + '    现象: ' + self.processor.data[model_display_index[i]][0] + '</pre>'
                res += '<pre>' + '    原因: ' + self.processor.data[model_display_index[i]][1] + '</pre>'
                res += '<pre>' + '    方案: ' + self.processor.data[model_display_index[i]][2] + '</pre>'

                idx += 1

            else:
                print self.processor.data[model_display_index[i]][0]

            if i < len(keyword_display):
                displayed.append(self.processor.data[model_display_index[i]][0])

                if not keyword_display[i]['phenomenon'] in displayed:
                    displayed.append(keyword_display[i]['phenomenon'])
                    res += str(idx) + ' [K] :'
                    res += '<pre>' + '    现象: ' + keyword_display[i]['phenomenon'] + '</pre>'
                    res += '<pre>' + '    原因: ' + keyword_display[i]['reason'] + '</pre>'
                    res += '<pre>' + '    方案: ' + keyword_display[i]['solution'] + '</pre>'

                    idx += 1
        '''

        # return json
        res = []
        displayed = []
        for i in range(top_k):
            if not self.processor.data[model_display_index[i]][0] in displayed:
                record = dict()
                record['reason'] = self.processor.data[model_display_index[i]][1]
                record['phenomenon'] = self.processor.data[model_display_index[i]][0]
                record['solution'] = self.processor.data[model_display_index[i]][2]
                record['source'] = 'S'
                record['sim'] = str(prob[model_display_index[i]])
                res.append(record)

            if i < len(keyword_display):
                displayed.append(self.processor.data[model_display_index[i]][0])

                if not keyword_display[i]['phenomenon'] in displayed:
                    record = keyword_display[i]
                    record['source'] = 'K'
                    record['sim'] = 'N/A'
                    res.append(record)

        qa_t_end = datetime.now()
        logger.info("QA cost: %f seconds" %
                    (qa_t_end - qa_t_start).total_seconds())

        # return u'back(' + str({"list": res}).decode('utf-8') + u')'
        return res


if __name__ == '__main__':
    pass
