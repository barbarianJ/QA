import os
from neg_nlp.pre_utils import pre_filter,delete_comma,scene_delete_comma
from neg_nlp.pre_helper import load_json_dict,_load_vocab
from neg_nlp.regular import head_nlu_res,head_scene_res
from neg_nlp.preprocess import NLUPreprocess,CLSPreprocess

from neg_nlp.regular import ec_head_preprocess,ec_back_process,ec_relation_del

class Headprocess_model(object):
    def __init__(self):
        regular_path = os.getcwd() + '/sentence/neg_nlp/res_dict'
        '''
        load head data dict
        '''
        pre_path = regular_path + '/pre_data/'
        self.prefilter_voc = load_json_dict(pre_path + 'prefilter.json')
        # '''
        # load scene preprocess dict and replace module
        # '''
        # scene_reg_path = regular_path + '/scene/'
        # self.scene_regular_dict = load_json_dict(scene_reg_path + 'scene_res_dict.json')
        # self.cls_preprocess = CLSPreprocess(vocab_dir=scene_reg_path)
        # '''
        # load nlu preprocess dict and replace module
        # '''
        # nlu_regular_path = regular_path + '/nlu/'
        # self.regular_nmt_dict = load_json_dict(nlu_regular_path + 'ssp_nmt.json')
        # self.regular_ner_dict = load_json_dict(nlu_regular_path + 'ssp_ner.json')
        # # load nlu-pre
        # self.nlu_preprocess = NLUPreprocess(nlu_regular_path + 'media.voc')
    def head_func(self,data_input):
        data_ = pre_filter(data_input, self.prefilter_voc)
        return data_
    def nlu_func(self,data_input):
        data_input = delete_comma(data_input).strip()
        parse_data = head_nlu_res(data_input,self.regular_nmt_dict,self.regular_ner_dict)
        if type(parse_data) == dict:
            return parse_data
        else:
            parse_word, replace_dict = self.nlu_preprocess.sen_filter(parse_data)
            return [parse_word, replace_dict]
    def scene_func(self,data_input):
        parse_word = scene_delete_comma(data_input)
        parse_word = head_scene_res(parse_word,self.scene_regular_dict)
        if type(parse_word) == dict:
            pass
        else:
            parse_word = self.cls_preprocess._replace_name(parse_word, 10)
        return parse_word
    def chat_func(self,data_input):
        return data_input

class Innerpreprocess(object):
    def __init__(self):
        nlu_path = os.getcwd() + '/sentence/neg_nlp/res_dict/nlu/'
        self.ec_del = _load_vocab(nlu_path + 'ec_del.voc')
        self.ec_relation_del = _load_vocab(nlu_path + 'ec_pre_relathion.voc')
        self.ec_repeat_del = load_json_dict(nlu_path + 'ec_repeat.json')
    def wakeup_call(self,nmt_data):
        return ec_head_preprocess(nmt_data,self.ec_del)
    def back_call(self,nmt_data):
        return ec_back_process(nmt_data,self.ec_repeat_del)
    def relation_call(self,nmt_data):
        return ec_relation_del(nmt_data,self.ec_relation_del)