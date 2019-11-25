# coding=utf-8
import os
from NLP.ner_helpers import mergeMainObj, recovery_ner
from NLP.ec_helpers import merge_ner_ec
from NLP.nmt_helpers import nmt_res_prosess

from neg_nlp.regular import nmt_preprocess
import NLP.config as cf
from NLP.nlp_web_module import QASeverModel


class NLUModel(object):
    def __init__(self):
        '''
        serveral model loading
        '''
        model_path = os.getcwd() + '/sentence/NLP/models/'
        self.qa_func = QASeverModel(model_path)

    def special_code_replace(self, parse_word):
        # special word replace
        special_codes = []
        for item in parse_word:
            if item in cf.SPECAL_LIST:
                special_codes.append(item)
                parse_word = parse_word.replace(item, cf.NAME, 1)
        return special_codes, parse_word

    def ec_and_ner_process(self,
                           ec_preprocess,
                           parse_word,
                           replace_dict,
                           task_type=''):
        ec_res, ner_res, recover_ec, obj_location = '0', None, None, []
        special_codes, parse_word = self.special_code_replace(parse_word)
        main_ner, nmt_data_origin, ssp_nmt = self.ner_func(parse_word, special_codes)
        ner_res = main_ner
        obj_location = self.ner_obj_func(nmt_data_origin)
        if len(obj_location) > 0:
            ner_res = mergeMainObj(nmt_data_origin, main_ner, obj_location)
        if len(replace_dict) > 0:
            ner_res = recovery_ner(ner_res, replace_dict)
        nmt_data = ec_preprocess.wakeup_call(nmt_data_origin)
        nmt_data = ec_preprocess.back_call(nmt_data)
        if task_type != 'relationset':
            nmt_data = ec_preprocess.relation_call(nmt_data)
        if nmt_data != nmt_data_origin:
            ec_res = '1'
        recover_ec = merge_ner_ec(ner_res, nmt_data, ssp_nmt)
        nmt_data, ner_res = nmt_preprocess(nmt_data, ner_res)
        data_res = nmt_data, {'ner': ner_res, 'ec': ec_res, 'ec_result': recover_ec}
        return data_res

    def nmt_process(self, nmt_data, task_type=''):
        nmt_res, cls_pre, ppl, ppl_r = None, None, None, None
        if nmt_data == '':
            return data_res
        if task_type == 'tvnmt':
            res, ppl, ppl_r, cls_pre = self.defalut_nmt_func(nmt_data)
            return {'nmt': res, 'cls': cls_pre}
        elif task_type == 'homenmt':
            res, ppl, ppl_r, cls_pre = self.home_nmt_func(nmt_data)
            return {'nmt': res, 'cls': cls_pre}
        elif task_type == 'relationnmt':
            res, ppl, ppl_r, cls_pre = self.relations_nmt_func(nmt_data)
            return {'nmt': res, 'cls': cls_pre}
        elif task_type == 'controlset':
            nmt_res, ppl, ppl_r, cls_pre = self.home_nmt_func(nmt_data)
        elif task_type == 'relationset':
            nmt_res, ppl, ppl_r, cls_pre = self.relations_nmt_func(nmt_data)
        elif task_type == 'tvset':
            nmt_res, ppl, ppl_r, cls_pre = self.defalut_nmt_func(nmt_data)
        else:
            raise Exception('set type not match please check')
        nmt_res = nmt_res_prosess(nmt_res)
        nmt_res = {'nmt': nmt_res, 'cls': cls_pre, 'ppl': ppl, 'ppl_r': ppl_r}
        return nmt_res

    def chat_process(self, parse_word):
        return self.chat_func(parse_word)

    def qa_process(self, question):
        return self.qa_func(question)