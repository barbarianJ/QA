# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import logging
import os
import sys
from datetime import datetime

import NLP.web_mode_config as webconfig
from NLP.web_utils import nmt_cmd_replace,nmt_replace_dict
import nlu
import scene
from rule import Innerpreprocess,Headprocess_model
logger = logging.getLogger('save')
logger_save = logging.getLogger('debug_save')
reload(sys)
sys.setdefaultencoding('utf-8')

class NLPHandler():
    def __init__(self):
        # self.nmt_dict = nmt_replace_dict(webconfig.CONFIG_PATH)
        # self.head_preprocess = Headprocess_model()
        # self.ec_preprocess = Innerpreprocess()
        # self.scene_model = scene.NlpScene()
        self.nlu_model = nlu.NLUModel()
    def run_scene_models(self,data_in):
        data_in = self.head_preprocess.scene_func(data_in)
        return self.scene_model(data_in)
    def run_nlu_models(self, data_in, task_type):
        '''
        preprocess
        '''
        data_res, final,parse_word,replace_dict = {},None, None, None
        data_in = self.head_preprocess.nlu_func(data_in)
        if type(data_in) == dict:
            final = nmt_cmd_replace(data_in['nmt'],data_in['ner'],self.nmt_dict)
            data_in['version'] = webconfig.version
            data_in['final'] = final
            return data_in
        else:
            parse_word,replace_dict = data_in
        # several nlp task
        if parse_word == '':
            return data_res
        if task_type == 'ner_ec':
            nmt_data,data_res = self.nlu_model.ec_and_ner_process(
                                                self.ec_preprocess,
                                                parse_word,
                                                replace_dict,
                                                'ner_ec')
        elif 'nmt' in task_type:
            data_res = self.nlu_model.nmt_process(parse_word,
                                                task_type)
        else:
            '''
            ner model process,extact name entity and object entity
            '''
            nmt_data,ner_ec = self.nlu_model.ec_and_ner_process(
                                                self.ec_preprocess,
                                                parse_word,
                                                replace_dict,
                                                task_type)
            '''
            nmt model process
            '''
            logger_save.info(nmt_data)
            nmt_res = self.nlu_model.nmt_process(nmt_data,
                                                task_type)
            '''
            collect ner and nmt result, combine nmt and ner to a new final result
            '''
            final = nmt_cmd_replace(nmt_res['nmt'], ner_ec['ner'], self.nmt_dict)
            '''
            combine all data back
            '''
            data_res = { 'final': final, 'version': webconfig.version}
            data_res.update(ner_ec)
            data_res.update(nmt_res)
        return data_res

    def run_chat_models(self,data_in):
        data_in = self.head_preprocess.chat_func(data_in)
        data_res = self.nlu_model.chat_process(data_in)
        return data_res

    def run_qa_model(self, data_in):
        return self.nlu_model.qa_process(data_in)

    def __call__(self, data_in, task_type):
        logger.info(data_in)
        t_start = datetime.now()
        # data_in = self.head_preprocess.head_func(data_in)
        if task_type == 'scene':
            scene_res = self.run_scene_models(data_in)
            all_t_end = datetime.now()
            logger.info("scene model cost: %f seconds" %(all_t_end - t_start).total_seconds())
            return scene_res
        elif task_type == 'chat':
            chat_res = self.run_chat_models(data_in)
            all_t_end = datetime.now()
            logger.info("chat model cost: %f seconds" %(all_t_end - t_start).total_seconds())
            return chat_res
        elif task_type == 'qa':
            qa_res = self.run_qa_model(data_in)
            all_t_end = datetime.now()
            logger.info("qa model cost: %f seconds" %(all_t_end - t_start).total_seconds())
            return qa_res
        else:
            nlu_res = self.run_nlu_models(data_in, task_type)
            all_t_end = datetime.now()
            logger.info("NLU model cost: %f seconds" %(all_t_end - t_start).total_seconds())
            return nlu_res