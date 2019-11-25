import os
import NLP.web_mode_config as webconfig
from NLP.nlp_web_module import SceneCLS, SceneCLSCNN,NgtvCLSCNN
from NLP.cls_scene_helpers import get_result,get_tuple_list_three,get_tuple_list_four

class NlpScene():
    def __init__(self):
        '''
        load scene model
        '''
        if webconfig.CONFIG_MODE == webconfig.TV_MODE:
            model_path = os.getcwd() + '/sentence/NLP/models/nlu/'
            self.scenecls = SceneCLSCNN(model_path)
            self.scenecls_map = SceneCLS(model_path, 'map')
            self.scenecls_ticket = SceneCLS(model_path, 'ticket')
            self.scenecls_order = SceneCLS(model_path, 'order')
            self.scenecls_audio = SceneCLS(model_path, 'audio')
            self.scenecls_baike = SceneCLS(model_path, 'baike')
        elif webconfig.CONFIG_MODE == webconfig.NGTV_MODE:
            model_path = os.getcwd() + '/sentence/NLP/models/ngtv/'
            self.scenecls = NgtvCLSCNN(model_path)
            self.scenecls_map = SceneCLS(model_path, 'map')
            self.scenecls_ticket = SceneCLS(model_path, 'ticket')
            self.scenecls_order = SceneCLS(model_path, 'order')
            self.scenecls_audio = SceneCLS(model_path, 'audio')
            self.scenecls_baike = SceneCLS(model_path, 'baike')

    def __call__(self, parse_word):
        if type(parse_word) == dict:
            parse_word['version'] = webconfig.version
            return parse_word
        # ngtv or tv mode
        if webconfig.CONFIG_MODE == webconfig.NGTV_MODE or webconfig.CONFIG_MODE == webconfig.TV_MODE:
            son_label_list = ['map_server', 'ticket_server', 'order_server', 'audio', 'baike']

            cls_res, top1, prob1, top2, prob2 = self.scenecls(parse_word)

            if top1 in son_label_list and top2 not in son_label_list:

                if top1 == 'map_server':
                    son_cls_res, son_top1, son_prob1, son_top2, son_prob2 = self.scenecls_map(parse_word)
                    tuple_list = get_tuple_list_three(prob1, top2, prob2, son_top1, son_prob1, son_top2, son_prob2)
                    cls_res = get_result(tuple_list, 2)
                if top1 == 'ticket_server':
                    son_cls_res, son_top1, son_prob1, son_top2, son_prob2 = self.scenecls_ticket(parse_word)
                    tuple_list = get_tuple_list_three(prob1, top2, prob2, son_top1, son_prob1, son_top2, son_prob2)
                    cls_res = get_result(tuple_list, 2)
                if top1 == 'order_server':
                    son_cls_res, son_top1, son_prob1, son_top2, son_prob2 = self.scenecls_order(parse_word)
                    tuple_list = get_tuple_list_three(prob1, top2, prob2, son_top1, son_prob1, son_top2, son_prob2)
                    cls_res = get_result(tuple_list, 2)
                if top1 == 'audio':
                    son_cls_res, son_top1, son_prob1, son_top2, son_prob2 = self.scenecls_audio(parse_word)
                    tuple_list = get_tuple_list_three(prob1, top2, prob2, son_top1, son_prob1, son_top2, son_prob2)
                    cls_res = get_result(tuple_list, 2)
                if top1 == 'baike':
                    son_cls_res, son_top1, son_prob1, son_top2, son_prob2 = self.scenecls_baike(parse_word)
                    tuple_list = get_tuple_list_three(prob1, top2, prob2, son_top1, son_prob1, son_top2, son_prob2)
                    cls_res = get_result(tuple_list, 2)

            if top1 not in son_label_list and top2 in son_label_list:

                if top2 == 'map_server':
                    son_cls_res, son_top1, son_prob1, son_top2, son_prob2 = self.scenecls_map(parse_word)
                    tuple_list = get_tuple_list_three(prob2, top1, prob1, son_top1, son_prob1, son_top2, son_prob2)
                    cls_res = get_result(tuple_list, 2)
                if top2 == 'ticket_server':
                    son_cls_res, son_top1, son_prob1, son_top2, son_prob2 = self.scenecls_ticket(parse_word)
                    tuple_list = get_tuple_list_three(prob2, top1, prob1, son_top1, son_prob1, son_top2, son_prob2)
                    cls_res = get_result(tuple_list, 2)
                if top2 == 'order_server':
                    son_cls_res, son_top1, son_prob1, son_top2, son_prob2 = self.scenecls_order(parse_word)
                    tuple_list = get_tuple_list_three(prob2, top1, prob1, son_top1, son_prob1, son_top2, son_prob2)
                    cls_res = get_result(tuple_list, 2)
                if top2 == 'audio':
                    son_cls_res, son_top1, son_prob1, son_top2, son_prob2 = self.scenecls_audio(parse_word)
                    tuple_list = get_tuple_list_three(prob2, top1, prob1, son_top1, son_prob1, son_top2, son_prob2)
                    cls_res = get_result(tuple_list, 2)
                if top2 == 'baike':
                    son_cls_res, son_top1, son_prob1, son_top2, son_prob2 = self.scenecls_baike(parse_word)
                    tuple_list = get_tuple_list_three(prob2, top1, prob1, son_top1, son_prob1, son_top2, son_prob2)
                    cls_res = get_result(tuple_list, 2)

            if top1 in son_label_list and top2 in son_label_list:

                if top1 == 'map_server':

                    first_son_cls_res, first_son_top1, first_son_prob1, first_son_top2, first_son_prob2 = self.scenecls_map(
                        parse_word)

                    if top2 == 'ticket_server':
                        second_son_cls_res, second_son_top1, second_son_prob1, second_son_top2, second_son_prob2 = self.scenecls_ticket(
                            parse_word)
                        tuple_list = get_tuple_list_four(prob1, prob2, first_son_top1, first_son_prob1,
                                                         first_son_top2,
                                                         first_son_prob2, second_son_top1, second_son_prob1,
                                                         second_son_top2, second_son_prob2)
                        cls_res = get_result(tuple_list, 2)

                    if top2 == 'order_server':
                        second_son_cls_res, second_son_top1, second_son_prob1, second_son_top2, second_son_prob2 = self.scenecls_order(
                            parse_word)
                        tuple_list = get_tuple_list_four(prob1, prob2, first_son_top1, first_son_prob1,
                                                         first_son_top2,
                                                         first_son_prob2, second_son_top1, second_son_prob1,
                                                         second_son_top2, second_son_prob2)
                        cls_res = get_result(tuple_list, 2)

                    if top2 == 'audio':
                        second_son_cls_res, second_son_top1, second_son_prob1, second_son_top2, second_son_prob2 = self.scenecls_audio(
                            parse_word)
                        tuple_list = get_tuple_list_four(prob1, prob2, first_son_top1, first_son_prob1,
                                                         first_son_top2,
                                                         first_son_prob2, second_son_top1, second_son_prob1,
                                                         second_son_top2, second_son_prob2)
                        cls_res = get_result(tuple_list, 2)

                    if top2 == 'baike':
                        second_son_cls_res, second_son_top1, second_son_prob1, second_son_top2, second_son_prob2 = self.scenecls_baike(
                            parse_word)
                        tuple_list = get_tuple_list_four(prob1, prob2, first_son_top1, first_son_prob1,
                                                         first_son_top2,
                                                         first_son_prob2, second_son_top1, second_son_prob1,
                                                         second_son_top2, second_son_prob2)
                        cls_res = get_result(tuple_list, 2)

                if top1 == 'ticket_server':

                    first_son_cls_res, first_son_top1, first_son_prob1, first_son_top2, first_son_prob2 = self.scenecls_ticket(
                        parse_word)
                    if top2 == 'map_server':
                        second_son_cls_res, second_son_top1, second_son_prob1, second_son_top2, second_son_prob2 = self.scenecls_map(
                            parse_word)
                        tuple_list = get_tuple_list_four(prob1, prob2, first_son_top1, first_son_prob1,
                                                         first_son_top2,
                                                         first_son_prob2, second_son_top1, second_son_prob1,
                                                         second_son_top2, second_son_prob2)
                        cls_res = get_result(tuple_list, 2)

                    if top2 == 'order_server':
                        second_son_cls_res, second_son_top1, second_son_prob1, second_son_top2, second_son_prob2 = self.scenecls_order(
                            parse_word)
                        tuple_list = get_tuple_list_four(prob1, prob2, first_son_top1, first_son_prob1,
                                                         first_son_top2,
                                                         first_son_prob2, second_son_top1, second_son_prob1,
                                                         second_son_top2, second_son_prob2)
                        cls_res = get_result(tuple_list, 2)

                    if top2 == 'audio':
                        second_son_cls_res, second_son_top1, second_son_prob1, second_son_top2, second_son_prob2 = self.scenecls_audio(
                            parse_word)
                        tuple_list = get_tuple_list_four(prob1, prob2, first_son_top1, first_son_prob1,
                                                         first_son_top2,
                                                         first_son_prob2, second_son_top1, second_son_prob1,
                                                         second_son_top2, second_son_prob2)
                        cls_res = get_result(tuple_list, 2)

                    if top2 == 'baike':
                        second_son_cls_res, second_son_top1, second_son_prob1, second_son_top2, second_son_prob2 = self.scenecls_baike(
                            parse_word)
                        tuple_list = get_tuple_list_four(prob1, prob2, first_son_top1, first_son_prob1,
                                                         first_son_top2,
                                                         first_son_prob2, second_son_top1, second_son_prob1,
                                                         second_son_top2, second_son_prob2)
                        cls_res = get_result(tuple_list, 2)

                if top1 == 'order_server':

                    first_son_cls_res, first_son_top1, first_son_prob1, first_son_top2, first_son_prob2 = self.scenecls_order(
                        parse_word)
                    if top2 == 'map_server':
                        second_son_cls_res, second_son_top1, second_son_prob1, second_son_top2, second_son_prob2 = self.scenecls_map(
                            parse_word)
                        tuple_list = get_tuple_list_four(prob1, prob2, first_son_top1, first_son_prob1,
                                                         first_son_top2,
                                                         first_son_prob2, second_son_top1, second_son_prob1,
                                                         second_son_top2, second_son_prob2)
                        cls_res = get_result(tuple_list, 2)


                    if top2 == 'ticket_server':
                        second_son_cls_res, second_son_top1, second_son_prob1, second_son_top2, second_son_prob2 = self.scenecls_ticket(
                            parse_word)
                        tuple_list = get_tuple_list_four(prob1, prob2, first_son_top1, first_son_prob1,
                                                         first_son_top2,
                                                         first_son_prob2, second_son_top1, second_son_prob1,
                                                         second_son_top2, second_son_prob2)
                        cls_res = get_result(tuple_list, 2)

                    if top2 == 'audio':
                        second_son_cls_res, second_son_top1, second_son_prob1, second_son_top2, second_son_prob2 = self.scenecls_audio(
                            parse_word)
                        tuple_list = get_tuple_list_four(prob1, prob2, first_son_top1, first_son_prob1,
                                                         first_son_top2,
                                                         first_son_prob2, second_son_top1, second_son_prob1,
                                                         second_son_top2, second_son_prob2)
                        cls_res = get_result(tuple_list, 2)

                    if top2 == 'baike':
                        second_son_cls_res, second_son_top1, second_son_prob1, second_son_top2, second_son_prob2 = self.scenecls_baike(
                            parse_word)
                        tuple_list = get_tuple_list_four(prob1, prob2, first_son_top1, first_son_prob1,
                                                         first_son_top2,
                                                         first_son_prob2, second_son_top1, second_son_prob1,
                                                         second_son_top2, second_son_prob2)
                        cls_res = get_result(tuple_list, 2)

                if top1 == 'audio':

                    first_son_cls_res, first_son_top1, first_son_prob1, first_son_top2, first_son_prob2 = self.scenecls_audio(
                        parse_word)
                    if top2 == 'map_server':
                        second_son_cls_res, second_son_top1, second_son_prob1, second_son_top2, second_son_prob2 = self.scenecls_map(
                            parse_word)
                        tuple_list = get_tuple_list_four(prob1, prob2, first_son_top1, first_son_prob1,
                                                         first_son_top2,
                                                         first_son_prob2, second_son_top1, second_son_prob1,
                                                         second_son_top2, second_son_prob2)
                        cls_res = get_result(tuple_list, 2)


                    if top2 == 'ticket_server':
                        second_son_cls_res, second_son_top1, second_son_prob1, second_son_top2, second_son_prob2 = self.scenecls_ticket(
                            parse_word)
                        tuple_list = get_tuple_list_four(prob1, prob2, first_son_top1, first_son_prob1,
                                                         first_son_top2,
                                                         first_son_prob2, second_son_top1, second_son_prob1,
                                                         second_son_top2, second_son_prob2)
                        cls_res = get_result(tuple_list, 2)

                    if top2 == 'order_server':
                        second_son_cls_res, second_son_top1, second_son_prob1, second_son_top2, second_son_prob2 = self.scenecls_order(
                            parse_word)
                        tuple_list = get_tuple_list_four(prob1, prob2, first_son_top1, first_son_prob1,
                                                         first_son_top2,
                                                         first_son_prob2, second_son_top1, second_son_prob1,
                                                         second_son_top2, second_son_prob2)
                        cls_res = get_result(tuple_list, 2)

                    if top2 == 'baike':
                        second_son_cls_res, second_son_top1, second_son_prob1, second_son_top2, second_son_prob2 = self.scenecls_baike(
                            parse_word)
                        tuple_list = get_tuple_list_four(prob1, prob2, first_son_top1, first_son_prob1,
                                                         first_son_top2,
                                                         first_son_prob2, second_son_top1, second_son_prob1,
                                                         second_son_top2, second_son_prob2)
                        cls_res = get_result(tuple_list, 2)

                if top1 == 'baike':

                    first_son_cls_res, first_son_top1, first_son_prob1, first_son_top2, first_son_prob2 = self.scenecls_baike(
                        parse_word)

                    if top2 == 'map_server':
                        second_son_cls_res, second_son_top1, second_son_prob1, second_son_top2, second_son_prob2 = self.scenecls_map(
                            parse_word)
                        tuple_list = get_tuple_list_four(prob1, prob2, first_son_top1, first_son_prob1,
                                                         first_son_top2,
                                                         first_son_prob2, second_son_top1, second_son_prob1,
                                                         second_son_top2, second_son_prob2)
                        cls_res = get_result(tuple_list, 2)


                    if top2 == 'ticket_server':
                        second_son_cls_res, second_son_top1, second_son_prob1, second_son_top2, second_son_prob2 = self.scenecls_ticket(
                            parse_word)
                        tuple_list = get_tuple_list_four(prob1, prob2, first_son_top1, first_son_prob1,
                                                         first_son_top2,
                                                         first_son_prob2, second_son_top1, second_son_prob1,
                                                         second_son_top2, second_son_prob2)
                        cls_res = get_result(tuple_list, 2)

                    if top2 == 'order_server':
                        second_son_cls_res, second_son_top1, second_son_prob1, second_son_top2, second_son_prob2 = self.scenecls_order(
                            parse_word)
                        tuple_list = get_tuple_list_four(prob1, prob2, first_son_top1, first_son_prob1,
                                                         first_son_top2,
                                                         first_son_prob2, second_son_top1, second_son_prob1,
                                                         second_son_top2, second_son_prob2)
                        cls_res = get_result(tuple_list, 2)

                    if top2 == 'audio':
                        second_son_cls_res, second_son_top1, second_son_prob1, second_son_top2, second_son_prob2 = self.scenecls_audio(
                            parse_word)
                        tuple_list = get_tuple_list_four(prob1, prob2, first_son_top1, first_son_prob1,
                                                         first_son_top2,
                                                         first_son_prob2, second_son_top1, second_son_prob1,
                                                         second_son_top2, second_son_prob2)
                        cls_res = get_result(tuple_list, 2)

            process_res = {'scene': cls_res,
                           'version': webconfig.version}
            return process_res
        else:
            return {}