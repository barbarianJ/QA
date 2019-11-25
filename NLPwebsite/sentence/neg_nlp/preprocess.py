#coding=utf-8
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import logging
logger = logging.getLogger('save')
from pre_helper import loop_KMP,get_repeat_arr

class NLUPreprocess(object):
    def __init__(self,media_vocab_data):
        self.media_vocab = self._load_vocab(media_vocab_data)
    def _load_vocab(self,vocab_dir):
        with open(vocab_dir,'r') as voc_r:
            lines = voc_r.read().decode('utf-8').strip().split('\n')
            lines = sorted(lines,key = lambda i:len(i),reverse=True)#from max_len to min_len
            return lines
    def replace_name(self, sentence, symbol):
        if len(self.media_vocab) == 0:
            return sentence,[]
        dict_list = []
        for substr in self.media_vocab:
            positions = loop_KMP(sentence,substr)
            if len(positions) > 0:
                for position in positions:
                    dict_list.append(substr)
                    sentence = sentence.replace(substr,symbol,1)
        return sentence,dict_list
    def sen_filter(self,data_input):
        #replace ssp data here
        replace_dict = []
        media_label = '#'
        data_input,replace_dict = self.replace_name(data_input,media_label)
        #delete repeat data
        repeat_list = get_repeat_arr(data_input)
        if len(repeat_list) == 0:
            return data_input,replace_dict
        min_data = repeat_list[0]
        min_data_len = len(min_data)
        data_len = len(data_input)
        list_len = len(repeat_list)
        if list_len > 0 and (len(min_data) > 1 or (min_data in [media_label])):
            if len(set(repeat_list)) == 1 and list_len * min_data_len == data_len:
                data_input = repeat_list[0]
        return data_input,replace_dict

class CLSPreprocess(object):
    """docstring for CLSName"""
    def __init__(self,vocab_dir):
        self.vocab = self._load_vocab(vocab_dir)
    def _load_vocab(self, voc_dir):

        voc = dict()
        symbol = ''
        for parent, dirnames, filenames in os.walk(voc_dir):
            for filename in filenames:
                file_path = os.path.join(parent, filename)
                lines = open(file_path, 'r').readlines()
                if filename == 'app.dic':
                    symbol = 'app'
                elif filename == 'music.dic':
                    symbol = '歌曲'
                elif filename == 'video.dic':
                    symbol = '电影'
                elif filename == 'tv_control.dic':
                    symbol = '界面'
                elif filename == 'channel.dic':
                    symbol = '频道'
                for i in range(len(lines)):
                    key = lines[i].decode('utf-8').split('\n')[0]
                    voc[key] = symbol
        logger.info('cls preprocess vocabulary len : %f' % len(voc.keys()))
        return voc

    def _replace_name(self, sentence, window_size):
        global BUFFER
        BUFFER = []
        sentence_len = len(sentence.decode('utf8'))
        if sentence_len < window_size:
            window_size = sentence_len

        start_index = window_size
        sen = sentence.decode('utf8')[-start_index:]

        t = self._bmm_cut(sen,self.vocab)
        end_index = t
        while end_index < sentence_len:
            start_index =start_index + t
            sen = sentence.decode('utf8')[-start_index:-end_index]
            t = self._bmm_cut(sen,self.vocab)
            end_index = end_index + t
        BUFFER.reverse()
        return ''.join(BUFFER)

    def _bmm_cut(self,sen, vocab):
        flag = True
        global BUFFER
        while flag is True:
            # if sen in vocab:
            #     BUFFER.append(symbol)
            #     flag = False
            #     return len(sen)
            if sen in vocab.keys():
                BUFFER.append(vocab.get(sen))
                flag = False
                return len(sen)
            elif len(sen) == 1:
                BUFFER.append(sen)
                flag = False
                return 1
            else:
                sen = sen[1:]
if __name__ == '__main__':
    print loop_KMP('abc','abc')