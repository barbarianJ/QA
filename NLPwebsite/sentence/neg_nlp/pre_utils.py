# -*- encoding: utf-8 -*-
import re
import neg_config as cf

def time_str_replace(sentence):
    hour = cf.HOUR
    mint = cf.MIN
    seconds = cf.SEC
    line_res = ''
    line = sentence.split(':')
    if len(line) == 3:
        line_res = line[0] + hour + line[1] + mint + line[2] + seconds
    elif len(line) == 2:
        line_res = line[0] + hour + line[1] + mint
    return line_res


def get_time_str(sentence):
    tmp_seq = sentence
    time_strs_1 = re.findall(
        r'[0-9]{0,4}:[0-9]{0,4}:[0-9]{0,4}', tmp_seq)
    for item in time_strs_1:
        tmp_seq = tmp_seq.replace(item, '')
    time_strs_2 = re.findall(r'[0-9]{0,4}:[0-9]{0,4}', tmp_seq)
    time_strs = time_strs_1 + time_strs_2
    if len(time_strs) == 0:
        return sentence
    for it_time in time_strs:
        it_time_new = time_str_replace(it_time)
        sentence = sentence.replace(it_time, it_time_new)
    return sentence


def math_op_replace(sentence):
    if '×' in sentence:
        sentence = sentence.replace('×', '*')
    if '÷' in sentence:
        sentence = sentence.replace('÷', '/')
    return sentence
'''
delete comma in sentence
'''
def delete_comma(sentence):
    sentence = sentence.strip()
    sentence = sentence.replace(" ", "_")
    sentence = math_op_replace(sentence)
    if ':' in sentence:
        sentence = get_time_str(sentence)
    sentence = re.sub(cf.RE_DEL, "", sentence)
    sentence = sentence.lower()
    return sentence


def scene_delete_comma(sentence):
    sentence = sentence.strip()
    sentence = sentence.replace(" ", "_")
    if cf.HOMECODE in sentence:
        sentence = sentence.replace(cf.HOMECODE, cf.NAME)
    sentence = sentence.lower()
    return sentence

def pre_filter(sentence, voc):
    sentence = sentence.decode('utf-8').strip()
    for key in voc.keys():
        if key in sentence:
            sentence = sentence.replace(key, voc[key])
    return sentence