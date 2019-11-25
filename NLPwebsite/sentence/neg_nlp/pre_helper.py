# -*- encoding: utf-8 -*-
import json
import copy
from collections import OrderedDict
from num_recog import ch_num_en_num_recognize
import neg_config as cf

def _load_vocab(vocab_dir):
    with open(vocab_dir,'r') as voc_r:
        lines = voc_r.read().decode('utf-8').strip().split('\n')
        lines = sorted(lines,key = lambda i:len(i),reverse=True)#from max_len to min_len
        return lines
def load_json_dict(dict_path):
    with open(dict_path, 'r') as dict_read:
        josn_dict = dict_read.read().strip()
        data_dict = json.loads(josn_dict)
        res_dict = OrderedDict()
        for key in sorted(data_dict.keys(),key = lambda i:len(i),reverse=True):
            res_dict[key] = data_dict[key]
        return res_dict
# kmp algorithm
def get_next(str_data):
    str_data = str_data + '='
    i=0
    j=-1
    next_array=[1]*len(str_data)
    next_array[0] = -1
    while(str_data[i] != '='):
        if(j == -1 or str_data[i] == str_data[j]):
            i = i + 1
            j = j + 1
            if(str_data[i]!=str_data[j]):
                next_array[i]=j;
            else:
                next_array[i]=next_array[j]
        else:
            j = next_array[j]
    return next_array[:len(str_data)-1]
def KMP(tgtstr,substr,pos=0):
    sub_next = get_next(substr)
    i = pos
    j = 0
    while(i < len(tgtstr) and j < len(substr)):
        if(j == -1 or tgtstr[i] == substr[j]):
            i = i + 1
            j = j + 1
        else:
            j = sub_next[j]
 
    if(j == len(substr)):
        return i-j
    else:
        return -1
def loop_KMP(tgtstr,substr,pos=0):
    tgtstrtmp = tgtstr
    postions = []
    fore_len = 0
    while len(tgtstrtmp) > 0:
        res = KMP(tgtstrtmp,substr,pos)
        if res != -1:
            postions.append(res + fore_len)
            tgtstrtmp = tgtstrtmp[res + len(substr):]
            fore_len = len(tgtstr) - len(tgtstrtmp)
        else:
            break
    return postions
def check_number_str(str_data):
    if ch_num_en_num_recognize(str_data) != 'NULL':
        return True
    else:
        return False
def get_repeat_arr(str_data):
    split_arrays = []
    head_sen = []
    real_next_array = []
    repeat_part = get_next(str_data)
    if repeat_part.count(-1) >=2:
        if repeat_part[0] == repeat_part[1]:#reapeat substr.eg:aaabaaab
            start_pos = 0
            if len(list(set(repeat_part))) == 1: #check same num,eg:190000000
                return []
            for index,item in enumerate(repeat_part):
                if (item >= 0) and (index + 1 < len(repeat_part)):
                    if repeat_part[index + 1] < 0:
                        str_ = str_data[start_pos : index + 1]
                        if not check_number_str(str_):
                            split_arrays.append(str_)
                        start_pos = index + 1
                elif index + 1 >= len(repeat_part) and item >= 0:#reach the end of sen
                    str_ = str_data[start_pos : index + 1]
                    if not check_number_str(str_):
                        split_arrays.append(str_)
                else:
                    pass
        else:
            i = 0
            chars = []
            while i < len(repeat_part):
                if repeat_part[i] == -1:
                    chars.append(str_data[i])
                    j = i + 1
                    while j < len(repeat_part) and repeat_part[j] >=0:
                        chars.append(str_data[j])
                        j = j + 1
                    i = j
                    str_out = ''.join(chars)
                    if not check_number_str(str_out):
                        split_arrays.append(str_out)
                    chars = []
        split_arrays = sorted(split_arrays,key=lambda t: len(t),reverse=False)
        return split_arrays
    else:
        return []

def get_head_ec_repeat_array(str_data):
    if len(str_data)<=2:
        return str_data
    for item in cf.reject_dict:
        if item in str_data:
            return str_data
    split_arrays = []
    origin_arrays = []
    repeat_part = get_next(str_data)
    if repeat_part.count(-1) >=2:
        i = 0
        chars = []
        while i < len(repeat_part):
            if repeat_part[i] == -1:
                chars.append(str_data[i])
                j = i + 1
                while j < len(repeat_part) and repeat_part[j] >=0:
                    chars.append(str_data[j])
                    j = j + 1
                i = j
                split_arrays.append(''.join(chars))
                chars = []
        origin_arrays = copy.deepcopy(split_arrays)
        split_arrays = sorted(split_arrays,key=lambda t: len(t),reverse=False)
        rejectfun = lambda a : (cf.NAME not in a and cf.NUM not in a)
        min_str = split_arrays[0] if rejectfun(split_arrays[0]) else None
        if min_str == None or (min_str in cf.ec_head_reject and len(split_arrays)>1):
            return str_data
        recover_data = []
        for item in split_arrays:
            if item == min_str:
                #reject min_str data
                recover_data.append(item)
                continue
            elif item != min_str and min_str in item and rejectfun(item):
                #update repeat array and recover array
                for i in range(len(split_arrays)):
                    if min_str in split_arrays:
                        split_arrays[split_arrays.index(min_str)] = None
                    if min_str in recover_data:
                        recover_data[recover_data.index(min_str)] = None
                min_str = item
                recover_data.append(item)
            elif min_str in item and not rejectfun(item):
                #update repeat array and recover array
                for i in range(len(split_arrays)):
                    if min_str in split_arrays:
                        split_arrays[split_arrays.index(min_str)] = None
                    if min_str in recover_data:
                        recover_data[recover_data.index(min_str)] = None
                recover_data.append(item)
            else:
               recover_data.append(item)
        if len(recover_data) == 0:
            return min_str
        else:
            out_dict = {}
            out_data = []
            for item in recover_data:
                if item is not None:
                    out_dict[origin_arrays.index(item)] = item
            for data in sorted(out_dict.keys(),key = lambda i:i,reverse=True):
                out_data.append(out_dict[data])
            return ''.join(out_data)
    else:
        return str_data

if __name__ == '__main__':
    pass