#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import config as cf
import numpy as np
import codecs
import re
import collections
import copy
import logging
logger_error = logging.getLogger('wrong')
'''
extract name entity of one sentence
'''
def extractNameEntity(label,sentence):
    name_content = []
    name_key = ''
    namelist = []
    entity_locs = []
    entity_indx = 0
    for index,keylabel in enumerate(label):
        if index >= len(sentence):
            break
        if sentence[index] == '_':
            sentence[index] = ' '
        if keylabel!='O':
            keylabel = keylabel.split('_')
            name_key = keylabel[1]
            if keylabel[0] == 'S':
                entity_indx = index
                name_content.append(sentence[index])
                key_content = ''.join(name_content)
                namedict = {name_key:key_content}
                namelist.append(namedict)
                entity_locs.append(entity_indx)
                name_content = []
                continue
            if keylabel[0] == 'B':
                name_content = []
                entity_indx = index
                name_content.append(sentence[index])
            if keylabel[0]== 'I':
                name_content.append(sentence[index])
            if keylabel[0] == 'E':
                name_content.append(sentence[index])
                key_content = ''.join(name_content)
                namedict = {name_key:key_content}
                namelist.append(namedict)
                entity_locs.append(entity_indx)
                name_content = []
    return namelist,entity_locs

def transform2NMTstyle(special_codes,ner_result,ner_locs,sentences):
    sentence = ''.join(sentences[0])
    '''
    finish extract entity,recover special for find the right obj location
    '''
    try:
        if cf.NAME in sentence and len(special_codes) > 0:
            ssp_i = 0
            for item in sentence:
                if item == cf.NAME:
                    sentence = sentence.replace(item,special_codes[ssp_i],1)
                    ssp_i = ssp_i + 1
    except Exception as e:
        logger_error.info(e)
    '''
    get sentence for nmt model
    '''
    sentence_cls = copy.deepcopy(sentence)
    sort_data = []
    for data in ner_result:
        sort_data.append(data.items()[0])
    for index,data in enumerate(sort_data):
        key,value = data
        vlen = len(value)
        pos = 0
        while pos >= 0:
            pos = sentence.find(value,pos)
            '''
            reach the end pos
            '''
            if pos < 0:
                break
            '''
            find the next pos,need subtract the last entity len and add 1
            '''
            current_pos = ner_locs[index]
            if index > 0:
                for i in range(index):
                    current_pos = current_pos - len(sort_data[i][1]) + 1
            '''
            replace entity here
            '''
            if pos == current_pos:
                head_seq = sentence[:pos]
                tail_seq = sentence[pos + vlen:]
                head = sentence_cls[:pos]
                tail = sentence_cls[pos + vlen:]
                sentence = head_seq + cf.trans_tag_dict[key] + tail_seq
                sentence_cls = head + cf.trans_tag_dict_scene[key] + tail
                break
            else:
                pos = pos + 1
    '''
    get nmt train data and updata special location
    '''
    re_indexs = []
    new_ner = [None]*20
    if len(special_codes) > 0:
        origin_sentence = sentence
        #update ner entity
        entity_i = 0
        new_entity_i = 0
        special_code_i = 0
        for i_seq in origin_sentence:
            if i_seq == cf.NAME or i_seq == cf.NUM:
                new_ner[new_entity_i] = ner_result[entity_i]
                entity_i = entity_i + 1
                new_entity_i = new_entity_i + 1
            if i_seq in cf.SPECAL_LIST:
                new_ner[new_entity_i] = {'NAME':cf.NAME+str(special_code_i+1)}
                new_entity_i = new_entity_i + 1
                special_code_i = special_code_i + 1
                sentence = sentence.replace(i_seq,cf.NAME,1)
        new_ner = [x for x in new_ner if x != None]
    else:
        new_ner = ner_result
        origin_sentence = sentence
    return sentence,origin_sentence,new_ner

def ner_return(ner_list):
    ner_res = collections.OrderedDict()
    for item in ner_list:
        key,value = item.items()[0]
        if ner_res.has_key(value):
            continue
        else:
            ner_res[value] = key
    return ner_res

def recovery_ner(ner_res,replace_list,replace_label='#'):
    replace_index = 0
    for index,ner_item in enumerate(ner_res):
        ner_key = ner_item.keys()[0]
        ner_item_content = ner_item[ner_key]
        if replace_label in ner_item_content:
            ner_item_content = ner_item_content.replace(replace_label,replace_list[replace_index])
            ner_res[index][ner_key] = ner_item_content
            replace_index = replace_index + 1
    return ner_res
'''
split ner seq data to list
'''
def split_sentences(lines,input_len):
    for index,line in enumerate(lines):
        line = line.decode('utf-8')
        line = [word for word in line]
        line = line[:input_len - 1]
        lines[index] = line
    return lines

def ner_convert_to_ids_by_single(sentence,src_datas_vocab,input_len):
    if (sentence == None or sentence == ''):
        return None
    seq = []
    seq_len = []
    sentence = sentence[:input_len - 1]
    for word in sentence:
        d_id = cf.NER_UNK_ID
        if (src_datas_vocab.has_key(word)):
            d_id = src_datas_vocab[word]
        seq.append(d_id)
    seq += [cf.NER_EOS_ID]
    if len(seq) < input_len:
        seq += [cf.NER_PADDING]*(input_len - len(seq))
    seq = np.array(seq, dtype=np.int32)
    seq = seq.reshape(1, -1)
    return seq

def get_batch_string_by_index(rnn_output,tgt_datas_rev_vocab):
    data_out = []
    if (tgt_datas_rev_vocab == None):
        return data_out
    output_len = len(rnn_output)
    for i in range(output_len):
        out = rnn_output[i]
        res_str_list = []
        out = out[:-1]
        for j in range(len(out)):
            t = tgt_datas_rev_vocab[out[j]]
            if (t != u'<eos>'):
                if t == u'<unk>' or t == u'<padding>':
                    t = 'O'
                res_str_list.append(t)
            else:
                break
        data_out.append(res_str_list)
    return data_out

def init_ner_vocab(src_vocab_file, tgt_vocab_file):
    # init the vocab from vocab_corpus
    src_datas_vocab = {}
    tgt_datas_vocab = {}
    tgt_datas_rev_vocab = {}

    if (src_vocab_file != ''):
        with codecs.open(src_vocab_file, 'r', encoding='utf-8') as src_f:
            src_vocab_lines = src_f.readlines()
            src_temp_vocab = {}
            for line in src_vocab_lines:
                line = line.strip()
                if (line.endswith(u'\n')):
                    line = line[:-1]
                src_temp_vocab[line] = len(src_temp_vocab)
            src_datas_vocab = src_temp_vocab
            del src_temp_vocab

    if(tgt_vocab_file != ''):
        with codecs.open(tgt_vocab_file, 'r', encoding='utf-8') as tgt_f:
            tgt_vocab_lines = tgt_f.readlines()
            tgt_temp_vocab = {}
            for line in tgt_vocab_lines:
                line = line.strip()
                if (line.endswith(u'\n')):
                    line = line[:-1]
                tgt_temp_vocab[line] = len(tgt_temp_vocab)
            tgt_datas_vocab = tgt_temp_vocab
            del tgt_temp_vocab

            temp_rev_vocab = {}
            for (i, j) in zip(tgt_datas_vocab.keys(), tgt_datas_vocab.values()):
                temp_rev_vocab[j] = i
            tgt_datas_rev_vocab = temp_rev_vocab
    return src_datas_vocab,tgt_datas_vocab,tgt_datas_rev_vocab

'''
find object location
'''
def findObjectLocation(labels_seq,sentence):
    index_obj = []
    i_obj = 0
    for index,item in enumerate(sentence):
        if item in [cf.NAME, cf.NUM]:
            i_obj = i_obj + 1
            if labels_seq[index] == 'O':
                pass
            elif labels_seq[index] == 'S_OBJ':
                index_obj.append(i_obj-1)
    return index_obj

def mergeMainObj(nmt_data,main_ner,obj_location):
    if len(obj_location) == 0:
        return main_ner
    placeholder_i = 0
    entity_i = 0
    for index,data_item in enumerate(nmt_data):
        if data_item == cf.NAME or data_item == cf.NUM:
            for i_location in obj_location:
                if placeholder_i == i_location:#find object position
                    obj_entity = main_ner[entity_i]
                    key,value = obj_entity.items()[0]
                    new_item = {key + '_obj':value}
                    main_ner[entity_i] = new_item
            #entity index increase
            entity_i = entity_i + 1
            #placeholder in sentence increase
            placeholder_i = placeholder_i + 1
    return main_ner
'''
use ner result relpace nmt $label
'''
def get_name_nmt_labels(nmt_res):
    nmt_res_temp = copy.deepcopy(nmt_res)
    names_label_list = cf.Name_labels
    mutil_names = []
    #contact mutil words
    for item in cf.mutil_words:
        if item in nmt_res_temp:
            item_sub = item.replace(' ','-')
            mutil_names.append(item_sub)
            nmt_res_temp = nmt_res_temp.replace(item,item_sub)
    #delete ( )
    if len(mutil_names) > 0:
        nmt_res_list = nmt_res_temp.split(' ')
        for index,nmt_item in enumerate(nmt_res_list):
            if nmt_item in mutil_names:
                if index -1 > 0 and index + 1 < len(nmt_res_list):
                    if nmt_res_list[index -1] == '(' and nmt_res_list[index +1] == ')':
                        nmt_res_list.pop(index -1)
                        nmt_res_list.pop(index)
        nmt_res_temp = ' '.join(nmt_res_list)
    names_label_list = names_label_list + mutil_names
    return names_label_list,nmt_res_temp

def get_replace_nerdict(ner_list):
    #resort ner entity for entity replace
    ner_res = {'NAME_obj':[],'NAME':[],'NUM':[],'NUM_obj':[]}
    ner_list_temp = copy.deepcopy(ner_list)
    for item in ner_list_temp:
        key = item.keys()[0]
        if key == 'NUM_obj':
            ner_res['NUM_obj'].append(item[key])
        if key == 'NUM':
            ner_res['NUM'].append(item[key])
        if key !='NUM' and key!='NUM_obj' and '_obj' in key:
            ner_res['NAME_obj'].append(item[key])
        if key !='NUM' and key!='NUM_obj' and '_obj' not in key:
            ner_res['NAME'].append(item[key])            
    for key_item in ner_res.keys():
        if len(ner_res[key_item]) == 0:
            del ner_res[key_item]
    return ner_res
def find_pos_str(nmt_res,pos):
    pos = nmt_res.find('$',pos)
    if pos < 0:
        return 'NULL',-1
    label_pos = nmt_res.find(' ',pos)
    if label_pos < 0:
        label_str = nmt_res[pos:]
    else:
        label_str = nmt_res[pos:label_pos]
    return label_str,pos

def replace_function(nmt_res,label_list,content):
    i = 0
    pos = 0
    while (i < len(content)):
        replaced = False
        label_str,pos = find_pos_str(nmt_res,pos)
        if label_str == 'NULL' or pos < 0:
            break
        if label_str in label_list:
            nmt_res = nmt_res.replace(label_str,content[i],1)
            replaced = True
        if pos >= 0:
            pos = pos + 1
        else:
            break
        if replaced:
            i = i + 1
    return nmt_res

def response2device(nmt_res,ner_list):
    if len(ner_list) == 0:
        return nmt_res
    names_label_list,nmt_res_replace = get_name_nmt_labels(nmt_res)
    ner_res = get_replace_nerdict(ner_list)
    try:
        if 'NAME_obj' in ner_res.keys():
            nmt_res_replace = replace_function(nmt_res_replace,names_label_list,ner_res['NAME_obj'])
        if 'NUM_obj' in ner_res.keys():
            nmt_res_replace = replace_function(nmt_res_replace,cf.Num_labels,ner_res['NUM_obj'])
        if 'NAME' in ner_res.keys():
            nmt_res_replace = replace_function(nmt_res_replace,names_label_list,ner_res['NAME'])
        if 'NUM' in ner_res.keys():
            nmt_res_replace = replace_function(nmt_res_replace,cf.Num_labels,ner_res['NUM'])
        if len(ner_res.keys()) == 0:
            nmt_res_replace = replace_function(nmt_res_replace,names_label_list,cf.NAME)
    except Exception, e:
        return 'NULL'
    return nmt_res_replace
if __name__ == '__main__':
    pass
