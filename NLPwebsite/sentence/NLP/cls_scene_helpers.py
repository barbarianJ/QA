# -*- coding: utf-8 -*-
import numpy as np
import config as cf
import web_mode_config as wf

'''
scene cls data process relate code define here
get_batch_data is the last function
'''


def read_vocabulary(voc_dir):
    voc = dict()
    lines = open(voc_dir, 'r').readlines()
    for i in range(len(lines)):
        key = lines[i].decode('utf-8').split('\n')[0]
        voc[key] = i
    return voc


def sentence2matrix(sentences, max_length, vocs):
    sentences_num = len(sentences)
    data_dict = np.zeros((sentences_num, max_length), dtype='int32')

    for index, sentence in enumerate(sentences):
        data_dict[index, :] = map2id(sentence, vocs, max_length)

    return data_dict


def map2id(sentence, voc, max_len):
    array_int = np.zeros((max_len,), dtype='int32')
    min_range = min(max_len, len(sentence))

    for i in range(min_range):
        item = sentence[i]
        array_int[i] = voc.get(item, voc['<unk>'])

    return array_int


def read_data_from_strs(lines, max_sentence_length):
    data_line = []

    for line in lines:
        line = line.decode('utf-8')
        line = ''.join([word + ' ' for word in line])
        line = line.strip().lower()
        line = line.split(' ')

        if len(line) > max_sentence_length:
            line = line[:max_sentence_length]
        else:
            line.extend(['<pad>'] * (max_sentence_length - len(line)))

        data_line.append(line)
    return data_line


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1

    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def get_batch_data(data_item, vocab, max_len):
    BATCH_SIZE = 16  # set default batch size
    data = []
    if type(data_item) == str or type(data_item) == unicode:
        data.append(data_item)
    elif type(data_item) == list:
        data = data + data_item
    else:
        raise 'scene input data wrong,list or str'

    data = read_data_from_strs(data, max_len)
    x_test = sentence2matrix(data, max_len, vocab)
    batchs = batch_iter(x_test, BATCH_SIZE, 1, shuffle=False)
    return batchs


def translate_readable_logit(logit, num, Data_tup_tv_predict):
    predict_index = logit.index(max(logit))
    predict = Data_tup_tv_predict[predict_index][2]
    logit_tuplist = []
    logit_label_list = []
    for i in range(len(Data_tup_tv_predict)):
        logit_label_list.append(Data_tup_tv_predict[i][2])
    for i in range(len(logit)):
        logit_tuplist.append((logit_label_list[i], float('%.2f' % logit[i])))
    logit_tuplist.sort(key=lambda x: x[1], reverse=True)
    high_logits = logit_tuplist[:num]
    top1_score = high_logits[0][0]
    top1_prob = high_logits[0][1]
    top2_score = high_logits[1][0]
    top2_prob = high_logits[1][1]
    res_data = None
    res_list = []
    for item in high_logits:
        res_list.append({item[0]: item[1]})
    res_data = res_list
    if len(res_list) == 1:
        res_data = res_list[0]
    return res_data, top1_score, top1_prob, top2_score, top2_prob

def get_result(list, number):
    data_list=[]
    res_list = list[:number]
    for item in res_list:
        data_list.append({item[0]: item[1]})
    cls_res = data_list
    return cls_res

def get_tuple_list_four(prob1,prob2,first_son_top1,first_son_prob1,first_son_top2,first_son_prob2,second_son_top1,second_son_prob1,second_son_top2,second_son_prob2):
    first_son_prob1 = float('%.2f' % (first_son_prob1 * prob1))
    first_son_prob2 = float('%.2f' % (first_son_prob2 * prob1))
    second_son_prob1 = float('%.2f' % (second_son_prob1 * prob2))
    second_son_prob2 = float('%.2f' % (second_son_prob2 * prob2))
    tuple_list = [(first_son_top1, first_son_prob1), (first_son_top2, first_son_prob2),
                  (second_son_top1, second_son_prob1), (second_son_top2, second_son_prob2)]
    tuple_list.sort(key=lambda x: x[1],reverse=True)
    return tuple_list

def get_tuple_list_three(prob1,top2,prob2,son_top1,son_prob1,son_top2, son_prob2):
    son_prob1 = float('%.2f' % (son_prob1 * prob1))
    son_prob2 = float('%.2f' % (son_prob2 * prob1))
    tuple_list = [(top2, prob2), (son_top1, son_prob1), (son_top2, son_prob2)]
    tuple_list.sort(key=lambda x: x[1], reverse=True)
    return tuple_list

# def get_tuple_list_three(prob2,top1,prob1,son_top1,son_prob1,son_top2, son_prob2):
#     son_prob1 = float('%.2f' % (son_prob1 * prob2))
#     son_prob2 = float('%.2f' % (son_prob2 * prob2))
#     tuple_list = [(top1, prob1), (son_top1, son_prob1), (son_top2, son_prob2)]
#     tuple_list.sort(key=lambda x: x[1], reverse=True)