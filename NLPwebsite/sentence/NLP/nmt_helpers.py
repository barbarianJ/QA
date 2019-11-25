import codecs
import numpy as np
import config as cf


def nmt_get_string_by_index(rnn_output, tgt_datas_rev_vocab):
    if (tgt_datas_rev_vocab == None):
        return ''

    output_len = len(rnn_output)
    res_str = ''
    for i in range(output_len):
        out = rnn_output[i]
        for j in range(len(out)):
            t = tgt_datas_rev_vocab[out[j]]
            if (t != u'</s>'):
                res_str += t
                res_str += u' '
        if (res_str.endswith(u' ')):
            res_str = res_str[:-1]
        res_str += u'\n'
    if (res_str.endswith(u'\n')):
        res_str = res_str[:-1]
    return res_str


def get_cls_by_index(cls_output, cls_datas_vocab):

    cls_datas_ver_vocab = {v: k for k, v in cls_datas_vocab.items()}

    key = cls_datas_ver_vocab[int(cls_output)]

    return key


def nmt_convert_to_ids_by_single(sentence, src_datas_vocab):
    if (sentence == None or sentence == ''):
        return None, None
    seq = []
    seq_len = []
    sentence = sentence.encode('utf-8')
    if (False):  # sentence.isalpha()
        word = sentence
        d_id = 0
        if (src_datas_vocab.has_key(word)):
            d_id = src_datas_vocab[word]
        seq.append(d_id)
        seq += [cf.NMT_EOS_ID]
        seq_len.append(len(seq))
    else:
        sentence = sentence.decode('utf-8')
        sentence = sentence[:cf.SRC_MAX_LENGTH]
        for word in sentence:
            d_id = 0
            if (src_datas_vocab.has_key(word)):
                d_id = src_datas_vocab[word]
            seq.append(d_id)
        seq += [cf.NMT_EOS_ID]
        seq_len.append(len(seq))
    seq = np.array(seq, dtype=np.int32)
    seq_len = np.array(seq_len, dtype=np.int32)
    seq = seq.reshape(1, -1)
    return seq, seq_len


def init_nmt_vocab(src_vocab_file, tgt_vocab_file, cls_vocab_file):
    # init the vocab from vocab_corpus
    src_datas_vocab = {}
    tgt_datas_vocab = {}
    tgt_datas_rev_vocab = {}
    cls_datas_vocab = {}

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

    if (cls_vocab_file != ''):
        with codecs.open(cls_vocab_file, 'r', encoding='utf-8') as cls_f:
            cls_vocab_lines = cls_f.readlines()
            cls_temp_vocab = {}
            for line in cls_vocab_lines:
                line = line.strip()
                if (line.endswith(u'\n')):
                    line = line[:-1]
                cls_temp_vocab[line] = len(cls_temp_vocab)
            cls_datas_vocab = cls_temp_vocab
            del cls_temp_vocab

    return src_datas_vocab, tgt_datas_vocab, tgt_datas_rev_vocab, cls_datas_vocab


def nmt_res_prosess(nmt_res):
    nmt_res = nmt_res.replace('$ ', '$')
    nmt_res = nmt_res.replace('#', 'this')
    return nmt_res


def softmax(word_vec):

    word_vec = word_vec.astype(np.float64)
    exp = np.exp(word_vec)
    word_sum = np.sum(exp)
    word_softmax = exp / word_sum
    max_prob = np.max(word_softmax)

    return max_prob


def get_preplexity(logits, time_major=True):

    if(time_major):
        logits = logits.reshape(
            logits.shape[1], logits.shape[0], logits.shape[2])

    shape = logits.shape
    batch_preplexity = []
    batch_prob = []
    for i in range(shape[0]):
        per_sentence = logits[i, :]
        sentence_softmax = []
        per_sen_prob = []
        for j in range(shape[1]):
            per_word = logits[i, j, :]
            word_softmax = softmax(per_word)
            sentence_softmax.append(word_softmax)

        sentence_softmax = np.array(sentence_softmax)

        sentence_prob = np.mean(sentence_softmax)
        sentence_preplexity = -np.log(sentence_prob) * 100
        batch_preplexity.append(sentence_preplexity)
        batch_prob.append(sentence_prob)

    return batch_preplexity, batch_prob
