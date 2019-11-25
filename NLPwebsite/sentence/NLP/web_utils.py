# -*- encoding: utf-8 -*-
from ner_helpers import response2device
'''
nmt command replace here
'''
def nmt_replace_dict(config_path):
    dict_res = {}
    with open(config_path, 'r') as nmt_r:
        nmt_cmds = nmt_r.read().strip().decode('utf-8').split('\n')
        for cmd in nmt_cmds:
            if '#' not in cmd[0]:
                cmd_t = cmd.split('/')
                dict_res[cmd_t[0]] = cmd_t[0] + '/' + cmd_t[-1]
        return dict_res


def nmt_cmd_replace(nmt, ner_res, nmt_dict):

    if nmt in nmt_dict.keys():
        nmt_cmd = nmt_dict[nmt]
        if '/' in nmt_cmd or ']/' in nmt_cmd:
            if '/' in nmt_cmd:
                spt = '/'
            if ']/' in nmt_cmd:
                spt = ']/'
            nmt_cmd_n = nmt_cmd.split(spt)
            nmt_cmd_res = response2device(nmt_cmd_n[0], ner_res)
            nmt_cmd = nmt_cmd_res + spt + nmt_cmd_n[-1]
        else:
            nmt_cmd = response2device(nmt_cmd, ner_res)
    else:
        nmt_cmd = response2device(nmt, ner_res)
    return nmt_cmd


def re_op(sentence):
    # this func is to simple sentence eg:爱情的情怎么写 --> 情怎么写

    # '*'and '+' are key word in re so replace
    sentence = sentence.replace('*', '!').replace('+', '@')
    for i in range(len(sentence)):
        pattern = sentence[i] + '(.?)的' + sentence[i]
        re_pattern = re.search(pattern, sentence)
        if (re_pattern != None):
            idx = re_pattern.span()[1]
            sen = sentence[idx - 1:]
            sen = sen.replace('!', '*').replace('@', "+")
            return sen
    sentence = sentence.replace('!', '*').replace('@', "+")
    return sentence

    pass


if __name__ == '__main__':
    print nmt_cmd_replace('+ degree', 'conf/home.command.conf').encode('utf-8')
