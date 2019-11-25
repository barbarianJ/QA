import codecs
import re

origin = []
sep = ' && '

with codecs.open('../data/issues.data', 'r', encoding='utf-8') as f:
    for line in f:
        origin.append(line.strip())


with codecs.open('issues.data', 'w', encoding='utf-8') as f1:
    with codecs.open('1-100.data', 'r', encoding='utf-8') as f2:
        for idx, line in enumerate(f2):
            l = line.strip().split('##')
            f1.write(origin[idx] + sep.join(l) + '\n')

    with codecs.open('101-200.data', 'r', encoding='utf-8') as f2:
        cur_idx = 100

        for line in f2:

            l = line.strip().split(' ## ')

            while not origin[cur_idx].startswith(l[0][:5]):
                f1.write(origin[cur_idx] + '\n')
                cur_idx += 1

            f1.write(origin[cur_idx] + sep.join(l) + '\n')
            cur_idx += 1

    with codecs.open('201-300.txt', 'r', encoding='utf-8') as f2:
        cur_sents = []
        cur_idx = 200
        for idx, line in enumerate(f2):
            l = line.strip().split(' #### ')

            if l[-1] == origin[cur_idx].split(' #### ')[-1] and not idx == 17 \
                    and not idx == 89 and not idx == 165 and not idx == 169:
                cur_sents.append(l[0])
            else:
                f1.write(origin[cur_idx] + sep.join(cur_sents) + '\n')
                cur_sents = [l[0]]
                cur_idx += 1
        f1.write(origin[cur_idx] + sep.join(cur_sents) + '\n')
        cur_idx += 1

    with codecs.open('301-400.txt', 'r', encoding='utf-8') as f2:
        cur_sents = []
        cur_idx = 300
        f2.readline()

        for line in f2:

            if re.match('^[34][0-9]{2}:$', line):
                f1.write(origin[cur_idx] + sep.join(cur_sents) + '\n')
                cur_sents = []
                cur_idx += 1
            else:
                cur_sents.append(line.strip())
        f1.write(origin[cur_idx] + sep.join(cur_sents) + '\n')

    with codecs.open('401-500.data', 'r', encoding='utf-8') as f2:
        cur_idx = 400

        for line in f2:
            l = line.split(' #### ')[0].split('$$$$')

            f1.write(origin[cur_idx] + sep.join(l) + '\n')
            cur_idx += 1

    with codecs.open('501-634.data', 'r', encoding='utf-8') as f2:
        cur_idx = 500

        for idx, line in enumerate(f2):
            l = line.strip().split(' && ')

            f1.write(origin[cur_idx] + sep.join(l) + '\n')
            cur_idx += 1
