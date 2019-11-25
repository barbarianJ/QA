import codecs
from random import randint
import jieba

f_name = 'data/issues.data'
generated_f_name = 'data/generated_issues.data'

with codecs.open(f_name, 'r', encoding='utf-8') as f:
    data = f.readlines()

with codecs.open(generated_f_name, 'r', encoding='utf-8') as f:
    generated_data = f.readlines()

if len(generated_data) < len(data):
    generated_data += [' && '] * (len(data) - len(generated_data))

while 1:
    length = len(data)
    index = randint(0, length)
    # q = data[index].strip().split(' && ')[0].strip().split(' #### ')[0]
    q = data[index].strip().split(' #### ')[0]
    tokens = list(jieba.cut(q))
    print '\n\n' + q + '\n'

    display = ''
    for idx, t in enumerate(tokens):
        display += str(idx) + ': ' + t + ', '
        if idx and idx % 10 == 0:
            display += '\n'
    print display
    picked_index = raw_input('pick the number to represent this question, separate by space, '
                             '"n" to skip to next question, '
                             '"q" to save\n:')
    if picked_index == 'q':
        break
    elif picked_index == 'n':
        continue

    picked_index = picked_index.split()
    picked_tokens = ''.join([tokens[int(i)] for i in picked_index])
    print picked_tokens

    confirm = raw_input('confirm? (y/n): ')
    if confirm == 'y':
        generated_data[index] = generated_data[index].rstrip() + ' && ' + picked_tokens + '\n'
    else:
        continue

confirm = raw_input('save the new data? (y/n):')
if confirm == 'y':
    with codecs.open(generated_f_name, 'w', encoding='utf-8') as f:
        f.write(''.join(generated_data))
