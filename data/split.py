import codecs


generated_data = []
with codecs.open('new_issues.data', 'r', encoding='utf-8') as f:
    for line in f:
        generated = line.strip().split(' && ')[1:]
        generated_data.append(' && ' + ' && '.join(generated) + '\n')

with codecs.open('generated_issues.data', 'w', encoding='utf-8') as f:
    f.write(''.join(generated_data))
