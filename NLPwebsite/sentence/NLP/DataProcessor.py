# encoding=utf-8

import codecs
import tokenization


max_seq_length = 30


class DataProcessor(object):

    def __init__(self, file_dir, generated_file_dir=None, separator=u' #### ', is_training=False):
        self.next_idx = 0
        self.data = []
        self.file_dir = file_dir
        self.generated_file_dir = generated_file_dir
        self.separator = separator

        if is_training and not generated_file_dir:
            raise ValueError('"generated_file_dir" must be set when training')

        generated = []
        if generated_file_dir:
            with codecs.open(generated_file_dir, 'r', encoding='utf-8') as f:
                for line in f:
                    generated.append(line.rstrip().split(' && ')[1:])

        with codecs.open(file_dir, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                try:
                    orig = line.strip()
                    desc, cause, solution = orig.split(separator)
                    if generated:
                        # training format
                        self.data.append([desc] + generated[idx])
                    else:
                        # infer format
                        self.data.append([desc, cause, solution])

                except ValueError:
                    print(u'badly formatted line: ' + line)



    def create_infer_data(self, infer_data, tokenizer):
        input1 = []
        mask1 = []

        ids, mask, _ = self._text_to_id(infer_data, max_seq_length, tokenizer)
        input1.append(ids)
        mask1.append(mask)

        return input1, mask1

    def create_save_n2_data(self, tokenizer):
        input2 = []
        mask2 = []

        for d in self.data:
            inp2, m2, _ = self._text_to_id(d[0], max_seq_length, tokenizer)
            input2.append(inp2)
            mask2.append(m2)

        return input2, mask2


    @staticmethod
    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def _text_to_id(self, text_a, max_seq_length, tokenizer, text_b=None):
        text_a = tokenization.convert_to_unicode(text_a)
        token1 = tokenizer.tokenize(text_a)

        token2 = None
        if text_b:
            text_b = tokenization.convert_to_unicode(text_b)
            token2 = tokenizer.tokenize(text_b)

            self._truncate_seq_pair(token1, token2, max_seq_length - 3)
        else:
            token1 = token1[:max_seq_length - 2]

        # format input data
        tokens = ['[CLS]'] + token1 + ['[SEP]']
        segment_ids = [0] * len(tokens)

        if token2:
            tokens += token2 + ['[SEP]']
            segment_ids += [1] * (len(token2) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # padding
        padding_length = max_seq_length - len(input_ids)
        padding_value = [0] * padding_length
        input_ids += padding_value
        input_mask += padding_value
        segment_ids += padding_value

        return input_ids, input_mask, segment_ids
