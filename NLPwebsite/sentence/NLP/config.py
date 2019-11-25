# coding=utf-8

KEYWORD_URL = "http://10.120.105.203:8810/_search?query='{}'"

QA_FROZEN_MODEL = 'frozen_qa.pb'

QA_INPUT1_NODE = 'input1_ids:0'
QA_MASK1_NODE = 'input1_mask:0'
QA_OUTOUT_NODE = 'norm1:0'

QA_DO_LOWER_CASE = True
QA_VOCAB_FILE = 'vocab.txt'
QA_FILE_DIR = 'issues.data'