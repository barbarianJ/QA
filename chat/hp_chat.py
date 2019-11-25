model_config_file = 'base_model/albert_config_base.json'
max_seq_length = 30
output_dir = 'result_chat/'
train = True
infer = False
train = False
infer = True

save_n2 = False
# save_n2 = True
                                                
vocab_file = 'base_model/vocab.txt'
do_lower_case = True
file_dir = 'data/combined_chat.txt'
index_file = 'data/corpora.data'

infer_output_dir = 'infer/'

init_checkpoint = 'result/ckpt-'
init_checkpoint = 'base_model/albert_model.ckpt'

batch_size = 64
num_epoch = 1000
learning_rate = 0.00005
num_warmup_proportion = 0.1

infer_start_index = 0
num_sent_to_compare = 100000

infer_lower_bound = -0.05
infer_upper_bound = 0.05
