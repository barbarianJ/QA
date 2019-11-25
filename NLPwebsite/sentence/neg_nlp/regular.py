#coding=utf-8
import neg_config as conf
from num_recog import ch_num_en_num_recognize
from pre_helper import get_head_ec_repeat_array

def head_nlu_res(parse_word,regular_nmt_dict,regular_ner_dict):
	if ch_num_en_num_recognize(parse_word)!='NULL':
		ner_res = [{'NUM': parse_word}]
		nmt_data = conf.NUM
		ec_res = '0'
		cls_pre = '1'
		nmt_res = regular_nmt_dict[nmt_data]
		ppl_r = 0.1231355
		ppl = 0.45131654646
		res = {'ner': ner_res, 'nmt': nmt_res, 'cls': cls_pre,
		'ppl': ppl, 'ppl_r': ppl_r,
		'ec': ec_res, 'ec_result': nmt_data}
	elif parse_word in regular_ner_dict:
		ner_ = regular_ner_dict[parse_word]
		ner_res = ner_[:-1]
		nmt_data = ner_[-1]['nmt_data']
		nmt_res = regular_nmt_dict[nmt_data]
		ec_res = '0'
		cls_pre = '1'
		ppl_r = 0.1231355
		ppl = 0.45131654646
		res = {'ner': ner_res, 'nmt': nmt_res, 'cls': cls_pre,
		'ppl': ppl, 'ppl_r': ppl_r,
		'ec': ec_res, 'ec_result': nmt_data}
	else:
		res = parse_word
	return res
def head_scene_res(parse_word,scene_regular_dict):
	if ch_num_en_num_recognize(parse_word) != 'NULL':
		parse_word = conf.NUM
	if parse_word in scene_regular_dict:
		cls_res = scene_regular_dict[parse_word]
		res = {'scene': cls_res}
	else:
		res = parse_word
	return res
#Inner preprocess
def ec_head_preprocess(parse_word,ec_del_regular_dict):
	for del_item in ec_del_regular_dict:
		if del_item in parse_word:
			parse_word = parse_word.replace(del_item,'')
	return parse_word
def ec_back_process(nmt_data,ec_repeat_dict):
	nmt_data = get_head_ec_repeat_array(nmt_data)
	nmt_data = get_head_ec_repeat_array(nmt_data)
	for item in ec_repeat_dict.items():
		if item[0] in nmt_data:
			nmt_data = nmt_data.replace(item[0],item[1])
	return nmt_data
def ec_relation_del(nmt_data,ec_relation_dict):
	for re_item in ec_relation_dict:
		if nmt_data.startswith(re_item):
			nmt_data = nmt_data.replace(re_item,'',1)
	return nmt_data

def nmt_preprocess(nmt_data,ner):
	def update_ner(ner,entity_i):
		ner_key,ner_value = ner[entity_i].items()[0]
		if 'NUM' in ner_key:
			new_key = ner_key.replace('NUM','NAME')
			ner[entity_i] = {new_key:ner_value}
		return ner
	if u'\u80a1\u7968\u1405' in nmt_data or u'\u57fa\u91d1\u1405' in nmt_data:
		'''
		股票ᐅ,基金ᐅ
		'''
		entity_i = 0
		for index,i_data in enumerate(nmt_data):
			if i_data == conf.NUM or i_data == conf.NAME:
				if i_data == conf.NUM:
					replace_str = nmt_data[index-2:index] + conf.NUM
					if nmt_data[index-2:index] == u'\u80a1\u7968':
						nmt_data = nmt_data.replace(replace_str ,u'\u80a1\u7968\u1401',1)
						ner = update_ner(ner,entity_i)
					elif nmt_data[index-2:index] == u'\u57fa\u91d1':
						nmt_data = nmt_data.replace(replace_str,u'\u57fa\u91d1\u1401',1)
						ner = update_ner(ner,entity_i)
				entity_i = entity_i + 1
		return nmt_data,ner
	elif nmt_data.endswith(u'\u7535\u5f71\u1405'):
		'''
		电影ᐅ
		'''
		movie_list = list(nmt_data)
		movie_list.reverse()
		n_nmt = ''.join(movie_list).replace(u'\u1405\u5f71\u7535',u'\u1401\u5f71\u7535',1)
		new_nmt = list(n_nmt)
		new_nmt.reverse()
		new_nmt = ''.join(new_nmt)
		ner = update_ner(ner,len(ner)-1)
		return new_nmt,ner
	else:
		return nmt_data,ner