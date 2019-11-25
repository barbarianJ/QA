import config as cf
import re

def merge_ner_ec(ner,nmt_data,ssp_nmt):
    recover_res = ''
    if len(ner) == 0:
        return nmt_data
    '''
    del special code entity cf.NAME = u'\u1401'
    '''
    ner_partern = re.compile(ur'^\u1401\d+',re.I|re.M)
    m_ner = []
    for i_ner in ner:
        if len(ner_partern.findall(i_ner.values()[0])) == 1:
            continue
        else:
            m_ner.append(i_ner)
    '''
    find out special code,normal placeholder set as None
    '''
    ssp_lists = []
    for item_ssp in ssp_nmt:
        if item_ssp in cf.SPECAL_LIST:
            ssp_lists.append(item_ssp)
        elif item_ssp == cf.NAME or item_ssp == cf.NUM:
            ssp_lists.append(None)
    '''
    get new recover nmt data
    '''
    ssp_i = 0
    new_nmt = []
    for index,i_nmt in enumerate(nmt_data):
        if i_nmt == cf.NAME or i_nmt == cf.NUM:
            if ssp_lists[ssp_i] != None:
                new_nmt.append(ssp_lists[ssp_i])
            else:
                new_nmt.append(i_nmt)
            ssp_i = ssp_i + 1   
        else:
            new_nmt.append(i_nmt)
    recover_res = ''.join(new_nmt)
    '''
    recover to ec origin data
    '''
    new_seq = []
    index = 0
    for item in recover_res:
        if item == cf.NAME or item == cf.NUM:
            ner_value = m_ner[index].values()[0]
            new_seq.append(ner_value)
            index = index + 1
        else:
            new_seq.append(item)
    return ''.join(new_seq)