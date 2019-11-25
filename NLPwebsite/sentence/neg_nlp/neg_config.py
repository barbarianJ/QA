# coding=utf-8
# tag name
NAME = u'\u1401'
NUM = u'\u1405'
TI = u'\u1402'
TV_CH = u'\u1573'
APP = u'\u1436'
LOC = u'\u146d'
PER = u'\u144c'
ORG = u'\u144d'
MEDIA = u'\u1465'
#special code
HOMECODE=u'\u15d1'
RELATIONCODE = u'\uffe1'
#re clean condition
CH = ur"[^\u4d15\u4e00-\u9fff"
EN = ur"A-Za-z"
NUM_DEL = ur"0-9"
SP = HOMECODE+RELATIONCODE+NAME+NUM
COM = "\._\/\-+%\*]"
RE_DEL = CH + EN + NUM_DEL + SP + COM
#time replace code
HOUR = u'\u70b9' #点 
MIN = u'\u5206' #分
SEC = u'\u79d2' #秒
# Chinese Number
ch_nums = {'零'.decode('utf-8'): '0','一'.decode('utf-8'): '1',
    '二'.decode('utf-8'): '2','两'.decode('utf-8'): '2',
    '三'.decode('utf-8'): '3','四'.decode('utf-8'): '4',
    '五'.decode('utf-8'): '5','六'.decode('utf-8'): '6',
    '七'.decode('utf-8'): '7','八'.decode('utf-8'): '8','九'.decode('utf-8'): '9'
}
ch_num_unit = {'十'.decode('utf-8'): '10','廿'.decode('utf-8'): '20',
    '百'.decode('utf-8'): '100','千'.decode('utf-8'): '1000',
    '万'.decode('utf-8'): '10000','亿'.decode('utf-8'): '100000000'}
decimal_tag = '点'.decode('utf-8')
ten_thous = '万'.decode('utf-8')
point_bilion = '亿'.decode('utf-8')
#ec head reject word
ec_head_reject = [u'\u59d0', # 姐
                u'\u7237', # 爷
                u'\u5a46', # 婆
                u'\u5976', # 奶
                u'\u7238', # 爸
                u'\u5988', # 妈
                u'\u59e5', # 姥
                u'\u54e5', # 哥
                u'\u5f1f', # 弟
                u'\u59b9', # 妹
                u'\u53d4', # 叔
                u'\u4f2f', # 伯
                u'\u6d17', # 洗
                u'\u6d17\u8863', # 洗衣 
                u'\u51c0\u6c34', # 净水
                u'\u70ed', # 热
                u'\u70ed\u6c34', # 热水
                u'\u7535\u89c6', # 电视
                u'\u6cb9\u70df', # 油烟
                u'\u7a7a\u6c14', # 空气
                u'\u51c0\u5316', # 净化
                u'\u63a2\u6d4b', # 探测
                u'\u611f\u5e94', # 感应
                u'\u5f00\u7a97',# 开窗
                ]
reject_dict = [u'\u5f00\u5173',#开关
               u'\u5f71\u96c6',#影集
               u'\u98ce\u96e8\u4f20\u611f\u5668' #风雨传感器
                ]