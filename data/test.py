# encoding=utf-8

from textrank4zh import TextRank4Keyword
import codecs
from random import choice


with codecs.open('issues.data', 'r', encoding='utf-8') as f:
    lines = f.readlines()

tr4w = TextRank4Keyword()


while 0:
    text = choice(lines).split(' #### ')[0]
    print(text)
    print
    tr4w.analyze(text=text,lower=True, window=3, pagerank_config={'alpha':0.85})
    for token in tr4w.get_keywords(8, word_min_len=2):
        print token.word

    print ('***************')


# expense, income
import numpy as np
money = np.array([
    [10.01, 102.15],        # 3, 话费
    [2765.01, 8573.11],     # 4, 300 郭有才 充值吃饭，370+345=715 机票， 591.12 代购， 230.40 上下九， 117.5 高铁 南京-杭州。， 亲情号给她： 1302.08 - 232 = 1070.08
    [6915.93, 14519.74],    # 5， 给她 3293 - 1500（后来还给我） = 1793， 又给450， 1391 自己买手机
    [450.46, 17660.24],     # 6， 亲情号 238.03， 法考报名122， 夜市228， 共同生活563.66
    [437.29, 14726.15],     # 7， 给她转 300 + 50 + 50 + 5 = 405， 吃饭727.08， 生活 554.69
    [10269.91, 15011.47],   # 8， 给她还花呗 379，回家 501.5+135.5
    [937.06, 14141.92],     # 9， 给她还花呗 823.64，
    [641.11, 12228.23]      # 10，
])

# 总共花 29379


# 总共亲情号 6508.01 - 1500 = 5008.01
# 5月， 5000 - 3000（后来房租） = 2000 给 马
# 8月 3000 给家里，
# 9月， 360认证学历




# 房租总共10500， 手机1391， 家里3000， 家里2000， 代购591.12 =   17482.12

# 29379 - 17482.12 = 11896.88

# 360 学历认证， 300 郭有才吃饭， 230.4 上下九   =  890.4

# 11896.88 - 890.4 = 11006.48

# 南京往返 1000， 买鞋 500
# 11006.48 - 1500 = 9506.48
