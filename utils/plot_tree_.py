import jieba
import tqdm
from pyltp import SentenceSplitter, Postagger, NamedEntityRecognizer, Parser
import os
# import pydotplus
import numpy as np

extra_vocabulary_path = 'dict'
LTP_DATA_DIR = '/data1/ssr/EE/data/ltp_model'  # ltp模型目录的路径

# jieba.load_userdict(extra_vocabulary_path)

par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')

postagger = Postagger()
postagger.load(pos_model_path)
parser = Parser()  # 初始化实例
parser.load(par_model_path)  # 加载模型

l_now = ''
l = 'TIME，被告人吉某某与吉超、雷俊等人至本市南马镇和平村新龙伺机盗窃。由被告人吉某某和雷俊负责望风，吉超等人采用撬棒撬门入户的手段，进入该村徐金华户，窃得绿驹牌TDR091Z型电动自行车1辆，计价值人民币2400元。得逞后，被告人吉某某等人驾驶电动自行车逃离，途经本市南马镇葛府村附近时，被巡逻民警人赃俱获。现公安机关已将赃车发还给失主徐金华。'
TIME = '2012年8月10日凌晨4时许'

words = jieba.lcut(l)
poses = ' '.join(postagger.postag(words)).split()

arcs = parser.parse(words, poses)
arcses = ' '.join("%d:%s" % (arc.head, arc.relation) for arc in arcs).split()
print(arcs)
print(arcses)
l = len(words)
adj_matrix = np.zeros((l, l))

for k in range(l):

    for j in range(l):
        # 数字表示父亲节点,0表示head
        if (int(arcses[k].split(':')[0]) - 1) == j:
            adj_matrix[j][k] = 1  # j->k
            adj_matrix[k][j] = -1  #

print(adj_matrix)
