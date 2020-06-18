from fractions import Fraction as fr

import pandas as pd
import numpy as np


class ConsistencyTest(object):



    def __init__(self, A):
        self.RI = [0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49, 1.51]
        self.A = A
        self.index = self.A.index.tolist()
        # 列向量求和，修改传入的A比较矩阵
        self.sum_a0 = []
        for i in A.index:
            self.sum_a0.append(np.sum(A[i]))
        self.A.loc['合计'] = self.sum_a0

    @property
    def eigenvector_max(self):
        """最大特征向量简化计算"""
        # 1.列向量归一化
        index_to_one = self.A[0:-1].apply(lambda x: x / self.A.loc['合计'], axis=1)
        # 2.求行和
        sum_col = np.sum(index_to_one, axis=1)
        # 3.归一化
        w = sum_col.apply(lambda x: x / np.sum(sum_col))
        return w

    @property
    def eigenvalue_max(self):
        """最大特征值，简便算法"""
        # 1.求Aw
        Aw = []
        for i in self.index:
            Aw.append(np.sum(self.A.loc[i] * self.eigenvector_max))
        Aw = pd.Series(Aw, index=self.index)

        # 2.求求最大特征向量
        r = np.sum(Aw / self.eigenvector_max) / 3
        return r

    def consistency(self):
        CI = (self.eigenvalue_max - len(self.index)) / (len(self.index) - 1)
        # CR = CI/RI
        CR = CI / self.RI[len(self.index)]
        if CR < 0.1:
            print('通过一致性检验!')
        else:
            print('没有通过!')


index = ['贡献', '收入', '发展', '声誉', '关系', '位置']
# 构造比较矩阵
A = pd.DataFrame({
    index[0]: [1, 9, 8, 6, fr(1, 2), 1],
    index[1]: [fr(1, 9), 1, fr(1, 8), fr(1, 9), fr(1, 7), fr(1, 5)],
    index[2]: [fr(1, 8), 8, 1, fr(1, 7), fr(1, 3), fr(1, 2)],
    index[3]: [fr(1, 6), 9, 7, 1, fr(1, 3), 2],
    index[4]: [2, 7, 3, 3, 1, 4],
    index[5]: [1, 5, 2, fr(1, 2), fr(1, 4), 1]
}, index=index)

test = ConsistencyTest(A)
print(test.A)
print(test.eigenvalue_max)
print(test.eigenvector_max)
test.consistency()