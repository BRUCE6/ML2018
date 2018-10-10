# Xiaoran Ni
# 2018.10.8

class knn_classifier():
    def __init__(self):
        self.data = []
    
    
    def fit(self, filename):
        with open(filename, 'r') as f:
            content = f.read()
            lines = content.split('\n')
            for line in lines:
                if line:
                    y, sen = int(line[0]), line[2:]
                    self.data.append((sen, y))
    
    def fit_lines(self, lines):
        for line in lines:
            if line:
                y, sen = int(line[0]), line[2:]
                self.data.append((sen, y))
                
    def predict(self, sen, k = 1, func = 'intersect'):
        dsts = []
        for i, (sen_i, y) in enumerate(self.data):
            dsts.append((y, self._distance(sen_i, sen, func = func)))
        dsts.sort(key = lambda x: x[-1])
        min_d = dsts[k - 1][-1]
        num_ones = sum([y for y, d in dsts[:k]])
        total = k
        for y, d in dsts[k:]:
            if d == min_d:
                num_ones += y
                total += 1
            else:
                break
        if num_ones >= total / 2:
            return 1
        else:
            return 0
    
    
    def _buildSet(self, sen):
        s = set()
        for w in sen.split():
            s.add(w)
        return s
    
    
    def _buildDict(self, sen):
        d = {}
        for w in sen.split():
            d[w] = d.get(w, 0) + 1
        norm = sum([v ** 2 for v in d.values()]) ** (1/2)
        for k in d:
            d[k] /= norm
        return d
    
    
    def _distance(self, sen1, sen2, func = 'intersect'):
        if func == 'intersect':
            s1, s2 = self._buildSet(sen1), self._buildSet(sen2)
            inter = len(s1 & s2)
            if not inter:
                return float('inf')
            return 1 / inter
        if func == 'cosine':
            d1, d2 = self._buildDict(sen1), self._buildDict(sen2)
            res = 0
            for w in d1.keys():
                res += d1[w] * d2.get(w, 0)
            return 1 - res


def statistics_knn(test_data, k, func):
    KNN = knn_classifier()
    KNN.fit('reviewstrain.txt')
    acc = 0
    TP, FP, TN, FN = 0, 0, 0, 0
    for i, (sen, y) in enumerate(test_data):
#         if i == 17:
#             print(sen, KNN.predict(sen, k = k), y)
        predict = KNN.predict(sen, k = k, func = func)
        if predict == y:
            if predict == 1:
                TP += 1
            else:
                TN += 1
            acc += 1
        else:
            if predict == 1:
                FP += 1
            else:
                FN += 1
    acc /= len(test_data)
    print('k:', k)
    print('TP, FP, TN, FN:', TP, FP, TN, FN)
    print('accuracy:', acc)


if __name__ == '__main__':
    test_data = []
    with open('reviewstest.txt', 'r') as f:
        content = f.read()
        lines = content.split('\n')
        for line in lines:
            if line:
                y, sen = int(line[0]), line[2:]
                test_data.append((sen, y))
    # part (a)
    print('part(a)')
    statistics_knn(test_data, 1, 'intersect')
    statistics_knn(test_data, 5, 'intersect')
    print()

    # part (c)
    print('part(c)')
    with open('reviewstrain.txt', 'r') as f:
        content = f.read()
        all_lines = content.split('\n')
    all_lines = [line for line in all_lines if line]

    list_k = [3, 7, 99]
    for k in list_k:
        num_folds = 5
        step = len(all_lines) // num_folds
        acc = 0
        for i in range(num_folds):
            if i < num_folds - 1:
                val_lines = all_lines[step * i: step * (i + 1)]
                train_lines = all_lines[:step * i] + all_lines[step * (i + 1):]
            else:
                val_lines = all_lines[step * i:]
                train_lines = all_lines[:step * i]
            KNN = knn_classifier()
            KNN.fit_lines(train_lines)
            for line in val_lines:
                y, sen = int(line[0]), line[2:]
                predict = KNN.predict(sen, k = k)
                acc += int(predict == y)
        print('k', k, 'accuracy', acc / len(all_lines))
    statistics_knn(test_data, 3, 'intersect')
    print()

    # part (d)
    print('part(d)')
    statistics_knn(test_data, 1, 'cosine')
    statistics_knn(test_data, 5, 'cosine')

