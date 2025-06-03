from sklearn.metrics import cohen_kappa_score

def read_scores(file_path):
    with open(file_path, 'r') as file:
        return [int(line.strip()) for line in file]

def score_distribution(scores):
    count_1 = count_2 = count_3 = 0

    # 遍历所有评分
    for score in scores:
        if score == 1:
            count_1 += 1
        elif score == 2:
            count_2 += 1
        elif score == 3:
            count_3 += 1
        else:
            # 如果发现无效评分，抛出异常
            raise ValueError(f"无效评分值: {score}。只允许1,2,3")

    # 计算总数
    total = len(scores)

    # 计算比例（避免除以零错误）
    if total > 0:
        prop_1 = count_1 / total
        prop_2 = count_2 / total
        prop_3 = count_3 / total
    else:
        prop_1 = prop_2 = prop_3 = 0.0

    # 返回结果
    counts = (count_1, count_2, count_3)
    proportions = (prop_1, prop_2, prop_3)

    return counts, proportions, total

scores_p1 = read_scores('score_p1')
scores_p2 = read_scores('score_p2')
scores_f = read_scores('score_f')

kappa = cohen_kappa_score(scores_p1, scores_p2)
counts, proportions, total = score_distribution(scores_f)
print(f"Cohen's Kappa: {kappa:.4f}")
print(proportions)