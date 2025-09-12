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

            raise ValueError(f"invalid score: {score}")

    total = len(scores)

    if total > 0:
        prop_1 = count_1 / total
        prop_2 = count_2 / total
        prop_3 = count_3 / total
    else:
        prop_1 = prop_2 = prop_3 = 0.0

    counts = (count_1, count_2, count_3)
    proportions = (prop_1, prop_2, prop_3)

    return counts, proportions, total

scores_p1 = read_scores('score_p1')
scores_p2 = read_scores('score_p2')
scores_f = read_scores('score_f')

kappa = cohen_kappa_score(scores_p1, scores_p2)
counts, proportions, total = score_distribution(scores_f)
with open('../rq_results/integrity_score','w') as f:
    print(f"Cohen's Kappa: {kappa:.4f}",file=f)
    print(proportions,file=f)