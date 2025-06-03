import re
from sklearn.metrics import cohen_kappa_score

def score_to_label(score):
    mapping = {0.0: 0, 0.5: 1, 1.0: 2}
    return mapping[score]

def extract_scores_from_file(filename):
    pattern = re.compile(r'^\s*(0|0\.5|1)\s+(0|0\.5|1)\s*$')
    raw_rater1 = []
    raw_rater2 = []
    labels_rater1 = []
    labels_rater2 = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.match(line.strip())
            if match:
                score1, score2 = map(float, match.groups())
                raw_rater1.append(score1)
                raw_rater2.append(score2)
                labels_rater1.append(score_to_label(score1))
                labels_rater2.append(score_to_label(score2))

    return raw_rater1, raw_rater2, labels_rater1, labels_rater2
model_list = ['BLIP','GIT','OFA','OSCAR']
for model_name in model_list:
    raw1, raw2, label1, label2 = extract_scores_from_file(f'sample/{model_name}_sample')
    kappa = cohen_kappa_score(label1, label2)
    mean1 = (200 - sum(raw1)) / len(raw1)
    mean2 = (200 - sum(raw2)) / len(raw2)

    print(f"{model_name}'s Cohen's Kappa: {kappa:.4f}")
    print(f"{model_name}'s Rater 1 Mean Score: {mean1:.4f}")
    print(f"{model_name}'s Rater 2 Mean Score: {mean2:.4f}")


