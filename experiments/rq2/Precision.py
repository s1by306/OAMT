import re
from sklearn.metrics import cohen_kappa_score

def score_to_label(score):
    mapping = {0.0: 0, 0.5: 1, 1.0: 2}
    return mapping[score]

def extract_scores_from_file(filename):
    score_pair_pattern = re.compile(r'^\s*(0|0\.5|1)\s+(0|0\.5|1)\s*$')
    final_score_pattern = re.compile(r'^\s*(0|0\.5|1)\s*$')
    raw_rater1 = []
    raw_rater2 = []
    final_score = []
    labels_rater1 = []
    labels_rater2 = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            score_pair_match = score_pair_pattern.match(line.strip())
            final_score_match = final_score_pattern.match(line.strip())
            if score_pair_match:
                score1, score2 = map(float, score_pair_match.groups())
                raw_rater1.append(score1)
                raw_rater2.append(score2)
                labels_rater1.append(score_to_label(score1))
                labels_rater2.append(score_to_label(score2))
            if final_score_match:
                score = float(final_score_match.group())
                final_score.append(score)
    return raw_rater1, raw_rater2, labels_rater1, labels_rater2, final_score



model_list = ['BLIP','GIT','OFA','OSCAR']
for model_name in model_list:
    raw1, raw2, label1, label2, final_score = extract_scores_from_file(f'sample/{model_name}')
    kappa = cohen_kappa_score(label1, label2)
    with open('../rq_results/precision','a') as f:
        mean = (len(final_score) - sum(final_score))/len(final_score)
        print(f"{model_name}'s Cohen's Kappa: {kappa:.4f}",file = f)
        print(f"{model_name}'s Precision:{mean:.4f}",file = f)



