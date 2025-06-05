from collections import Counter
import re

# The sample is the same with rq2

model_list = ["BLIP","GIT","OFA","OSCAR"]


with open('../rq_results/error_distribution','w') as f:
    for model_name in model_list:
        file_path = f'manual/{model_name}_sample'
        label_counter = Counter()

        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                match = re.match(r'^(0|0\.5|1)\s+(0|0\.5|1)\s+(\S+)\s+(\S+)$', line)
                if match:
                    score1, score2, label1, label2 = match.groups()
                    if score1 != '1' and score2 != '1':
                        label_counter[label1] += 1
        total = sum(label_counter.values())
        print(f"{model_name}'s Error distribution (filtered):")
        for label, count in label_counter.items():
            proportion = count / total
            print(f"{label}: {count} ({proportion:.2%})")