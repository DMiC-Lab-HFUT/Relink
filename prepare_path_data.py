import json
from utils.neo4j_operator import deduplicates_list
import numpy as np
import random
from tqdm import tqdm

#
def collect_data_from_file(file):
    samples = []
    seen = set()
    with open(file, encoding='utf-8') as f:
        for line in tqdm(f):
            d = json.loads(line)
            key = (d[0], d[1]['format_string'])
            if key in seen:
                continue
            else:
                seen.update(key)
                try:
                    if len(d[1]['relations']) < 1:
                        continue
                    item = {
                        'query': d[0],
                        'path': [(r['begin']['mention'], r['r'], r['end']['mention'], r['title']) for r in
                                 d[1]['relations']],
                        'path_score': d[1]['score'],
                        'last_triple_score': d[1]['scores'][-1]
                    }
                    samples.append(item)
                except Exception as e:
                    print(e.__str__())
    return samples

samples = []
samples += collect_data_from_file('./results/WMQA/path.jsonl')
samples += collect_data_from_file('./results/CQA/path.jsonl')




random.shuffle(samples)
random.shuffle(samples)

with open('path_samples.jsonl', 'w', encoding='utf-8') as f:
    for d in tqdm(samples):
        f.write(json.dumps(d, ensure_ascii=False)+'\n')


with open('path_samples.json', 'w', encoding='utf-8') as f:
    json.dump(samples, f, ensure_ascii=False, indent=4)

