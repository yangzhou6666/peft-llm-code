import os
from tabulate import tabulate
import numpy as np
import json
import yaml
from collections import OrderedDict

def pass_at_k(n, c, k):
    """
    compute the pass@k value
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

class FunctionalCorrectness:
    K = [1, 5, 10, 25, 50, 100]
    def __init__(self, eval_dir):
        self.pass_k = [[] for _ in range(len(self.K))]
        for fname in os.listdir(eval_dir):
            if not fname.endswith('.yaml'): continue
            with open(os.path.join(eval_dir, fname)) as f:
                res_data = yaml.load(f, Loader=yaml.CLoader)
            n, c = 0, 0
            for r in res_data['results']:
                n += 1 # count the number of generations
                if r['status'] == 'OK':
                    c += 1 # count the number of correct generations
            for i, k in enumerate(self.K):
                # 对于每个问题，计算pass@k，并且存储
                self.pass_k[i].append(pass_at_k(n, c, k))
        for i, k in enumerate(self.K):
            # 对于每一个k，取平均.
            self.pass_k[i] = np.mean(self.pass_k[i])*100
            
    def pretty_print(self):
        header, row = [], []
        for i, k in enumerate(self.K):
            header.append(f'pass@{k}')
            row.append('{:.1f}'.format(self.pass_k[i]))
        print(tabulate([row], headers=header, stralign='right', tablefmt='orgtbl'))

    def get_pass_k(self):
        res = OrderedDict()
        for i, k in enumerate(self.K):
            res[f'pass@{k}'] = self.pass_k[i]
        return res
            
if __name__ == '__main__':
    
    for loss in ["all_after", "learn_added_penalize_deleted", "learn_added_penalize_deleted-enhance_correctness"]:
        
        result_path = f"correctness_eval/results/{loss}/codegen-350M-multi_lora/execution"
        print(result_path)
        functional_correctness = FunctionalCorrectness(result_path)
        functional_correctness.pretty_print()
    
    
    # result_path = f"correctness_eval/results/Salesforce/codegen-350M-multi_lora/execution"
    # print(result_path)
    # functional_correctness = FunctionalCorrectness(result_path)
    # functional_correctness.pretty_print()