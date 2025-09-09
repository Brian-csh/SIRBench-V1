import json
import os
import random
import numpy as np
import pandas as pd
from datasets import Dataset
from opencompass.datasets.base import BaseDataset
from opencompass.utils import get_data_path
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET
from Bio.Data.CodonTable import NCBICodonTableDNA
from Bio.Seq import Seq
from tqdm import tqdm
default_prompt = """### **Task Description**  
Your goal is to transform a given nucleotide sequence based on a hidden transformation rule.  

However, **the transformation rule is not fixed**—it may involve reversing sequences, generating complementary sequences, reverse-complement transformations, or other modifications such as **applying segmented transformations** or **mutating specific bases**. **Both the transformation type and pairing rules may vary**, so you **cannot assume a predefined rule**—instead, you must **infer the transformation pattern solely from the provided examples**.  

### **Your Task**  
I will provide several **input → output** transformation examples.  
You need to **carefully analyze these examples** and **deduce the transformation rule** based on the observed patterns.  

Then, using the **transformation rule you inferred**, convert the given input sequence into its corresponding output sequence.  

### **Example Transformations (for you to infer the pattern)**  
```
{example}
```  

### **Input Sequence to Transform**  
```
{input}
```  

### **Important Notes**  
- The transformation rule **must be inferred only from the provided examples**—do **not** rely on external knowledge or assume any fixed transformation logic.  
- Possible transformations include:  
  - **Reversing the sequence**  
  - **Generating the complementary sequence** (note: the nucleotide pairing rule may vary)  
  - **Reverse-complement transformation** (the complement rule follows the same variation as the complementary sequence rule)  
  - **Segmented transformation**: After every **x** bases, the next **n** bases undergo a transformation. **The transformation applied to these bases follows one of the previously observed transformation rules.**  
  - **Fixed-base mutation**: A specific nucleotide **X** is changed into another nucleotide **Y** (both X and Y may vary).  
- The exact transformation rule must be derived from the examples provided.  
- Only output the final transformed result—do **not** include the input sequence.  
- The transformed result should be placed between "```result" and "```".  

### **Transformed Result**
"""
def reverse(seq):
    return seq[::-1]

def complement(seq, complement_map):
    return ''.join([complement_map[base] for base in seq])

def reverse_complement(seq, complement_map):
    return reverse(complement(seq, complement_map))

def segmented_transform(seq, x, n, transform_type, complement_map):
    '''使其在每隔 x 个碱基后，对接下来的 n 个碱基进行一次变换。'''
    result = list(seq)
    i = 0
    while i < len(seq):
        sub_seq = seq[i:i + n]
        if transform_type == 'reverse':
            transformed_sub_seq = reverse(sub_seq)
        elif transform_type == 'complement':
            transformed_sub_seq = complement(sub_seq, complement_map)
        elif transform_type == 'reverse_complement':
            transformed_sub_seq = reverse_complement(sub_seq, complement_map)
        else:
            raise ValueError("Unknown transformation type")
        result[i:i + n] = transformed_sub_seq
        i += x + n
    return ''.join(result)

def fixed_base_mutation(seq, x, y):
    '''将固定的碱基 X 变异为碱基 Y'''
    return seq.replace(x, y)

def apply_transform(seq, transform_type, complement_map, x=None, n=None, base_x=None, base_y=None,segmented_transform_type='complement'):
    if transform_type == 'reverse':
        return reverse(seq)
    elif transform_type == 'complement':
        return complement(seq, complement_map)
    elif transform_type == 'reverse_complement':
        return reverse_complement(seq, complement_map)
    elif transform_type == 'segmented_transform':
        if x is None or n is None:
            raise ValueError("For 'segmented_transform' rule, x and n must be provided")
        return segmented_transform(seq, x, n, segmented_transform_type, complement_map)  # 假设默认变换类型为 'complement'
    elif transform_type == 'fixed_base_mutation':
        if base_x is None or base_y is None:
            raise ValueError("For 'fixed_base_mutation' rule, base_x and base_y must be provided")
        return fixed_base_mutation(seq, base_x, base_y)
    else:
        raise ValueError("Unknown transformation type")
@LOAD_DATASET.register_module()
class DNATransformDataset(BaseDataset):
    
    @staticmethod
    def process_sequences(test_df, train_df, icl_num, seq_len=400):
        


        data = []

        complement_maps = [
            {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'},
            {'A': 'C', 'T': 'G', 'C': 'A', 'G': 'T'},
            {'A': 'G', 'T': 'C', 'C': 'T', 'G': 'A'}
        ]

        for index, row in test_df.iterrows():
            test_seq = row['seq']

            key = random.choice(['reverse', 'complement', 'reverse_complement', 'segmented_transform', 'fixed_base_mutation'])
            examples = []

            # 从train_df中随机选择10个样本
            train_sample = train_df.sample(n=icl_num)

            # 随机裁剪出max_len长度的序列， test和train都要裁剪，开始位置随机
            
            start = random.randint(0, len(test_seq) - seq_len)
            test_seq = test_seq[start:start + seq_len]
            for i in train_sample.index:
                
                seq = train_df.loc[i, 'seq']
                start = random.randint(0, len(seq) - seq_len)
                train_df.loc[i, 'seq'] = seq[start:start+ seq_len]


            # 随机选择一个 complement_map
            complement_map = random.choice(complement_maps)
        

            transform_params = {
                "complement_map": complement_map,
                "transform_type": key
            }

            if key == 'segmented_transform':
                x = random.randint(1, 10)
                n = random.randint(1, 20)
                anwser = apply_transform(test_seq, key, complement_map, x, n)
                segmented_transform_type = random.choice(['reverse', 'complement', 'reverse_complement'])
                transform_params.update({"x": x, "n": n, "segmented_transform_type": segmented_transform_type})
                for i in train_sample.index:
                    seq = train_df.loc[i, 'seq']
                    output = apply_transform(seq, key, complement_map, x, n, segmented_transform_type=segmented_transform_type)
                    examples.append({"input": seq, "output": output})
            elif key == 'fixed_base_mutation':
                base_x = random.choice('ATCG')
                base_y = random.choice('ATCG'.replace(base_x, ''))
                anwser = apply_transform(test_seq, key, complement_map, base_x=base_x, base_y=base_y)
                transform_params.update({"base_x": base_x, "base_y": base_y})
                for i in train_sample.index:
                    seq = train_df.loc[i, 'seq']
                    output = apply_transform(seq, key, complement_map, base_x=base_x, base_y=base_y)
                    examples.append({"input": seq, "output": output})
            else:
                anwser = apply_transform(test_seq, key, complement_map)
                for i in train_sample.index:
                    seq = train_df.loc[i, 'seq']
                    output = apply_transform(seq, key, complement_map)
                    
                    examples.append({"input": seq, "output": output})

            data.append({
                "test_seq": test_seq,
                "anwser": anwser,
                "examples": examples,
                "transform_params": transform_params
            })

        return data


    @staticmethod
    def load(
        path: str,  # 数据集路径，包含 test 和 train 文件
        icl_num: int = 10,  # ICL 示例数量
        num_samples: int = 100,  # 总样本数
        seq_len: int = 400,  # max sequence length
        # min_len: int = 200,  # min sequence length
        template: str = default_prompt,  # 提示模板
        random_seed: int = 42,  # 随机种子
    ) -> Dataset:
        """
        加载 DNA 数据集并生成 few-shot 数据。
        """

        random.seed(random_seed)
        np.random.seed(random_seed)
        path = get_data_path(path)
        # 定义测试集和训练集路径
        test_path = os.path.join(path, "dna_translator/20230906_cds_res_nt2aa_test.csv")
        train_path = os.path.join(path, "dna_translator/20230906_cds_res_nt2aa_train.csv")

        # 读取CSV文件
        test_df = pd.read_csv(test_path)
        train_df = pd.read_csv(train_path)
        # test_df = test_df[test_df['seq'].str.len() <= max_len]
        test_df = test_df[seq_len<=test_df['seq'].str.len() ]
        # train_df = train_df[train_df['seq'].str.len() <= max_len]
        train_df = train_df[seq_len<=train_df['seq'].str.len() ]

        # 随机采样测试集
        test_df = test_df.sample(n=min(num_samples, len(test_df)), random_state=random_seed)

        # 生成 few-shot 数据
        data_path = os.path.join(path, f"dna_transform/data_{icl_num}-shot_test-num={num_samples}_seqlen={seq_len}.json")
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                results = json.load(f)
        else:
            results = DNATransformDataset.process_sequences(test_df, train_df, icl_num,seq_len=seq_len)
            with open(data_path, 'w') as f:
                json.dump(results, f, indent=4)


        # 转换为 HuggingFace Dataset 格式
        data = {'prompt': [], 'answer': []}
        for test_example in results:

            assert len(test_example['examples']) == icl_num, f"Expected {icl_num} examples but got {len(test_example['examples'])}"
            examples = '\n'.join([f"**Input:**\n{example['input']}\n**Output:**\n{example['output']}\n" for example in test_example['examples']])
            eval_prompt= template.format(example=examples, input=test_example['test_seq'])
            answer = test_example['anwser']
            data['prompt'].append(eval_prompt)
            data['answer'].append(answer)

        return Dataset.from_dict(data)



class DNATransformEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        """
        计算预测结果的准确率。
        """

        scores = []
        for pred, ref in zip(predictions, references):
            # 去除预测结果中的空格
            extracted_content = pred.replace(' ', '')

            # 验证准确率，一个氨基酸一个氨基酸对比，可能长度不一致
            acc = sum(1 for a, b in zip(ref, extracted_content) if a == b) / len(ref)
            scores.append(acc)

        # 返回平均准确率
        return {'accuracy': round(sum(scores) / len(scores) * 100, 2)}