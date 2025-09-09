import json
import os
import random
import Bio
import numpy as np
import pandas as pd
from datasets import Dataset
import test
from opencompass.datasets.base import BaseDataset
from .utils import generate_example_sequences
from opencompass.utils import get_data_path
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET
from Bio.Data.CodonTable import NCBICodonTableDNA
from Bio.Seq import Seq
from tqdm import tqdm
from .utils import generate_random_codon_table
default_prompt = """### **Task Description**  
Your goal is to transcribe a given DNA sequence into an amino acid sequence.  

However, **the codon table may not be standard**, so you **cannot rely on external knowledge or use a conventional codon table directly**. Instead, you must **derive the codon table solely from the provided examples**.  

Do not use code to solve this task. I need you to solve it yourself using inductive reasoning.

### **Your Task**  
I will provide several **DNA → amino acid** mapping examples.  
You need to **carefully analyze these examples** and **infer the codon table** based on the observed patterns.  

Then, using the **codon table you deduced**, transcribe the given DNA sequence into an amino acid sequence.  

### **Example Mappings (for you to infer the codon table)**  
```
{example}
```  

### **DNA Sequence to Transcribe**  
```
{DNA}
```  

### **Important Notes**  
- The codon table **must be inferred only from the provided examples**—do **not** use any external knowledge or assume the standard codon table.  
- The given examples contain enough information to determine the necessary codon mappings.  
- Only output the final transcribed result—do **not** include the input DNA sequence.  
- The transcribed result should be placed between "```result" and "```".  
- Do not use code to deduce or transcribe the sequence—solve it manually using inductive reasoning.

### **Transcription Result** 
"""
@LOAD_DATASET.register_module()
class DNATranslatorDataset(BaseDataset):

    
    @staticmethod
    def process_sequences(test_df, train_df,n=3, synthesize=False):
        """从 test.csv 读取 `seq`，在 train.csv 里找包含所有密码子的匹配项"""
        
        # 结果存储
        results = []

        for _, test_row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing sequences"):
            test_seq = test_row["seq"]
            test_aa = test_row["protein seq"]
            test_codons = set(test_seq[i:i+3] for i in range(0, len(test_seq) - 2, 3))  # 提取密码子集合
            
            
            attempts = 0
            max_attempts = 20  # 设置最大尝试次数，防止无限循环
            matched_rows = []  # 匹配的训练集行
            example_start_codons = set()
            example_stop_codons = set()
            while attempts < max_attempts:
                # 从训练集随机抽样
                sampled_rows = train_df.sample(n)
                train_codons = set()
                example_start_codons = set()
                example_stop_codons = set()
                for _, train_row in sampled_rows.iterrows():
                    train_seq = train_row["seq"]
                        
                    # 检查起始密码子和终止密码子是否匹配
                    example_start_codons.add(train_seq[:3])
                    example_stop_codons.add(train_seq[-3:])
                        
                    # 提取当前序列的密码子
                    current_codons = set(train_seq[i:i+3] for i in range(0, len(train_seq) - 2, 3))
                    # 合并到总集合中
                    train_codons = train_codons.union(current_codons)
                if test_seq[:3] not in example_start_codons or test_seq[-3:] not in example_stop_codons:
                    attempts += 1
                    continue
                # 检查训练序列是否包含测试序列的所有密码子
                if test_codons.issubset(train_codons):
                    # 避免重复添加相同的序列
                    
                    matched_rows = [(row["seq"], row["protein seq"]) for _, row in sampled_rows.iterrows()]
                    break
                
                attempts += 1

            # 如果找不到足够的匹配序列，可以输出警告
            if attempts == max_attempts:
                print(f"无法找到足够的匹配序列，尝试次数: {max_attempts}")
                continue

            
            if synthesize== False:  # 如果没有启用合成模式
                results.append({
                    "test_seq": test_seq,
                    "test_aa": test_aa,
                    "examples": [{"seq": seq, "aa": protein} for seq, protein in matched_rows],
                    
                })
            else:  # 如果启用了合成模式
                
                table = generate_random_codon_table(start_codons=list(example_start_codons),stop_codons=list(example_stop_codons))  # 生成随机密码表

                # 生成目标序列（纯净）
                target_aa = str(Seq(test_seq).translate(table=table))#.strip("*")

                # 验证目标序列
                synthesized = []

                for row in matched_rows:
                    synthesized_aa = str(Seq(row[0]).translate(table=table))#.strip("*")

                    synthesized.append((row[0], synthesized_aa))

                
                results.append({
                    "test_seq": test_seq,
                    "test_aa": target_aa,
                    "examples": [{"seq": seq, "aa": protein} for seq, protein in synthesized],
                    "table": {
                        "forward_table": table.forward_table,
                        "start_codons": table.start_codons,
                        "stop_codons": table.stop_codons,
                    },
                })
            

        
        return results
    @staticmethod
    def process_sequences_synthesize_example(num_samples,  n=3, synthesize=False, max_len=400, min_len=200, max_try=20):
        """从 test.csv 读取 `seq`，在 train.csv 里找包含所有密码子的匹配项"""
        
        # 结果存储
        results = []

        for _ in range(num_samples):
            attempts = 0  # 当前尝试次数
            matched_rows = []  # 匹配的训练集行
            while attempts < max_try:
                # 根据 synthesize 参数决定密码表的生成方式
                if synthesize:
                    # 使用随机密码表
                    table = generate_random_codon_table(
                    )
                else:
                    # 使用标准密码表
                    table = Bio.Data.CodonTable.generic_by_id[random.choice(list(Bio.Data.CodonTable.generic_by_id.keys()))]
                # 采样 n 个 example
                gen_rows = generate_example_sequences(table, n=n+1, max_len=max_len, min_len=min_len)
                sampled_rows = gen_rows[:n]  # 只保留前 n 个示例
                test_row = gen_rows[-1]  # 最后一个示例作为测试序列
                test_seq = test_row["seq"]
                test_aa = test_row["aa"]
                test_codons = set(test_seq[i:i+3] for i in range(0, len(test_seq) - 2, 3))  # 提取密码子集合
                train_codons = set()
                example_start_codons = set()
                example_stop_codons = set()
                for train_row in sampled_rows:
                    train_seq = train_row["seq"]
                        
                    # 检查起始密码子和终止密码子是否匹配
                    example_start_codons.add(train_seq[:3])
                    example_stop_codons.add(train_seq[-3:])
                        
                    # 提取当前序列的密码子
                    current_codons = set(train_seq[i:i+3] for i in range(0, len(train_seq) - 2, 3))
                    # 合并到总集合中
                    train_codons = train_codons.union(current_codons)
                if test_seq[:3] not in example_start_codons or test_seq[-3:] not in example_stop_codons:
                    attempts += 1
                    continue
                # 检查训练序列是否包含测试序列的所有密码子
                if test_codons.issubset(train_codons):
                    # 避免重复添加相同的序列
                    
                    matched_rows = [(row["seq"], row["aa"]) for row in sampled_rows]
                    break
                
                attempts += 1
                # 如果找不到足够的匹配序列，可以输出警告
            if attempts == max_try:
                print(f"无法找到足够的匹配序列，尝试次数: {max_try}")
                continue
            if matched_rows!=[]:
                results.append({
                    "test_seq": test_seq,
                    "test_aa": test_aa,
                    "examples": [{"seq": seq, "aa": protein} for seq, protein in matched_rows],
                    "table": {
                        "forward_table": table.forward_table,
                        "start_codons": table.start_codons,
                        "stop_codons": table.stop_codons,
                    },
                })
            else:
                raise ValueError(f"没有找到匹配的序列，尝试次数: {max_try}")

        return results

    @staticmethod
    def load(
        path: str,  # 数据集路径，包含 test 和 train 文件
        icl_num: int = 10,  # ICL 示例数量
        num_samples: int = 100,  # 总样本数
        max_len: int = 400,  # max sequence length
        min_len: int = 200,  # min sequence length
        synthesize_example: bool = False,  # 是否启用合成示例
        synthesize_table: bool = False,  # is the mapping synthetic
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

        # 加载测试集和训练集
        test_df = pd.read_csv(test_path)
        train_df = pd.read_csv(train_path)

        # 筛选符合长度要求的序列
        test_df = test_df[(test_df['seq'].str.len() <= max_len) & (test_df['seq'].str.len() >= min_len)]
        train_df = train_df[(train_df['seq'].str.len() <= max_len) & (train_df['seq'].str.len() >= min_len)]

        # 检查训练集是否足够生成 ICL 示例
        if len(train_df) < icl_num:
            raise ValueError(f"训练集中的序列数量 ({len(train_df)}) 少于请求的 ICL 示例数量 ({icl_num})")

        # 定义长度区间
        interval_width = (max_len - min_len) / 5
        intervals = []
        for i in range(5):
            start = int(min_len + i * interval_width)
            # For the last interval, make sure the end is exactly max_len
            end = int(min_len + (i + 1) * interval_width) if i < 4 else max_len
            intervals.append((start, end))
        samples_per_interval = 20
        stratified_samples = pd.DataFrame()

        # 从每个区间采样
        for min_interval, max_interval in intervals:
            # 筛选该区间的序列
            interval_df = test_df[(test_df['seq'].str.len() >= min_interval) & 
                                (test_df['seq'].str.len() < max_interval)]
            
            # 如果区间有足够数据，进行采样
            if len(interval_df) > 0:
                # 采样该区间最多20个样本
                sample_size = min(samples_per_interval, len(interval_df))
                interval_sample = interval_df.sample(n=sample_size, random_state=random_seed)
                stratified_samples = pd.concat([stratified_samples, interval_sample])

        # 如果总样本数不足num_samples，从整体数据中补充
        if len(stratified_samples) < num_samples:
            remaining_needed = num_samples - len(stratified_samples)
            # 排除已采样的样本
            remaining_df = test_df[~test_df.index.isin(stratified_samples.index)]
            if len(remaining_df) > 0:
                additional_samples = remaining_df.sample(n=min(remaining_needed, len(remaining_df)), 
                                                    random_state=random_seed)
                stratified_samples = pd.concat([stratified_samples, additional_samples])

        if len(stratified_samples) > num_samples:
            stratified_samples = stratified_samples.sample(n=num_samples, random_state=random_seed)
        # 使用分层采样的结果作为测试集
        test_df = stratified_samples

        # 生成 few-shot 数据
        data_path = os.path.join(path, f"dna_translator/data_{icl_num}-shot_test-num={num_samples}-maxlen={max_len}-minlen={min_len}-synthesize-example={synthesize_example}-synthesize-table={synthesize_table}.json")
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                results = json.load(f)
        else:
            if synthesize_example:
                # 生成合成示例
                results = DNATranslatorDataset.process_sequences_synthesize_example(num_samples, n=icl_num, synthesize=synthesize_table, max_len=max_len, min_len=min_len)
            else:
                results = DNATranslatorDataset.process_sequences(test_df, train_df, icl_num, synthesize_table)
            with open(data_path, 'w') as f:
                json.dump(results, f, indent=4)
                

        # 转换为 HuggingFace Dataset 格式
        data = {'prompt': [], 'answer': []}
        for test_example in results:
            few_shot_examples = [
                f"DNA:\n{ex['seq']}\nAA:\n{ex['aa']}\n"
                for ex in test_example["examples"]
            ]

            test_prompt = template.format(
                example="\n".join(few_shot_examples),
                DNA=test_example["test_seq"],
            )
            answer = test_example['test_aa']
            data['prompt'].append(test_prompt)
            data['answer'].append(answer)

        return Dataset.from_dict(data)



class DNATranslatorEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        """
        计算预测结果的准确率。
        """
        scores = []
        for pred, ref in zip(predictions, references):
            # 去除预测结果中的空格
            extracted_content = pred.replace(' ', '').replace('*', '')

            # 验证准确率，一个氨基酸一个氨基酸对比，可能长度不一致
            acc = sum(1 for a, b in zip(ref.rstrip('*'), extracted_content) if a == b) / len(ref.rstrip('*'))
            scores.append(acc)

        # 返回平均准确率
        return {'accuracy': round(sum(scores) / len(scores) * 100, 2)}