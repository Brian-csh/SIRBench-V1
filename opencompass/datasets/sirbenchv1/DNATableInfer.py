import json
import os
import random
import Bio
import numpy as np
import pandas as pd
from datasets import Dataset
from opencompass.datasets.base import BaseDataset
from opencompass.datasets.sirbenchv1.DNATranslator import DNATranslatorDataset
from opencompass.utils import get_data_path
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET
from Bio.Data.CodonTable import NCBICodonTableDNA
from Bio.Seq import Seq
from tqdm import tqdm
from .utils import generate_random_codon_table, generate_example_sequences
default_prompt = """### **Task Description**  
Your goal is to analyze the provided DNA → amino acid mapping examples and **infer the codon table**.  

However, **the codon table may not be standard**, so you **cannot rely on external knowledge or use a conventional codon table directly**. Instead, you must **derive the codon table solely from the provided examples**.  

### **Your Task**  
I will provide several **DNA → amino acid** mapping examples.  
You need to **carefully analyze these examples** and **infer the codon table** based on the observed patterns.  

Format the inferred codon table as follows:  

```result
{{
    "forward_table": {{
        "XXX": "M",
        ...
    }},
    "start_codons": [
        "XXX"
    ],
    "stop_codons": [
        "YYY",
        "ZZZ"
    ]
}}
```  

### **Example Mappings (for you to infer the codon table)**  
```
{example}
```  

### **Important Notes**  
- The codon table **must be inferred only from the provided examples**—do **not** use any external knowledge or assume the standard codon table.  
- The given examples contain enough information to determine the necessary codon mappings.  
- Only output the inferred codon table—do **not** include the transcribed amino acid sequence or the input DNA sequence.  
- The codon table should be placed between "```result" and "```". 
- Do **not** include any comments, explanations, or text inside the JSON structure.
"""
@LOAD_DATASET.register_module()
class DNATableInferDataset(BaseDataset):

    @staticmethod
    def process_sequences(num_samples, train_df, n=3, synthesize=False, max_try=20):
        """从 test.csv 读取 `seq`，在 train.csv 里找包含所有密码子的匹配项"""
        
        # 结果存储
        results = []

        for _ in range(num_samples):
            attempts = 0  # 当前尝试次数
            while attempts < max_try:
                # 采样 n 个 example
                sampled_rows = train_df.sample(n)
                example_start_codons = set()
                example_stop_codons = set()
                train_codons = set()

                for _, train_row in sampled_rows.iterrows():
                    train_seq = train_row["seq"]
                    # 提取起始密码子和终止密码子
                    example_start_codons.add(train_seq[:3])
                    example_stop_codons.add(train_seq[-3:])
                    # 提取当前序列的密码子
                    current_codons = set(train_seq[i:i+3] for i in range(0, len(train_seq) - 2, 3))
                    train_codons = train_codons.union(current_codons)

                # 根据 synthesize 参数决定密码表的生成方式
                if synthesize:
                    # 使用随机密码表
                    table = generate_random_codon_table(
                        start_codons=list(example_start_codons),
                        stop_codons=list(example_stop_codons)
                    )
                else:
                    # 使用标准密码表
                    table = Bio.Data.CodonTable.generic_by_id[1] #[random.choice(list(Bio.Data.CodonTable.generic_by_id.keys()))]

                # 检查 example 序列中的启动子和终止子是否在密码表中
                if not all(codon in table.start_codons for codon in example_start_codons):
                    attempts += 1
                    continue
                if not all(codon in table.stop_codons for codon in example_stop_codons):
                    attempts += 1
                    continue

                # 精简密码表，只保留 example 中出现的密码子
                reduced_table = {
                    codon: aa for codon, aa in table.forward_table.items() if codon in train_codons
                }
                reduced_start_codons = list(example_start_codons)
                reduced_stop_codons = list(example_stop_codons)


                if synthesize:
                    results.append({
                        "examples": [{"seq": row["seq"], "aa": str(Seq(row["seq"]).translate(table=table))} for _, row in sampled_rows.iterrows()],
                        "table": {
                            "forward_table": reduced_table,
                            "start_codons": reduced_start_codons,
                            "stop_codons": reduced_stop_codons,
                        },
                    })
                else:
                    results.append({
                        "examples": [{"seq": row["seq"], "aa": row["protein seq"]} for _, row in sampled_rows.iterrows()],
                        "table": {
                            "forward_table": reduced_table,
                            "start_codons": reduced_start_codons,
                            "stop_codons": reduced_stop_codons,
                        },
                    })

                break  # 成功找到匹配后退出尝试循环

            # 如果达到最大尝试次数仍未找到匹配，记录警告
            if attempts == max_try:
                raise ValueError(f"无法在 {max_try} 次尝试内找到匹配的序列，跳过该样本。")

        return results
    @staticmethod
    def process_sequences_synthesize_example(num_samples,  n=3, synthesize=False, max_len=400, min_len=200, max_try=20):
        """从 test.csv 读取 `seq`，在 train.csv 里找包含所有密码子的匹配项"""
        
        # 结果存储
        results = []

        for _ in range(num_samples):
            attempts = 0  # 当前尝试次数
            while attempts < max_try:
                # 根据 synthesize 参数决定密码表的生成方式
                if synthesize:
                    # 使用随机密码表
                    table = generate_random_codon_table(
                        start_codons=list(example_start_codons),
                        stop_codons=list(example_stop_codons)
                    )
                else:
                    # 使用标准密码表
                    table = Bio.Data.CodonTable.generic_by_id[random.choice(list(Bio.Data.CodonTable.generic_by_id.keys()))]
                # 采样 n 个 example
                sampled_rows = generate_example_sequences(table, n=n, max_len=max_len, min_len=min_len)
                example_start_codons = set()
                example_stop_codons = set()
                train_codons = set()

                for train_row in sampled_rows:
                    train_seq = train_row["seq"]
                    # 提取起始密码子和终止密码子
                    example_start_codons.add(train_seq[:3])
                    example_stop_codons.add(train_seq[-3:])
                    # 提取当前序列的密码子
                    current_codons = set(train_seq[i:i+3] for i in range(0, len(train_seq) - 2, 3))
                    train_codons = train_codons.union(current_codons)

                

                # 检查 example 序列中的启动子和终止子是否在密码表中
                if not all(codon in table.start_codons for codon in example_start_codons):
                    attempts += 1
                    continue
                if not all(codon in table.stop_codons for codon in example_stop_codons):
                    attempts += 1
                    continue

                # 精简密码表，只保留 example 中出现的密码子
                reduced_table = {
                    codon: aa for codon, aa in table.forward_table.items() if codon in train_codons
                }
                reduced_start_codons = list(example_start_codons)
                reduced_stop_codons = list(example_stop_codons)

                results.append({
                    "examples": [{"seq": row["seq"], "aa": Seq(row["seq"]).translate(table=table)} for row in sampled_rows],
                    "table": {
                        "forward_table": reduced_table,
                        "start_codons": reduced_start_codons,
                        "stop_codons": reduced_stop_codons,
                    },
                })

                break  # 成功找到匹配后退出尝试循环

            # 如果达到最大尝试次数仍未找到匹配，记录警告
            if attempts == max_try:
                raise ValueError(f"无法在 {max_try} 次尝试内找到匹配的序列，跳过该样本。")

        return results

    @staticmethod
    def load(
        path: str,  # 数据集路径，包含 test 和 train 文件
        icl_num: int = 10,  # ICL 示例数量
        num_samples: int = 100,  # 总样本数
        max_len: int = 400,  # max sequence length
        min_len: int = 200,  # min sequence length
        synthesize_table: bool = False,  # Whether to synthesize codon table, False uses the predefined one in the bio library
        synthesize_example: bool = False,  # Whether to enable synthesis example, if synthesize_table==False and synthesize_example==False, then the codon table can only use the generated codon table in the standard codon table
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
        data_path = os.path.join(path, f"dna_table_infer/data_{icl_num}-shot_test-num={num_samples}-maxlen={max_len}-minlen={min_len}-synthesize-example={synthesize_example}-synthesize-table={synthesize_table}.json")
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                results = json.load(f)
        else:
            if synthesize_example:
                results = DNATableInferDataset.process_sequences_synthesize_example(num_samples,  icl_num, synthesize_table, max_len=max_len, min_len=min_len)
            else:
                results = DNATableInferDataset.process_sequences(num_samples, train_df, icl_num, synthesize_table)
            with open(data_path, 'w') as f:
                json.dump(results, f, indent=4)
                

        # 转换为 HuggingFace Dataset 格式
        data = {'prompt': [], 'answer': []}
        for test_example in results:
            few_shot_examples = [
                f"example DNA {idx+1}:\n{ex['seq']}\nexample Protein {idx+1}:\n{ex['aa']}\n"
                for idx, ex in enumerate(test_example["examples"])
            ]

            test_prompt = template.format(
                example="\n".join(few_shot_examples),
            )
            answer = test_example["table"]
            # 将答案转换为字符串格式
            answer = json.dumps(answer, indent=4)
            data['prompt'].append(test_prompt)
            data['answer'].append(answer)

        return Dataset.from_dict(data)



class DNATableInferEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        """
        计算预测结果的准确率。
        """
        scores = []
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            try:
                # 将参考答案解析为 JSON 格式
                ref_table = json.loads(ref)
                
                try:
                    # 将预测答案解析为 JSON 格式
                    pred_table = json.loads(pred)
                    
                    # 提取密码表、启动子和终止子
                    ref_forward_table = ref_table.get("forward_table", {})
                    ref_start_codons = set(ref_table.get("start_codons", []))
                    ref_stop_codons = set(ref_table.get("stop_codons", []))
                    
                    pred_forward_table = pred_table.get("forward_table", {})
                    pred_start_codons = set(pred_table.get("start_codons", []))
                    pred_stop_codons = set(pred_table.get("stop_codons", []))
                    
                    # 计算密码表正确的数量
                    correct_codons = sum(
                        1 for codon, aa in ref_forward_table.items()
                        if codon in pred_forward_table and pred_forward_table[codon] == aa
                    )
                    
                    # 计算启动子和终止子正确的数量
                    correct_start_codons = len(ref_start_codons & pred_start_codons)
                    correct_stop_codons = len(ref_stop_codons & pred_stop_codons)
                    
                    # 计算总数
                    total_codons = len(ref_forward_table)
                    total_start_codons = len(ref_start_codons)
                    total_stop_codons = len(ref_stop_codons)
                    
                    # 防止除零错误
                    denominator = total_codons + total_start_codons + total_stop_codons
                    if denominator == 0:
                        print(f"Warning: Sample {i} has zero total elements in reference")
                        accuracy = 0.0
                    else:
                        # 计算准确率
                        accuracy = (correct_codons + correct_start_codons + correct_stop_codons) / denominator
                    
                    scores.append(accuracy)
                    
                except json.JSONDecodeError as e:
                    print(f"Error: Failed to parse prediction JSON for sample {i}: {e}")
                    scores.append(0.0)  # 预测解析失败时分配零分
                except Exception as e:
                    print(f"Error processing prediction for sample {i}: {e}")
                    scores.append(0.0)  # 处理预测时出错分配零分
                    
            except json.JSONDecodeError as e:
                print(f"Error: Failed to parse reference JSON for sample {i}: {e}")
                scores.append(0.0)  # 参考解析失败时分配零分
            except Exception as e:
                print(f"Error processing reference for sample {i}: {e}")
                scores.append(0.0)  # 处理参考时出错分配零分

        # 处理空列表情况
        if not scores:
            return {'accuracy': 0.0}
            
        # 返回平均准确率
        return {'accuracy': round(sum(scores) / len(scores) * 100, 2)}


class DNATableInferHREvaluator():
    def score(self, predicted_table_json, examples):
        """
        Evaluate a codon table by testing how accurately it translates DNA sequences from examples.
        
        Args:
            predicted_table_json (str): JSON string of the predicted codon table
            examples (list): List of dict with "seq" (DNA) and "aa" (protein) keys
            
        Returns:
            dict: Dictionary with accuracy metrics
        """
        try:
            # Parse the predicted codon table
            pred_table = json.loads(predicted_table_json)
            
            # Extract components
            forward_table = pred_table.get("forward_table", {})
            start_codons = set(pred_table.get("start_codons", []))
            stop_codons = set(pred_table.get("stop_codons", []))
            
            total_matches = 0
            total_aa = 0
            translations = []
            
            for example in examples:
                dna_seq = example["seq"]
                expected_protein = example["aa"]
                
                # Translate the DNA using the predicted table
                translated_protein = self.translate_dna(dna_seq, forward_table, start_codons, stop_codons)
                
                # Count matching amino acids
                min_len = min(len(translated_protein), len(expected_protein))
                matches = sum(translated_protein[i] == expected_protein[i] for i in range(min_len))
                
                # Count length differences as mismatches
                total_matches += matches
                total_aa += max(len(translated_protein), len(expected_protein))

                translations.append(translated_protein)
            
            # Calculate accuracy
            if total_aa == 0:
                return {"accuracy": 0.0, "translations": translations}
            
            accuracy = total_matches / total_aa
            return {"accuracy": round(accuracy * 100, 2), "translations": translations}
            
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse prediction JSON: {e}")
            return {"accuracy": 0.0, "translations": []}
        except Exception as e:
            print(f"Error processing prediction: {e}")
            return {"accuracy": 0.0, "translations": []}

    def translate_dna(self, dna_seq, forward_table, start_codons, stop_codons):
        """
        Translate a DNA sequence using the provided codon table.
        
        Args:
            dna_seq (str): DNA sequence to translate
            forward_table (dict): Mapping from codons to amino acids
            start_codons (set): Set of start codons
            stop_codons (set): Set of stop codons
        
        Returns:
            str: Translated protein sequence
        """
        protein = []
        
        # Handle start codon (first three nucleotides)
        start_codon = dna_seq[:3]
        if start_codon in start_codons:
            protein.append('M')  # Start codons typically code for Methionine
        elif start_codon in forward_table:
            protein.append(forward_table[start_codon])
        else:
            protein.append('X')  # Unknown codon
        
        # Translate the middle section
        for i in range(3, len(dna_seq), 3):
            codon = dna_seq[i:i+3]
            if codon in stop_codons:
                protein.append('*')
                break  # Stop translation when we hit a stop codon
            if codon in forward_table:
                protein.append(forward_table[codon])
            else:
                protein.append('X')  # Unknown codon
        
        # Note: We don't translate the stop codon
        
        return ''.join(protein)