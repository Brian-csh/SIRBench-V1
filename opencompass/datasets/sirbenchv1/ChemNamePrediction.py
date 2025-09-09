import json
import os
import random
import numpy as np
import pandas as pd
from datasets import Dataset
from opencompass.datasets.base import BaseDataset
from opencompass.datasets.sirbenchv1.chem_utils import get_scaffold_fp, molToCanonical, top_n_scaffold_similar_molecules, top_n_similar_strings
from opencompass.utils import get_data_path
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET
from Bio.Data.CodonTable import NCBICodonTableDNA
from Bio.Seq import Seq
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from rdkit import Chem
def create_prompt_smiles2iupac(input_text, examples):
    prompt = "You are an expert chemist. Given the molecular SMILES, your task is to predict the IUPAC name using your experienced chemical IUPAC name knowledge. \n\
Please strictly follow the format, no other information can be provided.\n"

    for example in examples:
        prompt += f"Molecular SMILES: {example[0]}\nMolecular IUPAC name: {example[1]}\n"
    prompt += f"Molecular SMILES: {input_text}\nMolecular IUPAC name:"
    return prompt


def create_prompt_iupac2smiles(input_text, examples):
    prompt = "You are an expert chemist. Given the molecular IUPAC name, your task is to predict the molecular SMILES using your experienced chemical IUPAC name knowledge. \n\
Please strictly follow the format, no other information can be provided. You should only reply with molecular SMILES string notations to represent the IUPAC name. The SMILES must be valid and chemically reasonable. \n"
    for example in examples:
        prompt += f"Molecular IUPAC name: {example[0]}\nMolecular SMILES: {example[1]}\n"
    prompt += f"Molecular IUPAC name: {input_text}\nMolecular SMILES:"
    return prompt


def create_prompt_smiles2formula(input_text, examples):
    prompt = "You are an expert chemist. Given the molecular SMILES, your task is to predict the molecular formula using your experienced chemical molecular formula knowledge. \n\
Please strictly follow the format, no other information can be provided.\n"

    for example in examples:
        prompt += f"Molecular SMILES: {example[0]}\nMolecular formula: {example[1]}\n"
    prompt += f"Molecular SMILES: {input_text}\nMolecular formula:"
    return prompt


def create_prompt_iupac2formula(input_text, examples):
    prompt = "You are an expert chemist. Given the molecular IUPAC name, your task is to predict the molecular formula using your experienced chemical molecular formula knowledge. \n\
Please strictly follow the format, no other information can be provided.\n"
    for example in examples:
        prompt += f"Molecular IUPAC name: {example[0]}\nMolecular formula: {example[1]}\n"
    prompt += f"Molecular IUPAC name: {input_text}\nMolecular formula:"
    return prompt

def get_input_output_columns_by_task(task):
    if task == 'smiles2iupac':
        return "smiles", "iupac"
    elif task == 'smiles2formula':
        return "smiles", "formula"
    elif task == 'iupac2smiles':
        return "iupac", "smiles"
    elif task == 'iupac2formula':
        return "iupac", 'formula'


def create_prompt(reactant, examples, task):
    if task == 'smiles2iupac':
        return create_prompt_smiles2iupac(reactant, examples)
    elif task == 'smiles2formula':
        return create_prompt_smiles2formula(reactant, examples)
    elif task == 'iupac2smiles':
        return create_prompt_iupac2smiles(reactant, examples)
    elif task == 'iupac2formula':
        return create_prompt_iupac2formula(reactant, examples)
    
@LOAD_DATASET.register_module()
class ChemNamePredictionDataset(BaseDataset):

    @staticmethod
    def load(
        path: str,  # 数据集路径，包含 test 和 train 文件
        icl_num: int = 10,  # ICL 示例数量
        num_samples: int = 100,  # 总样本数
        task: str = "smiles2iupac",  # 任务类型
        random_seed: int = 42,  # 随机种子
    ) -> Dataset:
        """
        加载 DNA 数据集并生成 few-shot 数据。
        """

        random.seed(random_seed)
        np.random.seed(random_seed)
        path = get_data_path(path)
        save_path = os.path.join(path, f"chem_name_prediction/chem_name_prediction-{icl_num}-shot-{num_samples}-samples-{task}.json")
        if os.path.exists(save_path):
            with open(save_path, 'r') as f:
                data = json.load(f)
            return Dataset.from_dict(data)
        # 定义测试集和训练集路径
        data_path = os.path.join(path, "chem_name_prediction/llm_test.csv")
        df = pd.read_csv(data_path)
        df = df[~df['iupac'].isna()]
        train, test = train_test_split(df, test_size=num_samples+50 #多一些,防止None
                                       , random_state=random_seed)
        train['scaffold_fp'] = train['smiles'].apply(lambda x: get_scaffold_fp(x))
        train = train[~train['scaffold_fp'].isna()]

        # valid['smiles'] = valid['smiles'].apply(lambda x: molToCanonical(x))
        train['smiles'] = train['smiles'].apply(lambda x: molToCanonical(x))
        test['smiles'] = test['smiles'].apply(lambda x: molToCanonical(x))
        train = train[~train['smiles'].isna()]
        test = test[~test['smiles'].isna()]
        test = test.reset_index()
        test = test.head(num_samples) #!

        input_col, output_col = get_input_output_columns_by_task(task)

        # 转换为 HuggingFace Dataset 格式
        data = {'prompt': [], 'answer': []}
        for idx, row in test.iterrows():
            reactant = row[input_col]
            product = row[output_col]
            index = row['index']
            if input_col == 'smiles':
                sim = top_n_scaffold_similar_molecules(reactant, list(
                    train['scaffold_fp']), list(train['smiles']), n=icl_num)
            else:
                # similarity by leven
                sim = top_n_similar_strings(
                    reactant, list(train[input_col]), n=icl_num)
            chunk = train[train[input_col].isin(sim)]
            examples = list(
                zip(chunk[input_col].values, chunk[output_col].values))
            prompt = create_prompt(reactant, examples, task)
            

            data['prompt'].append(prompt)
            data['answer'].append(product)

        with open(save_path, 'w') as f:
            json.dump(data, f)
        return Dataset.from_dict(data)



# class ChemNamePredictionEvaluator(BaseEvaluator):

#     def score(self, predictions, references,task):
#         """
#         计算预测结果的准确率。
#         """
#         # import debugpy
#         # try:
#         #     debugpy.listen(("localhost", 3000))
#         #     print("Waiting for debugger attach")
#         #     debugpy.wait_for_client()
#         # except Exception as e:
#         #     pass
#         correct = 0
#         for pred, ref in zip(predictions, references):
#             if task in ['iupac2smiles', 'formula2smiles']:
#                 try:
#                     mol = Chem.MolFromSmiles(pred)
#                     pred = Chem.MolToSmiles(mol)
#                 except Exception as e:
#                     continue
#             if ref == pred:
#                 correct += 1


#         # 返回平均准确率
#         return {'accuracy': round(correct / len(ref) * 100, 2)}

class ChemNamePredictionSimlesEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        """
        计算预测结果的准确率。,对于task = ['iupac2smiles', 'formula2smiles']
        """

        correct = 0
        for pred, ref in zip(predictions, references):

            try:
                mol = Chem.MolFromSmiles(pred)
                pred = Chem.MolToSmiles(mol)
            except Exception as e:
                continue
            if ref == pred:
                correct += 1


        # 返回平均准确率
        return {'accuracy': round(correct / len(references) * 100, 2)}

# class ChemNamePredictionSimlesEvaluator(BaseEvaluator):
#     def score(self, predictions, references):
#         """
#         计算预测结果的准确率。适用于 ['iupac2smiles']。
#         本版本对预测值和参考答案都做了 SMILES 标准化。
#         """
#         correct = 0
#         total = 0

#         for pred, ref in zip(predictions, references):
#             try:
#                 mol_pred = Chem.MolFromSmiles(pred)
#                 mol_ref = Chem.MolFromSmiles(ref)

#                 if mol_pred is None or mol_ref is None:
#                     continue  # 跳过非法结构

#                 norm_pred = Chem.MolToSmiles(mol_pred, canonical=True)
#                 norm_ref = Chem.MolToSmiles(mol_ref, canonical=True)

#                 if norm_pred == norm_ref:
#                     correct += 1
#                 total += 1
#             except Exception:
#                 continue

#         return {'accuracy': round(correct / total * 100, 2) if total > 0 else 0.0}
    

