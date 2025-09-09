import json
import os
import random
import numpy as np
import pandas as pd
from datasets import Dataset
from opencompass.datasets.base import BaseDataset
from opencompass.datasets.sirbenchv1.chem_utils import get_scaffold_fp, top_n_scaffold_similar_molecules
from opencompass.utils import get_data_path
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET
from rdkit import Chem


def create_prompt(input_text, examples):
    prompt = "You are an expert chemist. Given the reactants SMILES, your task is to predict the main product SMILES using your experienced chemical Reaction Prediction knowledge. \n\
Please strictly follow the format, no other information can be provided. You should only reply with SMILES string notations to represent the product. The input contains the reactants and reagents which are split by '.'. The product smiles must be valid and chemically reasonable. \n"

    for example in examples:
        prompt += f"Reactants+Reagents: {example[0]}\nProducts: {example[1]}\n"
    prompt += f"Reactants+Reagents: {input_text}\nProducts:"
    return prompt

@LOAD_DATASET.register_module()
class ChemReactionPredictionDataset(BaseDataset):


    @staticmethod
    def load(
        path: str,  # 数据集路径，包含 test 和 train 文件
        icl_num: int = 10,  # ICL 示例数量
        num_samples: int = 100,  # 总样本数
        random_seed: int = 42,  # 随机种子
    ) -> Dataset:
        """
        加载 DNA 数据集并生成 few-shot 数据。
        """

        random.seed(random_seed)
        np.random.seed(random_seed)
        path = get_data_path(path)
        save_path = os.path.join(path, f"chem_reaction_prediction/chem_reaction_prediction-{icl_num}-shot-{num_samples}-samples.json")
        if os.path.exists(save_path):
            with open(save_path, 'r') as f:
                data = json.load(f)
            return Dataset.from_dict(data)
        # 定义测试集和训练集路径
        data_path = os.path.join(path, "chem_reaction_prediction/uspto_mixed.pickle")
        df = pd.read_pickle(data_path)
        df['reactants_smiles'] = df['reactants_mol'].apply(
            lambda x: Chem.MolToSmiles(x))
        df['products_smiles'] = df['products_mol'].apply(lambda x: Chem.MolToSmiles(x))
        train = df[df['set'] == 'train']
        # valid = df[df['set'] == 'valid']
        test = df[df['set'] == 'test']

        test = test.sample(num_samples, random_state=random_seed)
        train['scaffold_fp'] = train['reactants_smiles'].apply(
            lambda x: get_scaffold_fp(x))
        
        test = test.reset_index()
        # test = test.head(num_samples)




        # 转换为 HuggingFace Dataset 格式
        data = {'prompt': [], 'answer': []}
        for idx, row in test.iterrows():
            reactant = row['reactants_smiles']
            product = row['products_smiles']
            index = row['index']

            sim = top_n_scaffold_similar_molecules(reactant, list(
                train['scaffold_fp']), list(train['reactants_smiles']), n=icl_num)
            chunk = train[train['reactants_smiles'].isin(sim)]
            examples = list(
                zip(chunk["reactants_smiles"].values, chunk["products_smiles"].values))

            # build prompt and save
            prompt = create_prompt(reactant, examples)
            


            data['prompt'].append(prompt)
            data['answer'].append(product)
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=4)
        return Dataset.from_dict(data)



class ChemReactionPredictionEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        """
        计算预测结果的准确率。
        """

        correct = 0
        for pred, ref in zip(predictions, references):
            try:
                mol = Chem.MolFromSmiles(pred)
                pred = Chem.MolToSmiles(mol)
            except Exception as e:
                continue

            pred_list = pred.split(".")
            if ref in pred_list:
                correct += 1


        # 返回平均准确率
        return {'accuracy': round(correct / len(references) * 100, 2)}
    


#@EVALUATOR.register_module()
class NewChemReactionPredictionEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        correct = 0
        total = 0
        invalid = 0

        for pred, ref in zip(predictions, references):
            try:
                mol_pred = Chem.MolFromSmiles(pred)
                mol_ref = Chem.MolFromSmiles(ref)
                
                if mol_pred is None or mol_ref is None:
                    invalid += 1
                    continue

                norm_pred = Chem.MolToSmiles(mol_pred, canonical=True)
                norm_ref = Chem.MolToSmiles(mol_ref, canonical=True)

                if norm_pred == norm_ref:
                    correct += 1
                total += 1

            except Exception:
                invalid += 1
        
        return {'accuracy': round(correct / total * 100, 2) if total > 0 else 0.0}