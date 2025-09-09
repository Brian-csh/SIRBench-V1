import json
import os
import random
import statistics
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
import csv
from nltk.translate.bleu_score import corpus_bleu
from Levenshtein import distance as lev
from fcd import get_fcd, load_ref_model, canonical_smiles
from transformers import BertTokenizerFast

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from rdkit.Chem import AllChem
import openai
import re
from openai import OpenAI

client_gpt = OpenAI(
  api_key='sk-GUxChNMVpEQCTVJAE68320C49b9a499d9f6cD868B379006e',
  #api_key='sk-yNNThZD66bREqZwy487c856805C74c90A7497d22Fc44DeEa',
  base_url='https://yeysai.com/v1/'
)
def evaluate1(outputs, verbose=False):
#     outputs = []

#     with open(osp.join(input_fp)) as f:
#         reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
#         for n, line in enumerate(reader):
#             gt_smi = line['ground truth']
#             ot_smi = line['output']
#             outputs.append((line['description'], gt_smi, ot_smi))


    bleu_scores = []
    #meteor_scores = []

    references = []
    hypotheses = []

    for i, (smi, gt, out) in enumerate(outputs):

        if i % 100 == 0:
            if verbose:
                print(i, 'processed.')


        gt_tokens = [c for c in gt]

        out_tokens = [c for c in out]

        references.append([gt_tokens])
        hypotheses.append(out_tokens)

        # mscore = meteor_score([gt], out)
        # meteor_scores.append(mscore)

    # BLEU score
    bleu_score = corpus_bleu(references, hypotheses)
    if verbose: print('BLEU score:', bleu_score)

    # Meteor score
    # _meteor_score = np.mean(meteor_scores)
    # print('Average Meteor score:', _meteor_score)

    rouge_scores = []

    references = []
    hypotheses = []

    levs = []

    num_exact = 0

    bad_mols = 0

    for i, (smi, gt, out) in enumerate(outputs):

        hypotheses.append(out)
        references.append(gt)

        try:
            m_out = Chem.MolFromSmiles(out)
            m_gt = Chem.MolFromSmiles(gt)

            if Chem.MolToInchi(m_out) == Chem.MolToInchi(m_gt): num_exact += 1
            #if gt == out: num_exact += 1 #old version that didn't standardize strings
        except:
            bad_mols += 1

        

        levs.append(lev(out, gt))


    # Exact matching score
    exact_match_score = num_exact/(i+1)
    if verbose:
        print('Exact Match:')
        print(exact_match_score)

    # Levenshtein score
    levenshtein_score = np.mean(levs)
    if verbose:
        print('Levenshtein:')
        print(levenshtein_score)
        
    validity_score = 1 - bad_mols/len(outputs)
    if verbose:
        print('validity:', validity_score)

    return bleu_score, exact_match_score, levenshtein_score, validity_score



def evaluate2(raw_outputs, morgan_r=2, verbose=False):
    bad_mols = 0
    outputs = []

    for desc, gt_smi, ot_smi in raw_outputs:
        try:
            gt_m = Chem.MolFromSmiles(gt_smi)
            ot_m = Chem.MolFromSmiles(ot_smi)

            if ot_m == None: raise ValueError('Bad SMILES')
            outputs.append((desc, gt_m, ot_m))
        except:
            bad_mols += 1
    validity_score = len(outputs)/(len(outputs)+bad_mols)
    if verbose:
        print('validity:', validity_score)


    MACCS_sims = []
    morgan_sims = []
    RDK_sims = []

    enum_list = outputs

    for i, (desc, gt_m, ot_m) in enumerate(enum_list):

        if i % 100 == 0:
            if verbose: print(i, 'processed.')

        MACCS_sims.append(DataStructs.FingerprintSimilarity(MACCSkeys.GenMACCSKeys(gt_m), MACCSkeys.GenMACCSKeys(ot_m), metric=DataStructs.TanimotoSimilarity))
        RDK_sims.append(DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(gt_m), Chem.RDKFingerprint(ot_m), metric=DataStructs.TanimotoSimilarity))
        morgan_sims.append(DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(gt_m,morgan_r), AllChem.GetMorganFingerprint(ot_m, morgan_r)))

    maccs_sims_score = np.mean(MACCS_sims)
    rdk_sims_score = np.mean(RDK_sims)
    morgan_sims_score = np.mean(morgan_sims)
    if verbose:
        print('Average MACCS Similarity:', maccs_sims_score)
        print('Average RDK Similarity:', rdk_sims_score)
        print('Average Morgan Similarity:', morgan_sims_score)
    return validity_score, maccs_sims_score, rdk_sims_score, morgan_sims_score



def evaluate3(gt_smis, ot_smis, verbose=False):

    model = load_ref_model()

    canon_gt_smis = [w for w in canonical_smiles(gt_smis) if w is not None]
    canon_ot_smis = [w for w in canonical_smiles(ot_smis) if w is not None]

    # ⚠️ 防止空列表报错
    if len(canon_gt_smis) == 0 or len(canon_ot_smis) == 0:
        if verbose:
            print("Warning: Empty SMILES after canonicalization. Returning score = 0.")
        return 0.0
    
    fcd_sim_score = get_fcd(canon_gt_smis, canon_ot_smis, model)
    if verbose:
        print('FCD Similarity:', fcd_sim_score)

    return fcd_sim_score


def evaluate_one_molecule_design(details_df):
    print(details_df[['SMILES', 'pred']])

    tem = list(zip(details_df['description'], details_df['SMILES'], details_df['pred']))

    
    bleu_score, exact_match_score, levenshtein_score, _ = evaluate1(tem)
    validity_score, maccs_sims_score, rdk_sims_score, morgan_sims_score = evaluate2(tem)
    fcd = evaluate3(list(details_df['SMILES']), list(details_df['pred']))
    

    
    return { "bleu": bleu_score, "exact_match": exact_match_score, "levenshtein": levenshtein_score, "validity": validity_score, "maccs_sims": maccs_sims_score, "rdk_sims": rdk_sims_score, "morgan_sims": morgan_sims_score, "fcd": fcd }
def evaluate4(raw_outputs, text_model='allenai/scibert_scivocab_uncased', text_trunc_length=512):
    outputs = []
    
    for smiles, gt, output in raw_outputs:
        out_tmp = output[6:] if output.startswith('[CLS] ') else output
        outputs.append((smiles, gt, out_tmp))

    text_tokenizer = BertTokenizerFast.from_pretrained(text_model)

    bleu_scores = []
    meteor_scores = []

    references = []
    hypotheses = []

    for i, (smi, gt, out) in enumerate(outputs):

        if i % 100 == 0: pass


        gt_tokens = text_tokenizer.tokenize(gt, truncation=True, max_length=text_trunc_length,
                                            padding='max_length')
        gt_tokens = list(filter(('[PAD]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[CLS]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[SEP]').__ne__, gt_tokens))

        out_tokens = text_tokenizer.tokenize(out, truncation=True, max_length=text_trunc_length,
                                            padding='max_length')
        out_tokens = list(filter(('[PAD]').__ne__, out_tokens))
        out_tokens = list(filter(('[CLS]').__ne__, out_tokens))
        out_tokens = list(filter(('[SEP]').__ne__, out_tokens))

        references.append([gt_tokens])
        hypotheses.append(out_tokens)

        mscore = meteor_score([gt_tokens], out_tokens)
        meteor_scores.append(mscore)

    bleu2 = corpus_bleu(references, hypotheses, weights=(.5,.5))
    bleu4 = corpus_bleu(references, hypotheses, weights=(.25,.25,.25,.25))

#     print('BLEU-2 score:', bleu2)
#     print('BLEU-4 score:', bleu4)
    _meteor_score = np.mean(meteor_scores)
#     print('Average Meteor score:', _meteor_score)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    rouge_scores = []

    references = []
    hypotheses = []

    for i, (smi, gt, out) in enumerate(outputs):

        rs = scorer.score(out, gt)
        rouge_scores.append(rs)

#     print('ROUGE score:')
    rouge_1 = np.mean([rs['rouge1'].fmeasure for rs in rouge_scores])
    rouge_2 = np.mean([rs['rouge2'].fmeasure for rs in rouge_scores])
    rouge_l = np.mean([rs['rougeL'].fmeasure for rs in rouge_scores])
#     print('rouge1:', rouge_1)
#     print('rouge2:', rouge_2)
#     print('rougeL:', rouge_l)
    return bleu2, bleu4, rouge_1, rouge_2, rouge_l, _meteor_score

def evaluate_one_molecule_captioning(details_df):
    
    tem = list(zip(details_df['SMILES'], details_df['description'], details_df['pred']))
    bleu2, bleu4, rouge_1, rouge_2, rouge_l, _meteor_score = evaluate4(tem)
    llm_score = evaluate4_llm(tem, model_call_fn=call_gpt)
    return { "bleu2": bleu2, "bleu4": bleu4, "rouge_1": rouge_1, "rouge_2": rouge_2, "rouge_l": rouge_l, "meteor_score": _meteor_score, "llm_score":llm_score}

def evaluate4_llm(raw_outputs, model_call_fn):
    results = []
    print(raw_outputs)
    for smiles, gt, pred in raw_outputs:
        prompt = f"""You are an expert molecular biologist.

Below is a SMILES string representing a molecule:  
{smiles}

Here is a reference description of the molecule:
"{gt}"

Here is a predicted description of the same molecule:
"{pred}"

Your task is to evaluate the **predicted** description **only** based on its scientific quality compared to the reference.

You must assign a **score from 1 to 10** based on the following criteria:

- **Score 10**: Nearly perfect — scientifically precise, complete, and fluent. Matches all key aspects of the reference (e.g., functional groups, chemical class, derivation, roles).
- **Score 8–9**: Very good — minor omissions or slight rewording, but the core structure-level and functional meaning is intact.
- **Score 6–7**: Reasonable — generally correct but may lack specific details (e.g., derivation or one functional role). Possibly vague phrasing.
- **Score 4–5**: Partial — captures the general category or one function but omits multiple important details or shows misunderstanding in phrasing.
- **Score 2–3**: Poor — vague, generic, or scientifically weak. May refer to the wrong compound type or confuse structural features.
- **Score 1**: Completely incorrect or irrelevant.

Only output a **single line** in the following format:  
Score: [1-10]
"""
        response = model_call_fn(prompt)
        print(f"+++{response}+++")
        match = re.search(r"Score:\s*(\d+)", response)
        score = int(match.group(1)) if match else 0

        results.append(score)
    print(f"-----results:{results}-------")
    
    return np.mean(results) 


def call_gpt(prompt):
    response = client_gpt.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
   
def read_data(filename):
    # Open the file for reading
    with open(filename, 'r') as f:
        # Create a CSV reader with tab delimiter
        reader = csv.reader(f, delimiter='\t')
        # Read the data into a list of tuples
        data = [tuple(row) for row in reader]
    df = pd.DataFrame(data[1:], columns=data[0])
    return df
def create_prompt_molecule_design(input_text, examples):
    prompt = "You are an expert chemist. Given the molecular requirements description, your task is to design a new molecule using your experienced chemical Molecular Design knowledge. \n\
Please strictly follow the format, no other information can be provided. You should only reply with SMILES \
string notations to represent the designed molecule. The SMILES must be valid and chemically reasonable. \n"
    
    for example in examples:
        prompt += f"Molecular requirements description: {example[0]}\nMolecular SMILES: {example[1]}\n"
    prompt += f"Molecular requirements description: {input_text}\nMolecular SMILES:"
    return prompt

def create_prompt_molecule_captioning(input_text, examples):
    prompt = "You are an expert chemist. Given the molecular SMILES, your task is to provide the detailed description of the molecule using your experienced chemical Molecular knowledge. \n\
Please strictly follow the format, no other information can be provided.\n"
    
    for example in examples:
        prompt += f"Molecule SMILES: {example[0]}\nMolecular Description: {example[1]}\n"
    prompt += f"Molecule SMILES: {input_text}\nMolecular Description:"
    return prompt

def get_input_output_columns_by_task(task):
    if task == 'molecule_design':
        return "description", "SMILES"
    elif task == 'molecule_captioning':
        return "SMILES", "description"
def create_prompt(reactant, examples, task):
    if task == 'molecule_design':
        return create_prompt_molecule_design(reactant, examples)
    elif task == 'molecule_captioning':
        return create_prompt_molecule_captioning(reactant, examples) 

@LOAD_DATASET.register_module()
class ChemMoleculeDesignDataset(BaseDataset):


    @staticmethod
    def load(
        path: str,  # 数据集路径，包含 test 和 train 文件
        icl_num: int = 10,  # ICL 示例数量
        num_samples: int = 100,  # 总样本数
        random_seed: int = 42,  # 随机种子
        task: str = 'molecule_design'  # 任务类型
    ) -> Dataset:




        random.seed(random_seed)
        np.random.seed(random_seed)
        path = get_data_path(path)
        save_path = os.path.join(path, f"chem_molecule_design/chem_molecule-{icl_num}-shot-{num_samples}-samples-{task}.json")
        if os.path.exists(save_path):
            with open(save_path, 'r') as f:
                data = json.load(f)
            return Dataset.from_dict(data)
        # 定义测试集和训练集路径
        data_path = os.path.join(path, "chem_molecule_design/ChEBI-20_data")

        train = read_data(os.path.join(data_path,"train.txt"))
        # valid = read_data(os.path.join(data_path, "validation.txt"))
        test = read_data(os.path.join(data_path, "test.txt"))

        test = test.sample(num_samples+50, random_state=random_seed) #!
        train['SMILES'] = train['SMILES'].apply(molToCanonical)
        test['SMILES'] = test['SMILES'].apply(molToCanonical)
        test = test[~test['SMILES'].isna()]
        train = train[~train['SMILES'].isna()]
        train['scaffold_fp'] = train['SMILES'].apply(lambda x: get_scaffold_fp(x))
        train = train[~train['scaffold_fp'].isna()]
        
        test = test.reset_index()
        test = test.head(num_samples)

        input_col, output_col = get_input_output_columns_by_task(task)
        train = train[~train[input_col].isna()]


        # 转换为 HuggingFace Dataset 格式
        data = {'prompt': [], 'answer': []}
        for idx, row in test.iterrows():
            reactant = row[input_col]
            product = row[output_col]


            if input_col == 'SMILES':
                
                sim = top_n_scaffold_similar_molecules(reactant, list(train['scaffold_fp']), list(train['SMILES']), n=icl_num)
            else:
                # similarity by leven
                sim = top_n_similar_strings(reactant, list(train[input_col]), n=icl_num)
            chunk = train[train[input_col].isin(sim)]
            
            examples = list(zip(chunk[input_col].values, chunk[output_col].values))

            # build prompt and save
            prompt = create_prompt(reactant, examples, task)
            


            data['prompt'].append(prompt)
            data['answer'].append(reactant+'-->'+product)
        with open(save_path, 'w') as f:
            json.dump(data, f)
        return Dataset.from_dict(data)



class ChemMoleculeDesignEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        """
        计算预测结果的准确率。
        """

        input_col, output_col = get_input_output_columns_by_task('molecule_design')
        
       

        details_results = []
        for pred, ref in zip(predictions, references):
            reactant, product = ref.split('-->')
            details_results.append([reactant]+[product]+[pred])
        details_df = pd.DataFrame(details_results, columns=[input_col, output_col, 'pred'])
        metrics = evaluate_one_molecule_design(details_df)

        return metrics
    
class ChemMoleculeCaptionEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        """
        计算预测结果的准确率。
        """

        input_col, output_col = get_input_output_columns_by_task('molecule_captioning')
        
       

        details_results = []
        for pred, ref in zip(predictions, references):
            reactant, product = ref.split('-->')
            details_results.append([reactant]+[product]+[pred])
        details_df = pd.DataFrame(details_results, columns=[input_col, output_col, 'pred'])
        metrics = evaluate_one_molecule_captioning(details_df)

        return metrics