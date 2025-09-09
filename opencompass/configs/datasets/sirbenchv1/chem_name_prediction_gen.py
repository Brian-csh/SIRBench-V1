from opencompass.datasets.sirbenchv1.ChemNamePrediction import ChemNamePredictionDataset, ChemNamePredictionSimlesEvaluator
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.sirbenchv1.utils import remove_think_tags, formula_postprocess
from opencompass.openicl.icl_inferencer import HRInferencer
from opencompass.openicl.icl_inferencer import NEWSCInferencer

# CWE Dataset
smiles2iupac_datasets = [
    {
        'abbr': 'chem_smiles2iupac',
        'path': 'opencompass/sirbenchv1',
        'type': ChemNamePredictionDataset,
        'icl_num': 5, # number of icl examples
        'num_samples': 30,  # number of test samples
        'task': 'smiles2iupac',
        'reader_cfg': dict(input_columns=['prompt'], output_column='answer'),
        'infer_cfg': dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=[
                        dict(role='HUMAN', prompt='{prompt}'),
                        dict(role='BOT', prompt='{answer}\n'),
                    ]
                ),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
            # inferencer=dict(type=NEWSCInferencer, max_out_len=16384, infer_type='sc', sc_size = 1,
            # generation_kwargs=dict(
            #             do_sample=True,  # Enable sampling
            #             temperature=1,  ),),
        ),
        'eval_cfg': dict(
            evaluator=dict(type=AccEvaluator),
            pred_postprocessor=dict(type=remove_think_tags),
        ),
        
    }
]
smiles2formula_datasets = [
    {
        'abbr': 'chem_smiles2formula',
        'path': 'opencompass/sirbenchv1',
        'type': ChemNamePredictionDataset,
        'icl_num': 5, # number of icl examples
        'num_samples': 30,  # number of test samples
        'task': 'smiles2formula',
        'reader_cfg': dict(input_columns=['prompt'], output_column='answer'),
        'infer_cfg': dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=[
                        dict(role='HUMAN', prompt='{prompt}'),
                        dict(role='BOT', prompt='{answer}\n'),
                    ]
                ),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
            # inferencer=dict(type=NEWSCInferencer, max_out_len=16384, infer_type='sc', sc_size = 1, 
            # generation_kwargs=dict(
            #             do_sample=True,  # Enable sampling
            #             temperature=1,  ),),
        ),
        'eval_cfg': dict(
            evaluator=dict(type=AccEvaluator),
            pred_postprocessor=dict(type=formula_postprocess),
        ),
        
    }
]
iupac2smiles_datasets = [
    {
        'abbr': 'chem_iupac2smiles',
        'path': 'opencompass/sirbenchv1',
        'type': ChemNamePredictionDataset,
        'icl_num': 5, # number of icl examples
        'num_samples': 30,  # number of test samples
        'task': 'iupac2smiles',
        'reader_cfg': dict(input_columns=['prompt'], output_column='answer'),
        'infer_cfg': dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=[
                        dict(role='HUMAN', prompt='{prompt}'),
                        dict(role='BOT', prompt='{answer}\n'),
                    ]
                ),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
            # inferencer=dict(type=NEWSCInferencer, max_out_len=16384, infer_type='sc', sc_size = 1, 
            # generation_kwargs=dict(
            #             do_sample=True,  # Enable sampling
            #             temperature=1,  ),),
        ),
        'eval_cfg': dict(
            evaluator=dict(type=ChemNamePredictionSimlesEvaluator),
            pred_postprocessor=dict(type=remove_think_tags),
        ),
        
    }
]
iupac2formula_datasets = [
    {
        'abbr': 'chem_iupac2formula',
        'path': 'opencompass/sirbenchv1',
        'type': ChemNamePredictionDataset,
        'icl_num': 5, # number of icl examples
        'num_samples': 30,  # number of test samples
        'task': 'iupac2formula',
        'reader_cfg': dict(input_columns=['prompt'], output_column='answer'),
        'infer_cfg': dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=[
                        dict(role='HUMAN', prompt='{prompt}'),
                        dict(role='BOT', prompt='{answer}\n'),
                    ]
                ),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
            # inferencer=dict(type=NEWSCInferencer, max_out_len=16384, infer_type='sc', sc_size = 1,
            # generation_kwargs=dict(
            #             do_sample=True,  # Enable sampling
            #             temperature=1,  ),),
        ),
        'eval_cfg': dict(
            evaluator=dict(type=AccEvaluator),
            pred_postprocessor=dict(type=formula_postprocess),
        ),
        
    }
]


smiles2iupac_hr_datasets = [
    {
        'abbr': 'chem_smiles2iupac',
        'path': 'opencompass/sirbenchv1',
        'type': ChemNamePredictionDataset,
        'icl_num': 5, # number of icl examples
        'num_samples': 30,  # number of test samples
        'task': 'smiles2iupac',
        'reader_cfg': dict(input_columns=['prompt', 'icl_in', 'icl_out'], output_column='answer'),
        'infer_cfg': dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=[
                        dict(role='HUMAN', prompt='{prompt}'),
                        dict(role='BOT', prompt='{answer}\n'),
                    ]
                ),
            ),
            retriever=dict(type=ZeroRetriever),
            # inferencer=dict(type=GenInferencer),
            inferencer=dict(
                type=HRInferencer,
                max_out_len=20000,
                mode='truncate',
                max_iterations=3,
                n_hypotheses=5,
                icl_in_regex = r"Molecular SMILES:\s*([^\n]+)",
                icl_out_regex = r"Molecular IUPAC name:\s*([^\n]+)",
                task_name = 'chem_smiles2iupac',
                task_description='Given the molecular SMILES, your task is to predict the IUPAC name.',
                generation_kwargs=dict(
                    do_sample=True,  # Enable sampling
                    temperature=1,  # Set temperature for diversity
                ),
            ),
        ),
        'eval_cfg': dict(
            evaluator=dict(type=AccEvaluator),
            pred_postprocessor=dict(type=remove_think_tags),
        ),
        
    }
]

smiles2formula_hr_datasets = [
    {
        'abbr': 'chem_smiles2formula',
        'path': 'opencompass/sirbenchv1',
        'type': ChemNamePredictionDataset,
        'icl_num': 5, # number of icl examples
        'num_samples': 30,  # number of test samples
        'task': 'smiles2formula',
        'reader_cfg': dict(input_columns=['prompt', 'icl_in', 'icl_out'], output_column='answer'),
        'infer_cfg': dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=[
                        dict(role='HUMAN', prompt='{prompt}'),
                        dict(role='BOT', prompt='{answer}\n'),
                    ]
                ),
            ),
            retriever=dict(type=ZeroRetriever),
            # inferencer=dict(type=GenInferencer),
            inferencer=dict(
                type=HRInferencer,
                max_out_len=20000,
                mode='truncate',
                max_iterations=3,
                n_hypotheses=5,
                icl_in_regex = r"Molecular SMILES:\s*([^\n]+)",
                icl_out_regex = r"Molecular formula:\s*([^\n]+)",
                task_name = 'chem_smiles2formula',
                task_description='Given the molecular SMILES, your task is to predict the formula name.',
                generation_kwargs=dict(
                    do_sample=True,  # Enable sampling
                    temperature=1,  # Set temperature for diversity
                ),
            ),
        ),
        'eval_cfg': dict(
            evaluator=dict(type=AccEvaluator),
            pred_postprocessor=dict(type=remove_think_tags),
        ),
        
    }
]


iupac2smiles_hr_datasets = [
    {
        'abbr': 'chem_iupac2smiles',
        'path': 'opencompass/sirbenchv1',
        'type': ChemNamePredictionDataset,
        'icl_num': 5, # number of icl examples
        'num_samples': 30,  # number of test samples
        'task': 'iupac2smiles',
        'reader_cfg': dict(input_columns=['prompt', 'icl_in', 'icl_out'], output_column='answer'),
        'infer_cfg': dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=[
                        dict(role='HUMAN', prompt='{prompt}'),
                        dict(role='BOT', prompt='{answer}\n'),
                    ]
                ),
            ),
            retriever=dict(type=ZeroRetriever),
            # inferencer=dict(type=GenInferencer),
            inferencer=dict(
                type=HRInferencer,
                max_out_len=20000,
                mode='truncate',
                max_iterations=3,
                n_hypotheses=5,
                icl_in_regex = r"Molecular IUPAC name:\s*([^\n]+)",
                icl_out_regex = r"Molecular SMILES:\s*([^\n]+)",
                task_name = 'chem_iupac2smiles',
                task_description='Given the IUPAC name, your task is to predict the molecular SMILES.',
                generation_kwargs=dict(
                    do_sample=True,  # Enable sampling
                    temperature=1,  # Set temperature for diversity
                ),
            ),
        ),
        'eval_cfg': dict(
            evaluator=dict(type=ChemNamePredictionSimlesEvaluator),
            pred_postprocessor=dict(type=remove_think_tags),
        ),
        
    }
]

iupac2formula_hr_datasets = [
    {
        'abbr': 'chem_iupac2formula',
        'path': 'opencompass/sirbenchv1',
        'type': ChemNamePredictionDataset,
        'icl_num': 5, # number of icl examples
        'num_samples': 30,  # number of test samples
        'task': 'iupac2formula',
        'reader_cfg': dict(input_columns=['prompt', 'icl_in', 'icl_out'], output_column='answer'),
        'infer_cfg': dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=[
                        dict(role='HUMAN', prompt='{prompt}'),
                        dict(role='BOT', prompt='{answer}\n'),
                    ]
                ),
            ),
            retriever=dict(type=ZeroRetriever),
            # inferencer=dict(type=GenInferencer),
            inferencer=dict(
                type=HRInferencer,
                max_out_len=20000,
                mode='truncate',
                max_iterations=3,
                n_hypotheses=5,
                icl_in_regex = r"Molecular IUPAC name:\s*([^\n]+)",
                icl_out_regex = r"Molecular formula:\s*([^\n]+)",
                task_name = 'chem_iupac2formula',
                task_description='Given the IUPAC name, your task is to predict the formula name.',
                generation_kwargs=dict(
                    do_sample=True,  # Enable sampling
                    temperature=1,  # Set temperature for diversity
                ),
            ),
        ),
        'eval_cfg': dict(
            evaluator=dict(type=AccEvaluator),
            pred_postprocessor=dict(type=remove_think_tags),
        ),
        
    }
]