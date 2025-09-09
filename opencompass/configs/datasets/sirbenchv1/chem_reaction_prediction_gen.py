from opencompass.datasets.sirbenchv1.ChemReactionPrediction import ChemReactionPredictionDataset, ChemReactionPredictionEvaluator
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.sirbenchv1.utils import remove_think_tags
from opencompass.openicl.icl_inferencer import HRInferencer
from opencompass.openicl.icl_inferencer import NEWSCInferencer

chem_reaction_prediction_datasets = [
    {
        'abbr': 'chem_reaction_prediction',
        'path': 'opencompass/sirbenchv1',
        'type': ChemReactionPredictionDataset,
        'icl_num': 5, # number of icl examples
        'num_samples': 30,  # number of test samples
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
            # inferencer=dict(type=NEWSCInferencer, max_out_len=16384, infer_type='sc', sc_size = 5,
            #                 generation_kwargs=dict(
            #                     do_sample=True,  # Enable sampling
            #                     temperature=1,  ),),
        ),
        'eval_cfg': dict(
            evaluator=dict(type=ChemReactionPredictionEvaluator),
            pred_postprocessor=dict(type=remove_think_tags),
        ),
        
    }
]

chem_reaction_prediction_hr_datasets = [
    {
        'abbr': 'chem_reaction_prediction_hr',
        'path': 'opencompass/sirbenchv1',
        'type': ChemReactionPredictionDataset,
        'icl_num': 5, # number of icl examples
        'num_samples': 30,  # number of test samples
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
            inferencer=dict(
                type=HRInferencer,
                max_out_len=16384,
                max_iterations=3,
                n_hypotheses=5,
                icl_in_regex = r"Reactants\+Reagents: (.*?)\n",
                icl_out_regex = r"Products: (.*?)\n",
                task_description = "Given the reactants SMILES, predict the main product SMILES.",
                task_name = "chem_reaction_prediction",
                generation_kwargs=dict(
                    do_sample=True,  # Enable sampling
                    temperature=1,  # Set temperature for diversity
                ),

            ),
        ),
        'eval_cfg': dict(
            evaluator=dict(type=ChemReactionPredictionEvaluator),
            pred_postprocessor=dict(type=remove_think_tags),
        ),
        
    }
]
