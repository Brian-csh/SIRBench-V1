from opencompass.datasets.sirbenchv1.ChemMoleculeDesign import ChemMoleculeCaptionEvaluator, ChemMoleculeDesignDataset
from opencompass.datasets.sirbenchv1.ChemMoleculeDesign import ChemMoleculeDesignEvaluator
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_inferencer import HRInferencer
from opencompass.openicl.icl_inferencer import NEWSCInferencer
from opencompass.datasets.sirbenchv1.utils import remove_think_tags



molecule_design_datasets = [
    {
        'abbr': 'chem_molecule_design',
        'path': 'opencompass/sirbenchv1',
        'type': ChemMoleculeDesignDataset,
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
            evaluator=dict(type=ChemMoleculeDesignEvaluator),
            pred_postprocessor=dict(type=remove_think_tags),
        ),
        
    }
]

molecule_design_hr_datasets = [
    {
        'abbr': 'chem_molecule_design',
        'path': 'opencompass/sirbenchv1',
        'type': ChemMoleculeDesignDataset,
        'icl_num': 5, # number of icl examples
        'num_samples': 2,  # number of test samples
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
                max_iterations=2,
                n_hypotheses=2,
                #icl_in_regex = r"Molecular requirements description:\s*(.+)",
                icl_in_regex = r"Molecular requirements description:\s*([^\n]+?)\s*Molecular SMILES:",
                icl_out_regex = r"Molecular SMILES:\s*([A-Za-z0-9@+\-\[\]\(\)=#$%/\\]+)",
                task_name = 'chem_molecule_design',
                task_description='Given the molecular requirements description, your task is to design a new molecule using your experienced chemical Molecular Design knowledge',
                generation_kwargs=dict(
                    do_sample=True,  # Enable sampling
                    temperature=1,  # Set temperature for diversity
                ),
            ),
        ),
        'eval_cfg': dict(
            evaluator=dict(type=ChemMoleculeDesignEvaluator),
            pred_postprocessor=dict(type=remove_think_tags),
        ),
        
    }
]

molecule_caption_datasets = [
    {
        'abbr': 'chem_molecule_caption',
        'path': 'opencompass/sirbenchv1',
        'type': ChemMoleculeDesignDataset,
        'icl_num': 5, # number of icl examples
        'num_samples': 30,  # number of test samples
        'task': 'molecule_captioning',
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
            evaluator=dict(type=ChemMoleculeCaptionEvaluator),
            pred_postprocessor=dict(type=remove_think_tags),
        ),
        
    }
]

molecule_caption_hr_datasets = [
    {
        'abbr': 'chem_molecule_caption',
        'path': 'opencompass/sirbenchv1',
        'type': ChemMoleculeDesignDataset,
        'icl_num': 5, # number of icl examples
        'num_samples': 2,  # number of test samples
        'task': 'molecule_captioning',
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
                max_iterations=2,
                n_hypotheses=5,
                #icl_in_regex = r"Molecular SMILES:\s*([A-Za-z0-9@+\-\[\]\(\)=#$%/\\]+)",
                icl_in_regex = r"Molecule SMILES:\s*([^\n]+?)\s*Molecular Description:",
                icl_out_regex = r"Molecular Description:\s*(.+)",
                task_name = 'chem_molecule_caption',
                task_description = 'Given the molecular SMILES, your task is to provide the detailed description of the molecule using your experienced chemical Molecular knowledge.',
                generation_kwargs=dict(
                    do_sample=True,  # Enable sampling
                    temperature=1,  # Set temperature for diversity
                ),
            ),
        ),
        'eval_cfg': dict(
            evaluator=dict(type=ChemMoleculeCaptionEvaluator),
            pred_postprocessor=dict(type=remove_think_tags),
        ),
        
    }
]