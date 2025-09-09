from opencompass.datasets.sirbenchv1.DNATranslator import DNATranslatorDataset
from opencompass.datasets.sirbenchv1.DNATranslator import DNATranslatorEvaluator
from opencompass.datasets.sirbenchv1.utils import extract_content
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_inferencer import HRInferencer
from opencompass.openicl.icl_inferencer import NEWSCInferencer



# CWE Dataset
dna_translator_datasets = [
    {
        'abbr': 'dna_traslator_implicit',
        'path': 'opencompass/sirbenchv1',
        'type': DNATranslatorDataset,
        'icl_num': 5, # number of icl examples
        'num_samples': 100,  # number of test samples
        'max_len':  450,  # max sequence length
        'min_len':  200,  # min sequence length
        'synthesize_table': False,  # is the mapping synthetic
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
        ),
        'eval_cfg': dict(
            evaluator=dict(type=DNATranslatorEvaluator),
            pred_postprocessor=dict(type=extract_content),
        ), 
    }
]

dna_translator_datasets_advanced = [
    {
        'abbr': 'dna_translator_explicit',
        'path': 'opencompass/sirbenchv1',
        'type': DNATranslatorDataset,
        'icl_num': 5, # number of icl examples
        'num_samples': 100,  # number of test samples
        'max_len':  450,  # max sequence length
        'min_len':  200,  # min sequence length
        'synthesize_table': False,  # is the mapping synthetic
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
                type=NEWSCInferencer,
                sc_size=1,
            ),
        ),
        'eval_cfg': dict(
            evaluator=dict(type=DNATranslatorEvaluator),
            pred_postprocessor=dict(type=extract_content),
        ),
    },
    {
        'abbr': 'dna_translator_sc_whole',
        'path': 'opencompass/sirbenchv1',
        'type': DNATranslatorDataset,
        'icl_num': 5, # number of icl examples
        'num_samples': 100,  # number of test samples
        'max_len':  450,  # max sequence length
        'min_len':  200,  # min sequence length
        'synthesize_table': False,  # is the mapping synthetic
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
                type=NEWSCInferencer,
                sc_size=5,
                generation_kwargs=dict(
                    do_sample=True,  # Enable sampling
                    temperature=1.0,  # Set temperature for diversity
                ),
            ),
        ),
        'eval_cfg': dict(
            evaluator=dict(type=DNATranslatorEvaluator),
            pred_postprocessor=dict(type=extract_content),
        ),
    },
    {
        'abbr': 'dna_translator_hr',
        'path': 'opencompass/sirbenchv1',
        'type': DNATranslatorDataset,
        'icl_num': 5, # number of icl examples
        'num_samples': 100,  # number of test samples
        'max_len':  450,  # max sequence length
        'min_len':  200,  # min sequence length
        'synthesize_table': False,  # is the mapping synthetic
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
                icl_in_regex = r"DNA:\n([^\n]*)",
                icl_out_regex = r"AA:\n([^\n]*)",
                task_description = "transcribe a given DNA sequence into an amino acid sequence",
                task_name = "dna_translate",
                generation_kwargs=dict(
                    do_sample=True,  # Enable sampling
                    temperature=1.0,  # Set temperature for diversity
                ),

            ),
        ),
        'eval_cfg': dict(
            evaluator=dict(type=DNATranslatorEvaluator),
            pred_postprocessor=dict(type=extract_content),
        ),
    },
]

dna_translator_synthetic_datasets = [
    {
        'abbr': 'dna_translator_synthetic_implicit',
        'path': 'opencompass/sirbenchv1',
        'type': DNATranslatorDataset,
        'icl_num': 5, # number of icl examples
        'num_samples': 100,  # number of test samples
        'max_len':  450,  # max sequence length
        'min_len':  200,  # min sequence length
        'synthesize_table': True,  # is the mapping synthetic
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
        ),
        'eval_cfg': dict(
            evaluator=dict(type=DNATranslatorEvaluator),
            pred_postprocessor=dict(type=extract_content),
        ),
    }
]

dna_translator_synthetic_datasets_advanced = [
    {
        'abbr': 'dna_translator_synthetic_explicit',
        'path': 'opencompass/sirbenchv1',
        'type': DNATranslatorDataset,
        'icl_num': 5, # number of icl examples
        'num_samples': 100,  # number of test samples
        'max_len':  450,  # max sequence length
        'min_len':  200,  # min sequence length
        'synthesize_table': True,  # is the mapping synthetic
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
                type=NEWSCInferencer,
                sc_size=1,
            ),
        ),
        'eval_cfg': dict(
            evaluator=dict(type=DNATranslatorEvaluator),
            pred_postprocessor=dict(type=extract_content),
        ),
    },
    {
        'abbr': 'dna_translator_synthetic_sc_whole',
        'path': 'opencompass/sirbenchv1',
        'type': DNATranslatorDataset,
        'icl_num': 5, # number of icl examples
        'num_samples': 100,  # number of test samples
        'max_len':  450,  # max sequence length
        'min_len':  200,  # min sequence length
        'synthesize_table': True,  # is the mapping synthetic
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
                type=NEWSCInferencer,
                sc_size=5,
                generation_kwargs=dict(
                    do_sample=True,  # Enable sampling
                    temperature=1.0,  # Set temperature for diversity
                ),
            ),
        ),
        'eval_cfg': dict(
            evaluator=dict(type=DNATranslatorEvaluator),
            pred_postprocessor=dict(type=extract_content),
        ),
    },
    {
        'abbr': 'dna_translator_synthetic_hr',
        'path': 'opencompass/sirbenchv1',
        'type': DNATranslatorDataset,
        'icl_num': 5, # number of icl examples
        'num_samples': 100,  # number of test samples
        'max_len':  450,  # max sequence length
        'min_len':  200,  # min sequence length
        'synthesize_table': True,  # is the mapping synthetic
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
                icl_in_regex = r"DNA:\n([^\n]*)",
                icl_out_regex = r"AA:\n([^\n]*)",
                task_description = "Transcribe a given DNA sequence into an amino acid sequence. The codon table may not be standard.",
                task_name = "dna_translate",
                generation_kwargs=dict(
                    do_sample=True,  # Enable sampling
                    temperature=1.0,  # Set temperature for diversity
                ),
            ),
        ),
        'eval_cfg': dict(
            evaluator=dict(type=DNATranslatorEvaluator),
            pred_postprocessor=dict(type=extract_content),
        ),
    }
]
