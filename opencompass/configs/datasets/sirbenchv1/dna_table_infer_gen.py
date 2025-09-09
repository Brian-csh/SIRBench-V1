from opencompass.datasets.sirbenchv1.DNATableInfer import DNATableInferDataset, DNATableInferEvaluator
from opencompass.datasets.sirbenchv1.utils import extract_content
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.utils.text_postprocessors import think_pred_postprocess
from opencompass.openicl.icl_inferencer import HRInferencer
from opencompass.openicl.icl_inferencer import NEWSCInferencer


# CWE Dataset
dna_table_infer_datasets = [
    {
        'abbr': 'dna_table_inference_implicit',
        'path': 'opencompass/sirbenchv1',
        'type': DNATableInferDataset,
        'icl_num': 5, # number of icl examples
        'num_samples': 100,  # number of test samples
        'min_len':  200,  # min sequence length
        'max_len':  450,  # max sequence length
        'synthesize_table': False,  # Whether to synthesize codon table, False uses the predefined one in the bio library
        'synthesize_example':  False,  # Whether to enable synthesis example, if synthesize_table==False and synthesize_example==False, then the codon table can only use the generated codon table in the standard codon table

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
            evaluator=dict(type=DNATableInferEvaluator),
            pred_postprocessor=dict(type=extract_content),
        ), 
    }
]

dna_table_infer_datasets_advanced = [
    {
        'abbr': 'dna_table_inference_explicit',
        'path': 'opencompass/sirbenchv1',
        'type': DNATableInferDataset,
        'icl_num': 5, # number of icl examples
        'num_samples': 100,  # number of test samples
        'min_len':  200,  # min sequence length
        'max_len':  450,  # min sequence length
        'synthesize_table': False,  # Whether to synthesize codon table, False uses the predefined one in the bio library
        'synthesize_example':  False,  # Whether to enable synthesis example, if synthesize_table==False and synthesize_example==False, then the codon table can only use the generated codon table in the standard codon table

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
            evaluator=dict(type=DNATableInferEvaluator),
            pred_postprocessor=dict(type=extract_content),
        ),
    },
    {
        'abbr': 'dna_table_inference_sc_whole',
        'path': 'opencompass/sirbenchv1',
        'type': DNATableInferDataset,
        'icl_num': 5, # number of icl examples
        'num_samples': 100,  # number of test samples
        'min_len':  200,  # min sequence length
        'max_len':  450,  # max sequence length
        'synthesize_table': False,  # Whether to synthesize codon table, False uses the predefined one in the bio library
        'synthesize_example':  False,  # Whether to enable synthesis example, if synthesize_table==False and synthesize_example==False, then the codon table can only use the generated codon table in the standard codon table

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
            evaluator=dict(type=DNATableInferEvaluator),
            pred_postprocessor=dict(type=extract_content),
        ),
    },
    {
        'abbr': 'dna_table_inference_hr',
        'path': 'opencompass/sirbenchv1',
        'type': DNATableInferDataset,
        'icl_num': 5, # number of icl examples
        'num_samples': 100,  # number of test samples
        'min_len':  200,  # min sequence length
        'max_len':  450,  # max sequence length
        'synthesize_table': False,  # Whether to synthesize codon table, False uses the predefined one in the bio library
        'synthesize_example':  False,  # Whether to enable synthesis example, if synthesize_table==False and synthesize_example==False, then the codon table can only use the generated codon table in the standard codon table
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
                icl_in_regex = r"example DNA \d+:\s*\n([ATGC]+)",
                icl_out_regex = r"example Protein \d+:\s*\n([A-Z\*]+)",
                task_description = "Analyze the provided DNA → amino acid mapping examples and **infer the codon table**. The codon table may not be standard.",
                task_name = 'dna_table_infer',
                generation_kwargs=dict(
                    do_sample=True,  # Enable sampling
                    temperature=1.0,  # Set temperature for diversity
                ),
            ),
        ),
        'eval_cfg': dict(
            evaluator=dict(type=DNATableInferEvaluator),
            pred_postprocessor=dict(type=extract_content),
        ),   
    }
]

dna_table_infer_synthetic_datasets = [
    {
        'abbr': 'dna_table_inference_synthetic_implicit',
        'path': 'opencompass/sirbenchv1',
        'type': DNATableInferDataset,
        'icl_num': 5, # number of icl examples
        'num_samples': 100,  # number of test samples
        'min_len':  200,  # min sequence length
        'max_len':  450,  # max sequence length
        'synthesize_table': True,  # Whether to synthesize codon table, False uses the predefined one in the bio library
        'synthesize_example':  False,  # Whether to enable synthesis example, if synthesize_table==False and synthesize_example==False, then the codon table can only use the generated codon table in the standard codon table

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
            evaluator=dict(type=DNATableInferEvaluator),
            pred_postprocessor=dict(type=extract_content),
        ),
    }
]

dna_table_infer_synthetic_datasets_advanced = [
    {
        'abbr': 'dna_table_inference_synthetic_explicit',
        'path': 'opencompass/sirbenchv1',
        'type': DNATableInferDataset,
        'icl_num': 5, # number of icl examples
        'num_samples': 100,  # number of test samples
        'min_len':  200,  # min sequence length
        'max_len':  450,  # max sequence length
        'synthesize_table': True,  # Whether to synthesize codon table, False uses the predefined one in the bio library
        'synthesize_example':  False,  # Whether to enable synthesis example, if synthesize_table==False and synthesize_example==False, then the codon table can only use the generated codon table in the standard codon table

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
            evaluator=dict(type=DNATableInferEvaluator),
            pred_postprocessor=dict(type=extract_content),
        ),
    },
    {
        'abbr': 'dna_table_inference_synthetic_sc_whole',
        'path': 'opencompass/sirbenchv1',
        'type': DNATableInferDataset,
        'icl_num': 5, # number of icl examples
        'num_samples': 100,  # number of test samples
        'min_len':  200,  # min sequence length
        'max_len':  450,  # max sequence length
        'synthesize_table': True,  # Whether to synthesize codon table, False uses the predefined one in the bio library
        'synthesize_example':  False,  # Whether to enable synthesis example, if synthesize_table==False and synthesize_example==False, then the codon table can only use the generated codon table in the standard codon table

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
            evaluator=dict(type=DNATableInferEvaluator),
            pred_postprocessor=dict(type=extract_content),
        ),
    },
    {
        'abbr': 'dna_table_inference_synthetic_hr',
        'path': 'opencompass/sirbenchv1',
        'type': DNATableInferDataset,
        'icl_num': 5, # number of icl examples
        'num_samples': 100,  # number of test samples
        'min_len':  200,  # min sequence length
        'max_len':  450,  # max sequence length
        'synthesize_table': True,  # Whether to synthesize codon table, False uses the predefined one in the bio library
        'synthesize_example':  False,  # Whether to enable synthesis example, if synthesize_table==False and synthesize_example==False, then the codon table can only use the generated codon table in the standard codon table
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
                icl_in_regex = r"example DNA \d+:\s*\n([ATGC]+)",
                icl_out_regex = r"example Protein \d+:\s*\n([A-Z\*]+)",
                task_description = "analyze the provided DNA → amino acid mapping examples and **infer the codon table**",
                task_name = 'dna_table_infer',
                generation_kwargs=dict(
                    do_sample=True,  # Enable sampling
                    temperature=1.0,  # Set temperature for diversity
                ),
            ),
        ),
        'eval_cfg': dict(
            evaluator=dict(type=DNATableInferEvaluator),
            pred_postprocessor=dict(type=extract_content),
        ),   
    }
]
