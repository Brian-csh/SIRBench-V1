from opencompass.datasets.sirbenchv1.DNATransform import DNATransformDataset
from opencompass.datasets.sirbenchv1.DNATransform import DNATransformEvaluator
from opencompass.datasets.sirbenchv1.utils import extract_content
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.utils.text_postprocessors import think_pred_postprocess
from opencompass.openicl.icl_inferencer import HRInferencer
from opencompass.openicl.icl_inferencer import NEWSCInferencer

td = '''transform a given nucleotide sequence based on a hidden transformation rule.  
 
- Possible transformations include:  
  - **Reversing the sequence**  
  - **Generating the complementary sequence** (note: the nucleotide pairing rule may vary)  
  - **Reverse-complement transformation** (the complement rule follows the same variation as the complementary sequence rule)  
  - **Segmented transformation**: After every **x** bases, the next **n** bases undergo a transformation. **The transformation applied to these bases follows one of the previously observed transformation rules.**  
  - **Fixed-base mutation**: A specific nucleotide **X** is changed into another nucleotide **Y** (both X and Y may vary).  
'''

# CWE Dataset
dna_transform_datasets = [
    {
        'abbr': 'dna_transform_implicit',
        'path': 'opencompass/sirbenchv1',
        'type': DNATransformDataset,
        'icl_num': 5, # number of icl examples
        'num_samples': 100,  # number of test samples
        'seq_len':  300,  # max sequence length
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
            evaluator=dict(type=DNATransformEvaluator),
            pred_postprocessor=dict(type=extract_content),
        ),
    }
]

dna_transform_datasets_advanced = [
    {
        'abbr': 'dna_transform_explicit',
        'path': 'opencompass/sirbenchv1',
        'type': DNATransformDataset,
        'icl_num': 5, # number of icl examples
        'num_samples': 100,  # number of test samples
        'seq_len':  300,  # max sequence length
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
            evaluator=dict(type=DNATransformEvaluator),
            pred_postprocessor=dict(type=extract_content),
        ),
    },
    {
        'abbr': 'dna_transform_sc_whole',
        'path': 'opencompass/sirbenchv1',
        'type': DNATransformDataset,
        'icl_num': 5, # number of icl examples
        'num_samples': 100,  # number of test samples
        'seq_len':  300,  # max sequence length
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
            evaluator=dict(type=DNATransformEvaluator),
            pred_postprocessor=dict(type=extract_content),
        ),
    },
    {
        'abbr': 'dna_transform_hr',
        'path': 'opencompass/sirbenchv1',
        'type': DNATransformDataset,
        'icl_num': 5, # number of icl examples
        'num_samples': 100,  # number of test samples
        'seq_len':  300,  # max sequence length
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
                icl_in_regex = r"\*\*Input:\*\*\n([ACGT]+)",
                icl_out_regex = r"\*\*Output:\*\*\n([ACGT]+)",
                task_description = td,
                task_name = "dna_transform",
                generation_kwargs=dict(
                    do_sample=True,  # Enable sampling
                    temperature=1.0,  # Set temperature for diversity
                ),
            ),
        ),
        'eval_cfg': dict(
            evaluator=dict(type=DNATransformEvaluator),
            pred_postprocessor=dict(type=extract_content),
        ),
    },
]
