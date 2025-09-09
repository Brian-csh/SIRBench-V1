from mmengine.config import read_base
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.summarizers.default import DefaultSummarizer
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask
with read_base():
    from opencompass.configs.datasets.sirbenchv1.chem_molecule_design_gen import molecule_design_datasets, molecule_caption_datasets, molecule_design_hr_datasets, molecule_caption_hr_datasets

from opencompass.models import HuggingFacewithChatTemplate, OpenAI
from opencompass.partitioners import NaivePartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.summarizers import DefaultSubjectiveSummarizer
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask


models = [
    dict(
        abbr='gpt-4.1',  # model name to be shown in the leaderboard
        type=OpenAI,
        path='gpt-4.1',   # model name in the api
        key='',
        openai_api_base='',
        # meta_template=api_meta_template,
        max_out_len=16384*2,
        max_seq_len=16384*2,
        batch_size=8,
        temperature=1.0,
    )
]

datasets = molecule_design_datasets + molecule_caption_datasets

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner),
    runner=dict(type=LocalRunner,
                max_num_workers=16,
                task=dict(type=OpenICLInferTask),
                retry=5),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner,
                max_num_workers=32,
                task=dict(type=OpenICLEvalTask)),
)
summarizer = dict(type=DefaultSummarizer)
work_dir = 'outputs/sirbenchv1/chem_md'

