"""new SC Generation Inferencer."""

import os
import os.path as osp
import re
from typing import List, Optional, Tuple

import mmengine
import torch
from tqdm import tqdm
from typing import Tuple

from opencompass.models.base import BaseModel
from opencompass.registry import ICL_INFERENCERS
from opencompass.models import OpenAI

from ..icl_prompt_template import PromptTemplate
from ..icl_retriever import BaseRetriever
from ..utils.logging import get_logger
from collections import Counter
from .icl_gen_inferencer import GenInferencer, GenInferencerOutputHandler

logger = get_logger(__name__)


@ICL_INFERENCERS.register_module()
class NEWSCInferencer(GenInferencer):
    """New Self-Consistency Inferencer class to generate hypotheses.

    1. Generates multiple candidate rules
    2. Apply each rule to new example and get outputs
    3. Conduct majority voting on those candidate outputs to get the final output
    """

    def __init__(
            self,
            model: BaseModel,
            max_out_len: int,
            max_seq_len: Optional[int] = None,
            batch_size: Optional[int] = 1,
            sc_size: Optional[int] = 1,
            extract_rule: Optional[str] = 'whole',
            gen_field_replace_token: Optional[str] = '',
            output_json_filepath: Optional[str] = './icl_inference_output',
            output_json_filename: Optional[str] = 'predictions',
            save_every: Optional[int] = 1,
            max_iterations: int = 3,
            n_hypotheses: int = 5,
            # can add other settings here for the inference strategy
            generation_kwargs: dict = {},
            **kwargs) -> None:
        super().__init__(
            model=model,
            max_out_len=max_out_len,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            gen_field_replace_token=gen_field_replace_token,
            output_json_filename=output_json_filename,
            output_json_filepath=output_json_filepath,
            save_every=save_every,
            **kwargs,
        )
        self.max_out_len = max_out_len
        self.max_iterations = max_iterations
        self.sc_size = sc_size
        self.extract_rule = extract_rule
        self.n_hypotheses = n_hypotheses
        self.generation_kwargs = generation_kwargs


        # # Initialize a dedicated model for rule extraction if config is provided
        # self.rule_extractor_model = None
        # if rule_extractor_model_config is not None:
        #     # Create default meta_template if not provided
        #     if 'meta_template' not in rule_extractor_model_config:
        #         rule_extractor_model_config['meta_template'] = dict(round=[
        #             dict(role='HUMAN', api_role='user'),
        #             dict(role='BOT', api_role='assistant', generate=True),
        #         ])

        #     logger.info("Initializing dedicated rule extractor model with config")
        #     self.rule_extractor_model = OpenAI(**rule_extractor_model_config)

    def generate_rules(self, x: str, flag: str) -> List[str]:
        """Generate multiple rules for a given input.
        Args:
            x (str): The prompt text.
        Returns:
            List[str]: List of generated rules.
        """
        base_prompt_separate = """Below is a full prompt about the reasoning task, which includes the in-context examples that you should learn from.

Your task is:
1. Read the full prompt to understand the task and identify the example input-output pairs.
2. For each example pair, generate a separate rule that explains how the input is transformed into the output.
3. Return your rules in the following format (each rule on its own line):


Rule 1: ...
Rule 2: ...
Rule 3: ...
...

Full prompt:
{full_prompt}
"""
        base_prompt_whole = """Below is a full prompt about the reasoning task, which includes the in-context examples that you should learn from.

Your task is:
1. Read the full prompt to understand the task and identify the example input-output pairs.
2. Analyze these example pairs and generate a series of rules that explains how each input is transformed to its corresponding output.
3. Return your rules in the following format (each rule on its own line):


Rule 1: ...
Rule 2: ...
Rule 3: ...
...

Full prompt:
{full_prompt}
"""

        rules = []
        if flag == 'whole':
            logger.info("----use whole_based prompt!-----")
            base_prompt = base_prompt_whole
        else:
            logger.info("----use separatebased prompt!-----")
            base_prompt = base_prompt_separate

        prompts = [base_prompt.format(full_prompt=x)]
        result = self.model.generate_from_template(
            prompts,
            max_out_len=self.max_out_len,
            **self.generation_kwargs)[0]

        # 匹配形如 "Rule 1: xxx" 的每一行
        pattern = r"Rule \d+:\s*(.+)"
        rules = re.findall(pattern, result)
        return rules

    def apply_rules_to_new_example(self, rules: List[str], x: str) -> str:
        base_prompt = """Below is a full prompt about the reasoning task, which includes the question that you should give the corresponding answer.

Your task is:
1. Read the full prompt to understand the task and identify the specific input question to answer.
2. Based on your understanding of the given rules, generate the corresponding output for the question.


Rules:
{rules}

Full prompt:
{full_prompt}

Answer:"""

        prompts = [base_prompt.format(rules=rules, full_prompt=x)]
        result = self.model.generate_from_template(
            prompts,
            max_out_len=self.max_out_len,
            **self.generation_kwargs)[0]

        output = self.extract_final_result(result)

        return output

    def usc_voting(self, full_prompt: str, responses: list[str]) -> str:
        usc_prompt = f"""Below is a full prompt about a reasoning task, which includes the question that you should answer.

    Your task is:
    1. Read the full prompt to understand the task and identify the specific input question.
    2. Evaluate the following responses to that question.
    3. Select the most consistent response based on majority consensus among the responses.

    Responses:
    {responses}

    Full prompt:
    {full_prompt}

    Return the most consistent response in the following format:
    the most consistent response: {{full response content}}"""

        model_to_use = self.model
        feedback = model_to_use.generate_from_template(
            [usc_prompt],
            max_out_len=self.max_out_len)[0]

        match = re.search(r"the most consistent response:\s*(.+)", feedback, re.DOTALL)
        if match:
            content = match.group(1).strip()
            logger.info(f"------the most consistent response:{content}-------")
            return content
        else:
            return f"[Warning] Could not parse USC result.\nLLM output: {feedback}"

    def new_usc_voting(self, full_prompt: str, responses: list[str]) -> str:
        usc_prompt = f"""You are given a reasoning task prompt and multiple candidate responses to the question in that prompt.

Your task is:
1. Read the full prompt carefully to understand the question being asked.
2. Examine all the candidate responses and determine whether any of them form a majority consensus.
   - A majority exists if **any single response appears more than any other** (either verbatim or semantically equivalent).
   - In case of a tie (e.g., all responses differ or two responses appear with equal frequency), consider that no majority exists.
3. If a majority exists, return that response as the final answer.
4. If no majority exists, then select the **most reasonable and task-appropriate** response based on the prompt.

Candidate responses:
{responses}

Full prompt:
{full_prompt}

Return your final answer using **exactly** the following format:

majority_found: [yes or no]  
selected_response: {{full response content}}

Example:
majority_found: yes  
selected_response: This is the most common (or semantically equivalent) response and correctly answers the question.
"""


        model_to_use = self.model
        feedback = model_to_use.generate_from_template(
            [usc_prompt],
            max_out_len=self.max_out_len)[0]
        
        majority_match = re.search(r'majority_found:\s*(yes|no)', feedback, re.IGNORECASE)

        # 匹配 selected_response 后面的内容（可能是多行）
        selected_match = re.search(
            r'selected_response:\s*(.*)',
            feedback,
            re.IGNORECASE | re.DOTALL
        )

        if selected_match:
            # 去除前后空格
            if majority_match:
                logger.info("Find the most consistent one!!")
                selected_response = selected_match.group(1).strip()
                logger.info(f"------the most consistent response:{selected_response}-------")
            else:
                logger.info("NO consistent! ")
                selected_response = selected_match.group(1).strip()
                logger.info(f"------the most reasonable response:{selected_response}-------")
        
            return selected_response
        else:
            return f"[Warning] Could not parse USC result.\nLLM output: {feedback}"  # 如果没有匹配成功

    def sc_solve(self, x: str) -> Tuple[List[str], str]:
        """Solve a problem using sc.
        Args:
            x (str): The initial prompt text
        Returns:
            list[str]: Final hypothesis
            str: output
        """
        logger.info("Starting self-consistency inference...")
        # Generate candidate rules
        rules = self.generate_rules(x,
                                    self.extract_rule)  # whole/separate用于控制model是从所有example pairs学习整体的系列规则；还是根据每个pair提取规则
        logger.info(f"Generated {len(rules)} rules.")
        i = 0
        for rule in rules:
            i += 1
            logger.info(f"Rule {i}: {rule}")

        # apply rules to test example
        output = self.apply_rules_to_new_example(rules, x)

        for i, rule in enumerate(rules):
            logger.info(f"Rule {i + 1}: {rule}")
        logger.info(f"Prediction on new example: {output}")

        return rules, output

    def extract_final_result(self, prompt_text: str) -> str:
        prompt = f"""You will see an output from a large language model. Please extract the final answer from the output without any additional information like 'result','answer'...  

Output from the model:  
{prompt_text}
"""
        model_to_use = self.model

        # Use specific generation parameters for rule extraction
        feedback = model_to_use.generate_from_template(
            [prompt],
            max_out_len=self.max_out_len)[0]
        return feedback

    def inference(self,
                  retriever: BaseRetriever,
                  ice_template: Optional[PromptTemplate] = None,
                  prompt_template: Optional[PromptTemplate] = None,
                  output_json_filepath: Optional[str] = None,
                  output_json_filename: Optional[str] = None) -> List:

        # 1. Preparation for output logs
        output_handler = GenInferencerOutputHandler()

        if output_json_filepath is None:
            output_json_filepath = self.output_json_filepath
        if output_json_filename is None:
            output_json_filename = self.output_json_filename

        # 2. Get results of retrieval process
        ice_idx_list = retriever.retrieve()

        # 3. Generate prompts for testing input
        prompt_list = self.get_generation_prompt_list_from_retriever_indices(
            ice_idx_list,
            retriever,
            self.gen_field_replace_token,
            max_seq_len=self.max_seq_len,
            ice_template=ice_template,
            prompt_template=prompt_template)

        # 3.1 Fetch and zip prompt & gold answer if output column exists
        ds_reader = retriever.dataset_reader
        if ds_reader.output_column:
            gold_ans = ds_reader.dataset['test'][ds_reader.output_column]
            prompt_list = list(zip(prompt_list, gold_ans))

        # Create tmp json file for saving intermediate results and future
        # resuming
        index = 0
        tmp_json_filepath = os.path.join(output_json_filepath,
                                         'tmp_' + output_json_filename)
        if osp.exists(tmp_json_filepath):
            tmp_result_dict = mmengine.load(tmp_json_filepath)
            output_handler.results_dict = tmp_result_dict
            index = len(tmp_result_dict)

        # 4. Wrap prompts with Dataloader
        dataloader = self.get_dataloader(prompt_list[index:], self.batch_size)

        # 5. Inference for prompts in each batch
        logger.info('Starting Self-consistency inference process...')
        for datum in tqdm(dataloader, disable=not self.is_main_process):
            if not datum:
                logger.warning("Empty datum encountered. Skipping...")
                continue
            if ds_reader.output_column:
                entries, golds = list(zip(*datum))
            else:
                entries = datum
                golds = [None for _ in range(len(entries))]

            with torch.no_grad():
                parsed_entries = self.model.parse_template(entries, mode='gen')
                all_sc_results = []
                i = 0
                for entry in entries:
                    i += 1
                    logger.info(f"entry {i}/{len(entries)}")
                    sc_results = []
                    record = {}
                    k = 0
                    for _ in range(self.sc_size):
                        k += 1
                        logger.info(f"SC Sampling {k}/{self.sc_size}")
                        # 返回两个参数，一个是对应总结的一套规则rule_set，一个是得到的一个output
                        rule_set, result = self.sc_solve(entry)
                        logger.info(f"rule_set:{rule_set},output:{result}")
                        sc_results.append(result)
                        record[
                            str(rule_set)] = result  # 所有返回的{rule_set1:output1, rule_set2:output2, rule_set3:output3....}

                    # 原始voting
                    # counter = Counter(sc_results)
                    # if counter:
                    #     most_common = counter.most_common(1)[0][0]
                    # else:
                    #     most_common = ""

                    # final_output = most_common
                    # all_sc_results.append(final_output)
                    # logger.info(f"Final output: {final_output} for entry: {entry}")

                    # usc_voting
                    if self.sc_size == 1:
                        all_sc_results.append(sc_results[0])
                    else:
                        #most_consistent_answer = self.usc_voting(entry, sc_results)
                        most_consistent_answer = self.new_usc_voting(entry, sc_results)
                        logger.info(f"the most consistent answer is -> {most_consistent_answer}")
                        final_output = most_consistent_answer
                        all_sc_results.append(final_output)
                        logger.info(f"Final output: {final_output} for entry: {entry}")

                # sc_prediction = list(map(list, zip(*sc_results)))
                generated = all_sc_results

            # 5-2. Save current output
            # parsed_entries = self.model.parse_template(entries, mode='gen')
            for prompt, prediction, gold in zip(parsed_entries, generated,
                                                golds):
                output_handler.save_results(prompt,
                                            prediction,
                                            index,
                                            gold=gold)
                index = index + 1

            # 5-3. Save intermediate results
            if (self.save_every is not None and index % self.save_every == 0
                    and self.is_main_process):
                output_handler.write_to_json(output_json_filepath,
                                             'tmp_' + output_json_filename)

        # 6. Output
        if self.is_main_process:
            os.makedirs(output_json_filepath, exist_ok=True)
            output_handler.write_to_json(output_json_filepath,
                                         output_json_filename)
            if osp.exists(tmp_json_filepath):
                os.remove(tmp_json_filepath)

        return [
            sample['prediction']
            for sample in output_handler.results_dict.values()
        ]