"""Hypothesis Refinement Generation Inferencer."""

import os
import os.path as osp
import re
from typing import List, Optional, Tuple

import mmengine
import torch
from tqdm import tqdm

from opencompass.models.base import BaseModel
from opencompass.registry import ICL_INFERENCERS

from ..icl_prompt_template import PromptTemplate
from ..icl_retriever import BaseRetriever
from ..utils.logging import get_logger
from .icl_gen_inferencer import GenInferencer, GenInferencerOutputHandler

import datetime
from opencompass.datasets.sirbenchv1.DNATransform import DNATransformEvaluator
from opencompass.datasets.sirbenchv1.DNATranslator import DNATranslatorEvaluator
from opencompass.datasets.sirbenchv1.DNATableInfer import DNATableInferHREvaluator
from opencompass.datasets.sirbenchv1.ChemReactionPrediction import ChemReactionPredictionEvaluator
from opencompass.datasets.sirbenchv1.ChemMoleculeDesign import ChemMoleculeDesignEvaluator, ChemMoleculeCaptionEvaluator
from opencompass.datasets.sirbenchv1.ChemNamePrediction import ChemNamePredictionSimlesEvaluator
from opencompass.openicl.icl_evaluator import AccEvaluator

logger = get_logger(__name__)


def get_evaluator(task_name: str):
    if task_name == 'dna_transform':
        return DNATransformEvaluator()
    elif task_name == 'dna_translate':
        return DNATranslatorEvaluator()
    elif task_name == 'dna_table_infer':
        return DNATableInferHREvaluator()
    elif task_name == 'chem_reaction_prediction':
        return ChemReactionPredictionEvaluator()
    elif task_name == 'chem_molecule_design':
        return ChemMoleculeDesignEvaluator()
    elif task_name == 'chem_molecule_caption':
        return ChemMoleculeCaptionEvaluator()
    elif task_name == 'chem_smiles2iupac' or task_name == 'chem_smiles2formula' or task_name == 'chem_iupac2formula': 
        return AccEvaluator()
    elif task_name == 'chem_iupac2smiles':
        return ChemNamePredictionSimlesEvaluator()
    else:
        raise ValueError(f"Unknown task name: {task_name}")


@ICL_INFERENCERS.register_module()
class HRInferencer(GenInferencer):
    """Hypothesis Refinement Inferencer class to refine hypotheses iteratively and apply the hypothesis to the problem.
    
    This inferencer implements the Hypothesis Refinement approach, which:
    1. Generates multiple initial hypotheses
    2. Evaluates each hypothesis and selects the best one
    3. Gets feedback on the hypothesis
    4. Refines the hypothesis based on feedback
    5. Repeats the process for a specified number of iterations
    6. Apply the hypothesis to obtain final answer

    Attributes:
        model (:obj:`BaseModelWrapper`, optional): The module to inference.
        max_out_len (:obj:`int`, optional): Maximum output length.
        max_seq_len (:obj:`int`, optional): Maximum sequence length allowed
            by the LM.
        batch_size (:obj:`int`, optional): Batch size for the
            :obj:`DataLoader`.
        output_json_filepath (:obj:`str`, optional): File path for output
            `JSON` file.
        output_json_filename (:obj:`str`, optional): File name for output
            `JSON` file.
        gen_field_replace_token (:obj:`str`, optional): Used to replace the
            generation field token when generating prompts.
        save_every (:obj:`int`, optional): Save intermediate results every
            `save_every` iters.
        generation_kwargs (:obj:`Dict`, optional): Parameters for the
            :obj:`model.generate()` method.
        max_iterations (:obj:`int`): Maximum number of refinement iterations.
        n_hypotheses (:obj:`int`): Number of hypotheses to generate.
        task_description (:obj:`str`, optional): Description of the task.
        icl_in_regex (:obj:`str`, optional): Regex pattern to extract
            in-context learning input from the prompt.
        icl_out_regex (:obj:`str`, optional): Regex pattern to extract
            in-context learning output from the prompt.
    """

    def __init__(
            self,
            model: BaseModel,
            max_out_len: int,
            max_seq_len: Optional[int] = None,
            batch_size: Optional[int] = 1,
            gen_field_replace_token: Optional[str] = '',
            output_json_filepath: Optional[str] = './icl_inference_output',
            output_json_filename: Optional[str] = 'predictions',
            save_every: Optional[int] = 1,
            max_iterations: int = 3,
            n_hypotheses: int = 5,
            task_description: Optional[str] = None,
            icl_in_regex: Optional[str] = None,
            icl_out_regex: Optional[str] = None,
            task_name: Optional[str] = None,
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
        self.n_hypotheses = n_hypotheses
        self.generation_kwargs = generation_kwargs
        self.task_description = task_description
        self.icl_in_regex = icl_in_regex
        self.icl_out_regex = icl_out_regex

        self.task_name = task_name
        self.evaluator = get_evaluator(task_name)
    
    def generate_hypotheses(self, x: str) -> List[str]:

        """Generate multiple hypothesis for a given input.

        Args:
            x (str): The prompt text.

        Returns:
            List[str]: List of generated rules.
        """

        if self.task_name == 'dna_translate':
            note = " Please provide all the codon to amino acid mappings."
        else:
            note = ""

        base_prompt = """Below is a full prompt about the reasoning task, which includes the in-context examples that you should learn from.

Your task is:
1. Read the full prompt to understand the task and identify the example input-output pairs.
2. Analyze these example pairs and generate a series of rules that explains how each input is transformed to its corresponding output.
3. Provide as much detail as possible in the rules, such as elaborating on the specific mapping.{note}
4. Return your rules in the following format (each rule on its own line):


<hypothesis>
Rule 1: ...
Rule 2: ...
Rule 3: ...
...
</hypothesis>

Full prompt:
{full_prompt}
"""

        hypotheses = []
        prompts = [base_prompt.format(full_prompt=x, note=note)]
        logger.info(f"Generating hypotheses for input: {prompts[0]}...")

        for _ in range(self.n_hypotheses):
            result = self.model.generate_from_template(
                    prompts,
                    max_out_len=self.max_out_len,
                    **self.generation_kwargs)[0]
        
            result = self.remove_think_tags(result)
            # Extract content between <hypothesis> tags
            pattern = r"<hypothesis>(.*?)</hypothesis>"
            matches = re.search(pattern, result, re.DOTALL)
            
            if matches:
                # Get the content between the hypothesis tags
                hypothesis_content = matches.group(1).strip()
                hypotheses.append(hypothesis_content)
            else:
                # Try to extract just the rules if no hypothesis tags are found
                rule_pattern = r"(?:\*\*Rule \d+\*\*|Rule \d+):\s*(.+)"
                rules = re.findall(rule_pattern, result, re.IGNORECASE)
                
                if rules:
                    rule_text = "\n".join([f"Rule {i+1}: {rule}" for i, rule in enumerate(rules)])
                    hypotheses.append(rule_text)
                else:
                    # If no structured content found, use the full result
                    hypotheses.append(result)

        return hypotheses
    
    def remove_think_tags(self, text):
        """Remove all content enclosed in <think></think> tags from the text."""
        pattern = r'<think>.*?</think>'
        # Use re.DOTALL to make the dot match newlines as well
        cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
        return cleaned_text
    
    # def get_icl_in(self, x: str) -> str:
    #     """Get in-context learning input from the prompt.

    #     Args:
    #         x (str): The input text.

    #     Returns:
    #         str: The in-context learning input examples joined by newlines.
    #     """
    #     # Match patterns like "**Input:**\nACGT..." and capture the sequence
    #     pattern = self.icl_in_regex
    #     matches = re.findall(pattern, x, re.DOTALL)
        
    #     # Format each match with the "**Input:**" header and join with newlines
    #     if matches:
    #         return "\n".join([f"Input {i+1}:\n{match}" for i, match in enumerate(matches)])
    #     return ""
    
    def get_icl_in(self, x: str) -> Tuple[str, List[str]]:
        """Get in-context learning input from the prompt.

        Args:
            x (str): The input text.

        Returns:
            str: The in-context learning input examples joined by newlines.
        """
        # Match patterns like "**Input:**\nACGT..." and capture the sequence
        pattern = self.icl_in_regex
        matches = re.findall(pattern, x, re.DOTALL)
        
        # Format each match with the "**Input:**" header and join with newlines
        if matches:
            formatted = "\n".join([f"Input {i+1}:\n{match}" for i, match in enumerate(matches)])
            return formatted, matches
        return ""

    def get_icl_out(self, x: str) -> str:
        """Get in-context learning output from the prompt.

        Args:
            x (str): The input text.

        Returns:
            str: The in-context learning output examples joined by newlines.
        """
        # Match patterns like "**Output:**\nACGT..." and capture the sequence
        pattern = self.icl_out_regex
        matches = re.findall(pattern, x, re.DOTALL)
        
        # Format each match with the "**Output:**" header and join with newlines
        if matches:
            return "\n".join([f"Output {i+1}:\n{match}" for i, match in enumerate(matches)])
        return ""
    
    def apply_rule(self, x: str, hypothesis: str) -> List[str]:
        prompt = f"""Below is a full prompt about the reasoning task, which includes the question that you should give the corresponding answer.

Your task is:
1. Read the full prompt to understand the task and identify the specific input question to answer.
2. Based on your understanding of the given rules, generate the corresponding output fot the question.


Rules:
{hypothesis}

Full prompt:
{x}

Enclose your answer with <answer></answer> tags.
"""
        result = self.model.generate_from_template(
            [prompt],
            max_out_len=self.max_out_len,
            **self.generation_kwargs
        )[0]

        clean_result = self.remove_think_tags(result)

        logger.info(f"Applying rule to input, obtained result: {clean_result}")
    
        # Extract content between <answer> tags
        pattern = r'<answer>(.*?)</answer>'
        match = re.search(pattern, clean_result, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        else:
            # If no answer tags found, return the original result
            return clean_result
        
    def apply_final_rule_batch(self, x: str, hypotheses: List[str]) -> List[str]:
        """Apply multiple hypotheses to an input in batch.
        
        Args:
            x (str): The input text.
            hypotheses (List[str]): List of hypotheses to apply.
            
        Returns:
            List[str]: List of results for each hypothesis.
        """
        # Create a prompt for each hypothesis
        prompts = []
        for hypothesis in hypotheses:
            prompt = f"""Below is a full prompt about the reasoning task, which includes the question that you should give the corresponding answer.

Your task is:
1. Read the full prompt to understand the task and identify the specific input question to answer.
2. Based on your understanding of the given rules, generate the corresponding output fot the question.


Rules:
{hypothesis}

Full prompt:
{x}

Enclose your answer with <answer></answer> tags.
"""
            prompts.append(prompt)
        
        # Generate responses for all prompts in one batch
        results = self.model.generate_from_template(
            prompts,
            max_out_len=self.max_out_len,
            **self.generation_kwargs
        )
        
        # Process each result to extract the answer
        answers = []
        for result in results:
            clean_result = self.remove_think_tags(result)
            logger.info(f"Applying rule to input, obtained result: {clean_result}")
            
            # Extract content between <answer> tags
            pattern = r'<answer>(.*?)</answer>'
            match = re.search(pattern, clean_result, re.DOTALL)
            
            if match:
                answers.append(match.group(1).strip())
            else:
                # If no answer tags found, return the original result
                answers.append(clean_result)
        
        return answers
        
    def apply_rule_by_batch(self, hypotheses: List[str], icl_in: str) -> List[str]:
        prompt = """### Task Description
{task_description}

Please apply the given hypothesis to the given list of inputs. Ensure that you provide the actual output for each input. Do not give a program, partial output, or placeholder.

### Hypothesis
{hypothesis}

### Input
{icl_in}

Format your output as follows:

<output>
Output 1: ...
Output 2: ...
...
</output>

"""
        prompts = []
        for hypothesis in hypotheses:
            prompts.append(prompt.format(hypothesis=hypothesis, icl_in=icl_in, task_description=self.task_description))

        results = self.model.generate_from_template(
            prompts,
            max_out_len=self.max_out_len,
            **self.generation_kwargs
        )

        outputs = []
        
        for i, result in enumerate(results):
            logger.info(f"Prompt for applying rule: {prompts[i]}")
            clean_result = self.remove_think_tags(result)
            logger.info(f"Applying rule to input, obtained result: {clean_result}")
            
            # First extract content between <output> tags
            output_block_pattern = r"<output>(.*?)</output>"
            output_block_match = re.search(output_block_pattern, clean_result, re.DOTALL)
            
            # If <output> tags are found, use only the content within them
            if output_block_match:
                content_to_parse = output_block_match.group(1).strip()
            else:
                # If no <output> tags, use the entire result
                content_to_parse = clean_result
            
            # Extract outputs using regex pattern
            pattern = r"Output (\d+):\s*(.*?)(?=\nOutput \d+:|$)"
            matches = re.findall(pattern, content_to_parse, re.DOTALL)
            
            if matches:
                # Create a dictionary to organize outputs by number
                output_dict = {int(num): output.strip() for num, output in matches}
                
                # Format the outputs in order
                max_output_num = max(output_dict.keys()) if output_dict else 0
                formatted_output = "\n".join([f"Output {i}: {output_dict.get(i, '')}" 
                                            for i in range(1, max_output_num + 1)])
                outputs.append(formatted_output)
            else:
                # If no pattern match, use the original content as fallback
                outputs.append(content_to_parse.strip())
        
        return outputs

    def refine_hypothesis(self, x: str, hypothesis: str, generated_output, expected_output, icl_in) -> List[str]:
        """Refine a hypothesis based on feedback.

        Args:
            x (str): The input text.
            hypothesis (str): The hypothesis to refine.

        Returns:
            List[str]: List of refined hypotheses.
        """
        if self.task_name == 'dna_translate':
            note = " Please provide all the codon to amino acid mappings."
        else:
            note = ""

        prompt = f"""You are given a candidate hypothesis that attempts to explain how each input is transformed into its output. A hypothesis consists of rules that explain how the inputs are mapped to the outputs. Your goal is to revise this hypothesis so it fully accounts for any discrepancies. You may add new rules, modify existing ones, or remove inaccurate ones. You can also propose a completely new hypothesis.

Context:
{self.task_description}

Current Hypothesis:
{hypothesis}

Input:
{icl_in}

Model Output:
{generated_output}

Expected Output:
{expected_output}


Steps:
1. List the exact differences between Model Output and Expected Output.
2. For each difference, identify which existing rule (if any) fails to cover it.
3. Revise existing rules or introduce new rules to fix these gaps.
4. Ensure the rules clearly state how the input is mapped into output in a detailed manner.{note}

Output only the refined hypothesis—do not solve the original task.

Format your output as follows:
<new_hypothesis>
Rule 1: ...
Rule 2: ...
Rule 3: ...
...
</new_hypothesis>
"""
        logger.info(f"Refining hypothesis with prompt: {prompt}")
        refined_hypotheses = []
        
        for i in range(self.n_hypotheses):
            results = self.model.generate_from_template(
                [prompt],
                max_out_len=self.max_out_len,
                **self.generation_kwargs)

            # Remove think tags first
            clean_result = self.remove_think_tags(results[0])
            logger.info(f"Refined hypothesis result {i+1}: {clean_result}")
            
            # Extract the refined hypothesis using regex
            pattern = r"<new_hypothesis>(.*?)</new_hypothesis>"
            matches = re.search(pattern, clean_result, re.DOTALL)
            
            if matches:
                refined = matches.group(1).strip()
                refined_hypotheses.append(refined)
            else:
                # If no pattern match, use the original result (fallback)
                refined_hypotheses.append(clean_result)

        return refined_hypotheses
    
    
    def rate_output_with_evaluator(self, outputs, icl_out):
        # Parse reference outputs from icl_out
        ref_pattern = r"Output (\d+):\s*(.*?)(?=\nOutput \d+:|$)"
        ref_matches = re.findall(ref_pattern, icl_out, re.DOTALL)
        
        # Create an ordered dictionary of reference outputs
        ref_dict = {int(num): output.strip() for num, output in ref_matches}
        reference_list = [ref_dict[i] for i in sorted(ref_dict.keys())]

        logger.info(f"Reference outputs: {reference_list}")

        # Evaluate each output
        scores = []
        for i, output_text in enumerate(outputs):
            # Parse outputs from this candidate
            pred_matches = re.findall(ref_pattern, output_text, re.DOTALL)
            pred_dict = {int(num): output.strip() for num, output in pred_matches}
            
            # Create list matching the order of reference outputs
            prediction_list = []
            for j in sorted(ref_dict.keys()):
                if j in pred_dict:
                    prediction_list.append(pred_dict[j])
                else:
                    prediction_list.append("")  # Empty string for missing outputs

            logger.info(f"Evaluating output {i+1}: {prediction_list}")
            
            # Score this output against the reference
            result = self.evaluator.score(prediction_list, reference_list)
            scores.append(result['accuracy'])
            logger.info(f"Output {i+1} score: {result['accuracy']}")
        
        # Find best score
        best_index = scores.index(max(scores))
        is_perfect = scores[best_index] == 100.0
        
        logger.info(f"Best output index: {best_index}")
        logger.info(f"Best output score: {scores[best_index]}")
        logger.info(f"Is perfect: {is_perfect}")
        
        return best_index, is_perfect, scores[best_index]
    
    def rate_output_with_evaluator_chem(self,outputs, icl_out, descriptions_or_smiles):
        # Parse reference outputs from icl_out
        ref_pattern = r"Output (\d+):\s*(.*?)(?=\nOutput \d+:|$)"
        ref_matches = re.findall(ref_pattern, icl_out, re.DOTALL)
        
        # Create an ordered dictionary of reference outputs
        ref_dict = {int(num): output.strip() for num, output in ref_matches}
        if self.task_name == 'chem_molecule_design' or self.task_name == 'chem_molecule_caption':
            reference_list_inital = [ref_dict[i] for i in sorted(ref_dict.keys())]
            logger.info(f"descriptions_or_smiles:{descriptions_or_smiles}\n")
            logger.info(f"reference_list_inital:{reference_list_inital}\n")
            reference_list = [f"{desc.strip()}-->{smiles.strip()}"for desc, smiles in zip(descriptions_or_smiles, reference_list_inital)]
    
        else:
            reference_list = [ref_dict[i] for i in sorted(ref_dict.keys())]

        # Evaluate each output
        scores = []
        for i, output_text in enumerate(outputs):
            # Parse outputs from this candidate
            pred_matches = re.findall(ref_pattern, output_text, re.DOTALL)
            pred_dict = {int(num): output.strip() for num, output in pred_matches}
            
            # Create list matching the order of reference outputs
            prediction_list = []
            for j in sorted(ref_dict.keys()):
                if j in pred_dict:
                    prediction_list.append(pred_dict[j])
                else:
                    prediction_list.append("")  # Empty string for missing outputs

            logger.info(f"Evaluating output {i+1}: {prediction_list}")
            logger.info(f"prediction_list: {prediction_list}")
            logger.info(f"refrence_list: {reference_list}")
            
            # Score this output against the reference
            result = self.evaluator.score(prediction_list, reference_list)
            if self.task_name == 'chem_molecule_design':
                logger.info(f"result:{result}")
                design_similarity_keys = ['maccs_sims', 'rdk_sims', 'morgan_sims']
                avg_score = sum(result[k] for k in design_similarity_keys) / len(design_similarity_keys)
                scores.append(avg_score)
                logger.info(f"Output {i+1} score: {avg_score}")
            elif self.task_name == 'chem_molecule_caption':
                logger.info(f"result:{result}")
                caption_similarity_keys = ['meteor_score','rouge_l','bleu4','llm_score']
                avg_score = sum(result[k] for k in caption_similarity_keys) / len(caption_similarity_keys)
                scores.append(avg_score)
                logger.info(f"Output {i+1} score: {avg_score}")
            else:
                scores.append(result['accuracy'])
                logger.info(f"Output {i+1} score: {result['accuracy']}")
        
        # Find best score
        best_index = scores.index(max(scores))
        is_perfect = scores[best_index] == 100.0
        
        logger.info(f"Best output index: {best_index}")
        logger.info(f"Best output score: {scores[best_index]}")
        logger.info(f"Is perfect: {is_perfect}")
        
        return best_index, is_perfect, scores[best_index]
        
    def rate_output_with_evaluator_dna_table(self, outputs, icl_in, icl_out):
        """Rate DNA table inferences based on their accuracy in translating DNA sequences.
        
        Args:
            outputs (List[str]): List of outputs containing predicted codon tables (in JSON format)
            icl_in (str): In-context learning input (DNA sequences)
            icl_out (str): In-context learning output (amino acid sequences)
            
        Returns:
            Tuple[int, bool]: Index of best hypothesis (0-based), and perfect flag
        """
        # Extract DNA sequences from icl_in
        dna_pattern = r"Input (\d+):\s*([ACGT]+)"
        dna_matches = re.findall(dna_pattern, icl_in, re.DOTALL)
        
        # Extract amino acid sequences from icl_out
        aa_pattern = r"Output (\d+):\s*([A-Z*]+)"
        aa_matches = re.findall(aa_pattern, icl_out, re.DOTALL)
        
        # Create examples for the evaluator
        examples = []
        dna_dict = {int(num): seq.strip() for num, seq in dna_matches}
        aa_dict = {int(num): seq.strip() for num, seq in aa_matches}
        
        for num in sorted(set(dna_dict.keys()) & set(aa_dict.keys())):
            examples.append({
                "seq": dna_dict[num],
                "aa": aa_dict[num]
            })
        
        logger.info(f"Created {len(examples)} examples for evaluation")
        
        # Evaluate each output
        scores = []
        generated_outputs = []
        for i, output in enumerate(outputs):
            # Extract JSON content from ```result``` tags
            json_pattern = r"```result\s*(.*?)```"
            json_match = re.search(json_pattern, output, re.DOTALL)
            
            if json_match:
                # Get the JSON content
                json_content = json_match.group(1).strip()
                logger.info(f"Evaluating output {i+1}: {json_content[:100]}...")  # Log first 100 chars to avoid huge logs
                
                # Score this output against the examples
                try:
                    result = self.evaluator.score(json_content, examples)
                    accuracy = result['accuracy']
                    interpreted_outputs = result['translations']
                    scores.append(accuracy)
                    logger.info(f"Output {i+1} score: {accuracy}")
                except Exception as e:
                    logger.warning(f"Error evaluating output {i+1}: {e}")
                    scores.append(0.0)  # Assign zero score for failed evaluations
            else:
                logger.warning(f"No JSON content found in output {i+1}")
                scores.append(0.0)  # Assign zero score for outputs without JSON content
        
        # Find best score
        if scores:
            best_index = scores.index(max(scores))
            is_perfect = scores[best_index] == 100.0
        else:
            best_index = 0
            is_perfect = False
        
        logger.info(f"Best output index: {best_index}")
        logger.info(f"Best output score: {scores[best_index] if scores else 0}")
        logger.info(f"Is perfect: {is_perfect}")
        
        return best_index, is_perfect, interpreted_outputs, scores[best_index]


    def rate_output(self, outputs, icl_out, x) -> Tuple[int, str, bool]:
        """Rate hypotheses based on their outputs compared with expected outputs.
        
        Args:
            hypotheses (List[str]): List of hypothesis strings
            outputs (List[str]): List of outputs generated by applying each hypothesis
            icl_out (str): Expected outputs to compare against
            
        Returns:
            Tuple[int, str, bool]: Index of best hypothesis (0-based), and perfect flag
        """
        all_outputs = ""
        # for i, (hypothesis, output) in enumerate(zip(hypotheses, outputs)):
        #     all_outputs += f"Hypothesis {i+1}:\n{hypothesis}\nResults:\n{output}\n\n"
        for i, output in enumerate(outputs):
            all_outputs += f"Response {i+1}:\n{output}\n\n"
        

        prompt = f"""Context:
{x}

Below are several generated responses. Select the single response whose output is closest to the ground truth. If one matches exactly, mark it as perfect; otherwise, choose the best.

### Generated Responses

{all_outputs}

### Ground Truth

{icl_out}

Return exactly in this format (no additional text):
Best Response: <index of the best response>
Is perfect: <yes/no>
"""
        logger.info(f"Rating hypotheses with prompt: {prompt}")

        local_kwargs = dict(self.generation_kwargs)
        local_kwargs['temperature'] = 0.001

        # Get rating from model
        result = self.model.generate_from_template(
            [prompt],
            max_out_len=self.max_out_len,
            **local_kwargs
        )[0]
        
        # Clean and extract information
        clean_result = self.remove_think_tags(result)
        logger.info(f"Rating result: {clean_result}")
        
        # Extract best hypothesis index
        index_pattern = r"Best Response:\s*(\d+)"
        index_match = re.search(index_pattern, clean_result)
        
        # Extract "is perfect" flag
        perfect_pattern = r"Is perfect:\s*(yes|no)"
        perfect_match = re.search(perfect_pattern, clean_result, re.IGNORECASE)
        
        # Process results
        if index_match:
            # Convert to 0-based index (hypotheses are 1-indexed in the prompt)
            best_index = int(index_match.group(1)) - 1
            # Determine if solution is perfect
            is_perfect = False
            if perfect_match:
                is_perfect = perfect_match.group(1).lower() == 'yes'
            
            return best_index, is_perfect
        else:
            # Default to first hypothesis if extraction fails
            logger.warning("Failed to extract best hypothesis index or feedback")
            return 0, clean_result, False



    def select_best_hypothesis(self, hypotheses: List[str], x: str, icl_in: str, icl_out: str, matches: str = None) -> Tuple[str, str, bool]:
        """Select the best hypothesis based on evaluation.

        Args:
            hypotheses (List[str]): List of hypotheses to evaluate.
            x (str): The input text.
            icl_in (str): Additional input examples for testing hypotheses.
            icl_out (str): Expected outputs for the test examples.
            matches (str, optional): Input descriptions or SMILES strings for chemistry tasks.

        Returns:
            Tuple[str, str, bool]: The best hypothesis, whether it's perfect, and the output.
        """
        best_hypothesis = None
        best_score = -1

        # Apply rules based on task type
        if self.task_name == 'dna_table_infer':
            outputs = self.apply_final_rule_batch(x, hypotheses)
        elif self.task_name == 'dna_translate' or self.task_name == 'dna_transform':
            outputs = self.apply_rule_with_code(hypotheses, icl_in)
        else:
            outputs = self.apply_rule_by_batch(hypotheses, icl_in)

        # Rate outputs based on task type
        if self.task_name == 'dna_table_infer':
            best_hypothesis_index, is_perfect, interpreted_outputs, best_score = self.rate_output_with_evaluator_dna_table(outputs, icl_in, icl_out)
        elif self.task_name == 'chem_molecule_design' or self.task_name == 'chem_molecule_caption':
            descriptions_or_smiles = matches
            best_hypothesis_index, is_perfect, best_score = self.rate_output_with_evaluator_chem(outputs, icl_out, descriptions_or_smiles)
        else:
            best_hypothesis_index, is_perfect, best_score = self.rate_output_with_evaluator(outputs, icl_out)
        
        best_hypothesis = hypotheses[best_hypothesis_index]
        
        logger.info(f"Best hypothesis index: {best_hypothesis_index + 1}")
        logger.info(f"Best hypothesis: {best_hypothesis}")
        logger.info(f"Is perfect: {is_perfect}")

        # Return appropriate output based on task type
        if self.task_name == 'dna_table_infer':
            return best_hypothesis, is_perfect, "\n".join([f"Output {i+1}:\n{out}" for i, out in enumerate(interpreted_outputs)]), best_score
        else:
            return best_hypothesis, is_perfect, outputs[best_hypothesis_index], best_score


    def hr_solve(self, x: str) -> str:
        """Solve a problem using Hypothesis Refinement.

        Args:
            x (str): The input text to be solved.

        Returns:
            str: Final hypothesis.
        """
        logger.info(f"Starting hypothesis refinement")
        icl_in,matches = self.get_icl_in(str(x))
        icl_out = self.get_icl_out(str(x))
        logger.info(f"ICL In: {icl_in}")
        logger.info(f"ICL Out: {icl_out}")
        
        # Generate initial hypotheses
        hypotheses = self.generate_hypotheses(x)
        logger.info(f"Generated {len(hypotheses)} hypotheses.")

        prev_best_score = -1
        curr_best_score = -1
        prev_best_hypothesis = None
        
        for iteration in range(self.max_iterations):
            logger.info(f"\n-- Iteration {iteration+1}/{self.max_iterations} --\n")

            # print out all the generated hypotheses
            for i, hypothesis in enumerate(hypotheses):
                logger.info(f"Hypothesis {i+1}: {hypothesis}")

            # Select the best hypothesis
            best_hypothesis, is_perfect, best_output, curr_best_score = self.select_best_hypothesis(
                hypotheses, str(x), icl_in, icl_out, matches)
            
            if curr_best_score < prev_best_score:
                # best_hypothesis = prev_best_hypothesis
                logger.info("early stopping")
                final_answer = self.apply_rule(str(x), prev_best_hypothesis)
                logger.info(f"Final answer: {final_answer}")
                return final_answer
            else:
                prev_best_score = curr_best_score
                prev_best_hypothesis = best_hypothesis
                
            
            
            # If perfect hypothesis found, return it
            if is_perfect:
                logger.info("Perfect hypothesis found!")
                final_answer = self.apply_rule(str(x), best_hypothesis)
                logger.info(f"Final answer: {final_answer}")
                return final_answer
            
            hypotheses = self.refine_hypothesis(str(x), best_hypothesis, best_output, icl_out, icl_in)
        
        # print out all the generated hypotheses
        for i, hypothesis in enumerate(hypotheses):
            logger.info(f"Hypothesis {i+1}: {hypothesis}")

        # Return the best hypothesis after all iterations
        best_hypothesis, _, _, _ = self.select_best_hypothesis(hypotheses, str(x), icl_in, icl_out, matches)

        logger.info(f"Final best hypothesis: {best_hypothesis}")

        final_answer = self.apply_rule(str(x), best_hypothesis)

        logger.info(f"Final answer: {final_answer}")

        return final_answer

    def inference(self,
                  retriever: BaseRetriever,
                  ice_template: Optional[PromptTemplate] = None,
                  prompt_template: Optional[PromptTemplate] = None,
                  output_json_filepath: Optional[str] = None,
                  output_json_filename: Optional[str] = None) -> List:
        """Run inference using Hypothesis Refinement.

        Args:
            retriever (BaseRetriever): Retriever for in-context examples.
            ice_template (PromptTemplate, optional): Template for ice.
            prompt_template (PromptTemplate, optional): Template for prompt.
            output_json_filepath (str, optional): Output file path.
            output_json_filename (str, optional): Output file name.

        Returns:
            List: Predictions.
        """

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
        logger.info('Starting Hypothesis Refinement inference process...')
        for datum in tqdm(dataloader, disable=not self.is_main_process):
            if ds_reader.output_column:
                entries, golds = list(zip(*datum))
            else:
                entries = datum
                golds = [None for _ in range(len(entries))]
                
            # 5-1. Inference with HR and local model
            with torch.no_grad():
                parsed_entries = self.model.parse_template(entries, mode='gen')
                generated = [self.hr_solve(entry) for entry in entries]

            # 5-2. Save current output
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
    
    def apply_rule_with_code(self, hypotheses: List[str], icl_in: str) -> List[str]:
        """Apply hypotheses by converting them to executable Python code.
        
        Args:
            hypotheses (List[str]): List of hypotheses to convert to Python code.
            icl_in (str): In-context learning input examples.
            
        Returns:
            List[str]: List of formatted outputs for each hypothesis.
        """
        import tempfile
        import subprocess
        from concurrent.futures import ThreadPoolExecutor, TimeoutError
        
        # Parse input examples
        input_pattern = r"Input (\d+):\s*(.*?)(?=\nInput \d+:|$)"
        input_matches = re.findall(input_pattern, icl_in, re.DOTALL)
        inputs = {int(num): inp.strip() for num, inp in input_matches}
        
        outputs = []
        
        for i, hypothesis in enumerate(hypotheses):
            logger.info(f"Converting hypothesis {i+1} to executable code")
            
            # Generate Python code from hypothesis
            code_prompt = f"""Convert the following hypothesis into a Python function called 'apply' that takes a string input and returns the transformed output. 
The function should implement the rules described in the hypothesis. Make sure to handle all the transformations correctly.

A task description is provided to give context to the task. Your task is to implement the hypothesis as a Python function. Do not add any transformation outside the hypothesis.

Task Description:
{self.task_description}

Hypothesis:
{hypothesis}

Your function should follow this template:
```python
def apply(input_str):
    # Implementation based on the hypothesis rules
    # ...
    return result
```

Return ONLY the Python code without any explanation or markdown formatting.
"""
            # Generate code using the model
            code_result = self.model.generate_from_template(
                [code_prompt],
                max_out_len=self.max_out_len,
                temperature=0.1,  # Lower temperature for more deterministic code generation
                **{k: v for k, v in self.generation_kwargs.items() if k != 'temperature'}
            )[0]
            
            # Extract code from the result
            code_pattern = r"```python\s*(.*?)```"
            code_match = re.search(code_pattern, code_result, re.DOTALL)
            if not code_match:
                code_pattern = r"def apply\(.*?\):.*?(?=\n\n|$)"
                code_match = re.search(code_pattern, code_result, re.DOTALL)
            
            if code_match:
                function_code = code_match.group(1) if '```python' in code_result else code_match.group(0)
            else:
                # If no code pattern found, assume the entire result is the function
                function_code = code_result
                
            logger.info(f"Generated code:\n{function_code}")
            
            # Create a complete script with the function and test code
            full_script = f"""
{function_code}

# Process each input and collect outputs
results = {{}}
try:
    for input_num, input_text in {inputs}.items():
        try:
            output = apply(input_text)
            results[input_num] = output
        except Exception as e:
            results[input_num] = f"Error: {{str(e)}}"
except Exception as e:
    print(f"Global error: {{str(e)}}")

# Format the results
for i in sorted(results.keys()):
    print(f"Output {{i}}: {{results[i]}}")
"""
            
            # Execute the script safely with timeout
            with tempfile.NamedTemporaryFile(suffix='.py', mode='w+', delete=False) as temp_file:
                temp_file_path = temp_file.name
                temp_file.write(full_script)
            
            try:
                # Run the script with a timeout
                process = subprocess.Popen(
                    ['python', temp_file_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                try:
                    stdout, stderr = process.communicate(timeout=10)  # 10 seconds timeout
                    if process.returncode != 0:
                        logger.warning(f"Code execution failed: {stderr}")
                        # Create a fallback output for failed execution
                        formatted_output = "\n".join([f"Output {i}: Error executing code" for i in sorted(inputs.keys())])
                    else:
                        formatted_output = stdout.strip()
                except subprocess.TimeoutExpired:
                    process.kill()
                    logger.warning("Code execution timed out")
                    formatted_output = "\n".join([f"Output {i}: Execution timed out" for i in sorted(inputs.keys())])
                    
            except Exception as e:
                logger.warning(f"Error during code execution: {str(e)}")
                formatted_output = "\n".join([f"Output {i}: Error: {str(e)}" for i in sorted(inputs.keys())])
            
            finally:
                # Clean up
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            
            outputs.append(formatted_output)
            logger.info(f"Code execution result for hypothesis {i+1}:\n{formatted_output}")
        
        return outputs