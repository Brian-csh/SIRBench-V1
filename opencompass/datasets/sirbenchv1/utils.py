import random
import re
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    RetryError,
)
from Bio.Data.CodonTable import NCBICodonTableDNA
from Bio.Seq import Seq


def extract_content(response):
    response = remove_think_tags(response)
    # 1. 先尝试匹配 ```result\n...``` 或 ```\nresult\n...```
    pattern_result = r'```(?:result\n|\nresult\n)(.*?)```'
    match_result = re.search(pattern_result, response, re.DOTALL)
    
    if match_result:
        return match_result.group(1).strip()  # 返回内容并去除首尾空白
    
    # 2. 如果无匹配，则找最后一个 ```...```
    pattern_generic = r'```(?:[^\n]*\n)?(.*?)```'
    matches_generic = list(re.finditer(pattern_generic, response, re.DOTALL))
    
    if matches_generic:
        last_match = matches_generic[-1]  # 取最后一个匹配
        return last_match.group(1).strip()
    
    # 3. 如果仍无匹配，返回整个 response
    print("No code block found, using the whole response")
    return response.strip()

# 选n/2个正面例和n/2个负面例子
def get_examples_for_test(data: list[dict],n,seed=42):
    random.seed(seed)
    positive_examples = [ i for i in data if i['anwser']==True]
    negative_examples = [ i for i in data if i['anwser']==False]

    # 正负例各一半
    positive_examples_num = n // 2
    negative_examples_num = n - positive_examples_num

    sample_positive_examples = positive_examples[:positive_examples_num]
    sample_negative_examples = negative_examples[:negative_examples_num]



    return sample_positive_examples + sample_negative_examples



def extract_json_from_codeblock(text):
    """Extract JSON content from within markdown code blocks.
    
    Args:
        text: Text potentially containing ```json ... ``` codeblocks
        
    Returns:
        The extracted JSON content or the original text if no codeblock found
    """
    json_pattern = r'```(?:json)?\s*([\s\S]*?)```'
    match = re.search(json_pattern, text)
    if match:
        return match.group(1).strip()
    return text


def remove_think_tags(text):
    """Remove all content enclosed in <think></think> tags from the text."""
    pattern = r'<think>.*?</think>'
    # Use re.DOTALL to make the dot match newlines as well
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
    return cleaned_text



def get_response(model: str, prompt: str):
    client = OpenAI(
        api_key='sk-phj5Mukj06LpMiNZ9961017e3cE34bCa9f5827C2A8042938',
        base_url='https://yeysai.com/v1/'
    )
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        reraise=True
    )
    def _make_api_call():
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
    
    try:
        return _make_api_call()
    except RetryError as e:
        raise
    except Exception as e:
        raise


def extract_difference_equation(response):
    output = remove_think_tags(response)

    extraction_prompt = f'''Analyze the following text and determine the final answer, including the difference equation and initial values.

For example, if the output is "The corresponding difference equation is y[n] = 2y[n-1] + 3y[n-2] + 5, and the initial values are y[0] = 1, y[1] = 2", then your output should be as follows:
```json
{{"coefficients": [2, 3], "initial_values": [1, 2], "constant_term": 5}}
```

Text: {output}

Return only the JSON object.
'''
    extraction_response = get_response("gpt-4o", extraction_prompt)
    extraction_output = extraction_response.choices[0].message.content.strip()

    return extraction_output


def extract_difference_equation_string(response):
    output = remove_think_tags(response)

    prompt = f'''You will see an output from a large language model solving a difference equation. Please omit all reasoning steps and extract the corresponding difference equation and initial values, wrapping them in <\equation> and <\initial> tags.  
    For example, if the output is "The corresponding difference equation is y[n] = 2y[n-1] + 3y[n-2], with initial values y[0] = 1, y[1] = 2," your output should be: "<\equation>y[n] = 2y[n-1] + 3y[n-2]<\equation> <\initial>1 2<\initial>".  
    If the model ultimately does not provide the difference equation or initial values, output <\equation>None<\equation> <\initial>None<\initial>.  

    Output:  
    {remove_think_tags(output)}
    '''
    res = get_response('gpt-4o', prompt)
    pattern = re.compile(r'<\\equation>(.*?)<\\equation> <\\initial>(.*?)<\\initial>')
    match = pattern.match(res.choices[0].message.content)
    if len(match.groups()) != 2:
        return None
    return match.groups()


def extract_scaling_law(response):
    output = remove_think_tags(response)


    # extract the coefficients from the output
    extraction_prompt = f'''Analyze the following text and find the final answer equation. Determine if it's in the form y = a*e^(b*x) + c or equivalent.
If it is, extract the coefficients a, b, and c as floating point numbers.

Text: {output}

Return the extracted content in valid JSON format like the following:
```json
{{"type": "exponential", "a": [value of a], "b": [value of b], "c": [value of c]}}
```

If it's not an exponential equation or cannot be parsed, return:
```json
{{"type": "error", "message": "Not exponential"}}
```

Return only the JSON object.
'''
    extraction_response = get_response("gpt-4o-mini", extraction_prompt)
    extraction_output = extraction_response.choices[0].message.content.strip()

    return extraction_output

def generate_random_codon_table(start_codons=None, stop_codons=None):
    """
    生成一个随机的密码子表，如果传入起始密码子或终止密码子，则确保这些密码子在表中，并补充1-3个额外的起始密码子或终止密码子。
    """
    # 可用的所有密码子
    bases = ['A', 'T', 'C', 'G']
    all_codons = [b1 + b2 + b3 for b1 in bases for b2 in bases for b3 in bases]
    
    # 如果传入了起始密码子，确保这些密码子分配，并补充1到3个随机的起始密码子
    if start_codons:
        start_codons = list(set(start_codons))  # 去重
        if len(start_codons) < 3:
            # 补充1到3个起始密码子
            additional_start_codons = random.sample(
                [codon for codon in ["ATG", "TTG", "CTG", "GTG", "ATA"] if codon not in start_codons],
                random.randint(1, 3 - len(start_codons)))
            start_codons.extend(additional_start_codons)
    else:
        # 随机选择1到3个起始密码子
        start_codons = random.sample(
            ["ATG", "TTG", "CTG", "GTG", "ATA"], random.randint(1, 3))

    # 如果传入了终止密码子，确保这些密码子分配，并补充1到3个随机的终止密码子
    if stop_codons:
        stop_codons = list(set(stop_codons))  # 去重
        if len(stop_codons) < 3:
            # 补充1到3个终止密码子
            additional_stop_codons = random.sample(
                [codon for codon in ["TAA", "TAG", "TGA", "TCA", "AGG"] if codon not in stop_codons],
                random.randint(1, 3 - len(stop_codons)))
            stop_codons.extend(additional_stop_codons)
    else:
        # 随机选择1到3个终止密码子
        stop_codons = random.sample(
            ["TAA", "TAG", "TGA", "TCA", "AGG"], random.randint(1, 3))

    # 从所有密码子中移除终止密码子
    for stop_codon in stop_codons:
        if stop_codon in all_codons:
            all_codons.remove(stop_codon)

    # 20种标准氨基酸
    amino_acids = [
        "F", "L", "I", "M", "V", "C", "A", "G", "P", "T", 
        "Y", "H", "Q", "R", "N", "K", "S", "W", "D", "E"
    ]
    
    # 随机分配密码子
    codon_table = {}
    
    # 给起始密码子分配氨基酸（起始密码子都分配"M"）
    for start_codon in start_codons:
        # codon_table[start_codon] = "M"
        if start_codon =="ATG":
            codon_table[start_codon] = "M"
        else:
            codon_table[start_codon] = random.choice(["M","L","I","V"])
        all_codons.remove(start_codon)  # 从可用的密码子列表中移除已分配的密码子

    # 随机分配其余密码子
    for aa in amino_acids:
        num_codons = random.randint(1, 3)  # 每个氨基酸分配1到4个密码子
        selected_codons = random.sample(all_codons, num_codons)
        for codon in selected_codons:
            codon_table[codon] = aa

            all_codons.remove(codon)  # 从可用的密码子列表中移除已分配的密码子

    # 确保所有剩余密码子都被分配
    for codon in all_codons:
        random_aa = random.choice(amino_acids)  # 随机选择一个氨基酸
        codon_table[codon] = random_aa

    # 返回 NCBICodonTableDNA 对象
    return NCBICodonTableDNA(
        id=random.randint(1000, 9999),  # 随机 ID
        names=["Random Codon Table"],
        table=codon_table,
        start_codons=start_codons,
        stop_codons=stop_codons  # 终止密码子
    )

def generate_example_sequences(codon_table, n=3, max_len=400, min_len=200):
        """
        根据密码表生成例子 [{"seq": seq, "aa": protein}]
        :param codon_table: 密码表对象
        :param n: 生成的例子数量
        :param max_len: 生成的 DNA 序列的最大长度
        :param min_len: 生成的 DNA 序列的最小长度
        """
        examples = []

        for _ in range(n):
            # 起始密码子
            start_codon = random.choice(codon_table.start_codons)
            # 终止密码子
            stop_codon = random.choice(codon_table.stop_codons)

            # 随机生成 DNA 序列
            dna_sequence = start_codon
            while len(dna_sequence) < max_len - len(stop_codon):
                # 随机选择一个密码子
                codon = random.choice(list(codon_table.forward_table.keys()))
                # 避免终止密码子出现在中间
                if codon in codon_table.stop_codons:
                    continue
                dna_sequence += codon
                # 如果长度达到最小要求并随机决定是否结束
                if len(dna_sequence) >= min_len and random.random() < 0.1:
                    break
            dna_sequence += stop_codon

            # 翻译 DNA 序列为氨基酸序列
            protein_sequence = str(Seq(dna_sequence).translate(table=codon_table))

            # 添加到例子列表
            examples.append({"seq": dna_sequence, "aa": protein_sequence})

        return examples

def extract_next_term(response):
    """
    Extract the next term value from a model response.
    
    Args:
        response (str): The model's response text
        
    Returns:
        str or None: The extracted next term value as a string, or None if not found
    """
    output = remove_think_tags(response)
    
    # First try to find next term in a specific format
    pattern = r"[Nn]ext [Tt]erm:\s*(?:\{)?(-?\d+\.?\d*)(?:\})?"
    match = re.search(pattern, output, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # If not found, try to extract any number
    pattern = r"(-?\d+\.?\d*|\.\d+)"
    match = re.search(pattern, output)
    if match:
        return match.group(1)
        
    return response


def extract_scaling_law_prediction(response):
    output = remove_think_tags(response)
    pattern = r"[Pp]rediction:\s*(?:\{)?(-?\d+\.?\d*(?:[eE][-+]?\d+)?)(?:\})?"
    match = re.search(pattern, output, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    pattern = r"(-?\d+\.?\d*|\.\d+)"
    match = re.search(pattern, output)
    if match:
        return match.group(1)
    return None


def fact_retriever_difference_equation(prompt_text: str) -> str:
    """Extract the sequence terms line from the difference equation prompt.
    
    This function uses regex to find and extract the line of sequence terms
    that follows after "Sequence:" (with optional number).
    
    Args:
        prompt_text (str): The full prompt text containing the sequence
        
    Returns:
        str: The extracted sequence terms line
    """
    # Pattern to match "Sequence" (optionally followed by number and colon)
    # and then capture the next non-empty line
    pattern = r"Sequence(?:\s*\d*)?:?\s*\n\s*\n?\s*([^$\n]+)"
    
    match = re.search(pattern, prompt_text)
    if match:
        return match.group(1).strip()
    
    # Fallback if the pattern doesn't match
    return prompt_text


def formula_postprocess(response):
    output = remove_think_tags(response)
    output = output.replace("Molecular formula:", "")
    output = output.strip()
    output = re.sub(r'(?<=[A-Za-z])1(?!\d)', '', output)
    return output