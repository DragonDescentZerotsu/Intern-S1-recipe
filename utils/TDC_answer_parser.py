import re

# ====== 解析工具：从模型文本中剥离最终选项并映射到 0/1 ======
def extract_answer(response:str):
    """从模型回答中提取最后一个Answer:之后的内容"""
    # 找到所有"Answer:"的位置
    format_correct = False
    if 'Answer:' in response:
        format_correct = True
        answer_matches = list(re.finditer(r'Answer:', response, re.IGNORECASE))
        if not answer_matches:
            return None, format_correct
    elif 'answer is' in response:
        format_correct = True
        answer_matches = list(re.finditer(r'answer is', response, re.IGNORECASE))
        if not answer_matches:
            return None, format_correct
    elif '<answer>' in response:
        format_correct = True
        answer_matches = list(re.finditer(r'<answer>', response, re.IGNORECASE))
        if not answer_matches:
            return None, format_correct

    # 获取最后一个"Answer:"之后的内容
    if 'Answer:' in response or 'answer is' in response or '<answer>' in response:
        last_answer_pos = answer_matches[-1].end()
        answer_text = response[last_answer_pos:].strip()
    else:
        answer_text = response

    return answer_text, format_correct

def parse_prob_answer(answer_text, format_correct, think_is_on:bool=True):
    """
    解析计分制的答案（0-100 probability）。
    提取出最后出现的数字作为预测分数。
    考虑到宏F1（macro F1）的计算，需要将概率阈值化为二分类：
    Probability < 50 -> 0 (负类)
    Probability >= 50 -> 1 (正类)
    """
    if answer_text is None:
        return None
        
    text_to_search = answer_text
    
    # 兼容思绪标签如果没有写完整的Answer的情况
    if think_is_on and not format_correct:
        if '</think>' in answer_text:
            text_to_search = answer_text.rsplit('</think>')[-1].strip()

    # 正则提取文本中 0 到 100 范围的整数
    # 匹配独立的数字 0-100
    numbers = re.findall(r'\b(100|[1-9]?[0-9])\b', text_to_search)
    
    if numbers:
        # 取最后一个出现的数字作为模型最终落定的概率
        prob = int(numbers[-1])
        # 按照 50 为阈值映射回二分类标签
        return 1 if prob >= 50 else 0
    
    return None

def parse_answer(answer_text, format_correct, think_is_on:bool=True, score_based:bool=False):
    """解析答案: (A) -> 0 (负类), (B) -> 1 (正类)"""
    if score_based:
        return parse_prob_answer(answer_text, format_correct, think_is_on)
        
    if think_is_on:
        if answer_text is None:
            return None
        if format_correct:
            if '(A)' in answer_text:
                return 0
            elif 'A**' in answer_text:
                return 0
            elif 'A)' in answer_text:
                return 0
            elif '\\boxed{A}' in answer_text:
                return 0
            elif '\\text{A}' in answer_text:
                return 0
            elif '(B)' in answer_text:
                return 1
            elif 'B**' in answer_text:
                return 1
            elif 'B)' in answer_text:
                return 1
            elif '\\boxed{B}' in answer_text:
                return 1
            elif '\\text{B}' in answer_text:
                return 1
            elif ' B ' in answer_text:
                return 1
            elif ' A ' in answer_text:
                return 0
            elif ' B.' in answer_text:
                return 1
            elif ' A.' in answer_text:
                return 0
            elif 'B' in answer_text:
                return 1
            elif 'A' in answer_text:
                return 0
            else:
                return None
        else:
            if '</think>' in answer_text:  # more robust for answer matching without correct "Answer:" flags
                answer_text = answer_text.rsplit('</think>')[1].strip()
            if '\\boxed{A}' in answer_text:
                return 0
            elif '\\text{A}' in answer_text:
                return 0
            elif '\n(A)' in answer_text:
                return 0
            elif '\\boxed{B}' in answer_text:
                return 1
            elif '\\text{B}' in answer_text:
                return 1
            elif '\n(B)' in answer_text:
                return 1
            else:
                return None

    else:
        if answer_text is None:
            return None
        if '(A)' in answer_text:
            return 0
        elif '(B)' in answer_text:
            return 1
        elif 'Yes' in answer_text:
            return 1
        elif 'yes' in answer_text:
            return 1
        elif 'B' in answer_text:
            return 1
        elif 'A' in answer_text:
            return 0
        else:
            return None