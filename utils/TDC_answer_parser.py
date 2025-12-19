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

    # 获取最后一个"Answer:"之后的内容
    if 'Answer:' in response or 'answer is' in response:
        last_answer_pos = answer_matches[-1].end()
        answer_text = response[last_answer_pos:].strip()
    else:
        answer_text = response

    return answer_text, format_correct

def parse_answer(answer_text, format_correct, think_is_on:bool):
    """解析答案: (A) -> 0 (负类), (B) -> 1 (正类)"""
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
            elif 'B' in answer_text:
                return 1
            elif 'A' in answer_text:
                return 0
            else:
                return None
        else:
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