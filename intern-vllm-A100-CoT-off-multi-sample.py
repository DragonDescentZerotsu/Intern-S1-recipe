import os, multiprocessing as mp
# ---------------- vLLM & CUDA 环境 ----------------
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TORCH_SDPA")
mp.set_start_method("spawn", force=True)

# 选择显卡
# os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

from transformers import AutoTokenizer, AutoProcessor
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import torch
import json
from huggingface_hub import hf_hub_download
from tdc.single_pred import ADME, Tox, HTS, Develop
from tdc.multi_pred.ppi import PPI
from tdc.multi_pred.tcr_epi import TCREpitopeBinding
from tdc.multi_pred.trialoutcome import TrialOutcome
from tdc.multi_pred.peptidemhc import PeptideMHC
from tdc.utils import retrieve_label_name_list
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np
import ast
import re
from pathlib import Path
import logging
from datetime import datetime
import argparse

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

# ====== 统一：构造 chat 模板 ======
def to_prompt_user_block(tokenizer, text: str, enable_thinking:bool) -> str:
    conv = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    return tokenizer.apply_chat_template(
        conv, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )

def main():
    # ---------------- 解析命令行参数 ----------------
    parser = argparse.ArgumentParser(
        description='Run TDC benchmark tasks with Intern-S1 model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
可用的任务组 (TASK_GROUP_NAMEs):
  ADME                  - 药物吸收、分布、代谢和排泄任务
  Tox                   - 毒性预测任务
  HTS                   - 高通量筛选任务
  Develop               - 药物开发相关任务
  PPI                   - 蛋白质-蛋白质相互作用任务
  TCREpitopeBinding     - T细胞受体-表位结合任务
  TrialOutcome          - 临床试验结果预测任务
  PeptideMHC            - 肽-MHC结合任务

示例:
  python %(prog)s --task-groups Tox Develop
  python %(prog)s --task-groups ADME HTS PPI
  python %(prog)s --task-groups all  # 运行所有任务组
        '''
    )

    parser.add_argument(
        '--task-groups',
        nargs='+',
        choices=['ADME', 'Tox', 'HTS', 'Develop', 'PPI', 'TCREpitopeBinding', 'TrialOutcome', 'PeptideMHC', 'all'],
        default=['Tox', 'Develop'],
        help='选择要运行的任务组 (可以选择多个)'
    )

    parser.add_argument(
        '--n-samples',
        type=int,
        default=16,
        help='每题采样次数 (默认: 16)'
    )

    args = parser.parse_args()

    # 如果选择了 'all'，则运行所有任务组
    if 'all' in args.task_groups:
        TASK_GROUP_NAMEs = ['ADME', 'Tox', 'HTS', 'Develop', 'PPI', 'TCREpitopeBinding', 'TrialOutcome', 'PeptideMHC']
    else:
        TASK_GROUP_NAMEs = args.task_groups

    # ---------------- 配置 Logging ----------------
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'experiment_log_{timestamp}.log'

    # 配置 logging：同时输出到控制台和文件
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            # logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    logger.info(f"Selected task groups: {TASK_GROUP_NAMEs}")

    # ---------------- 配置区 ----------------
    MODEL_NAME = "internlm/Intern-S1-mini"         # 也可用 "internlm/Intern-S1-mini-FP8"
    LORA_PATH = "checkpoints/Intern-S1-mini/lora/sft/checkpoint-180000"  # 可为空字符串禁用 LoRA
    USE_LORA = False

    # 采样与打分
    N_SAMPLES = args.n_samples  # 每题采样次数，从命令行参数获取
    logger.info(f"Sampling times: {N_SAMPLES}")
    TEMPERATURE = 0.7
    TOP_P = 1.0
    TOP_K = 50
    # 可选：是否在题干中插入“先思考再回答，最后把最终选项写在 Answer: 后”
    INJECT_STEPS_BEFORE_ANSWER = False  # 要不要开 thinking
    ENABLE_THINKING = INJECT_STEPS_BEFORE_ANSWER
    MAX_TOKENS = 1024 * 4 if INJECT_STEPS_BEFORE_ANSWER else 8 # 保证能"想完话"

    # ---------------- 模型 & tokenizer ----------------
    # processor 仅在部分模型需要（比如多模态），此处保留可选
    # try:
    #     _ = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    # except Exception:
    #     pass

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    llm = LLM(
        model=MODEL_NAME,
        enforce_eager=True,
        max_model_len=1024 * 24,
        max_num_batched_tokens=1024 * 24,
        quantization="fp8" if 'FP8' in MODEL_NAME else None,  # "fp8",          # 触发 FP8 W8A8 路径（或自动识别 FP8 检查点）
        dtype="bfloat16",
        tensor_parallel_size=1 if 'Intern-S1-mini' in MODEL_NAME else 8,
        trust_remote_code=True,
        gpu_memory_utilization=0.92,
        max_num_seqs=256,
        limit_mm_per_prompt={"video": 0, "image": 0},
        enable_lora=USE_LORA,
        tokenizer_mode="auto"
    )

    lora_req = None
    if USE_LORA and LORA_PATH and os.path.isdir(LORA_PATH):
        lora_req = LoRARequest(lora_name='lora', lora_int_id=1, lora_path=LORA_PATH)

    sp = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        max_tokens=MAX_TOKENS,
        n=N_SAMPLES,
        detokenize=True,
    )

    # ---------------- 读取 TDC prompt 模板 ----------------
    tdc_prompts_filepath = hf_hub_download(
        repo_id="google/txgemma-9b-predict",
        filename="tdc_prompts.json",
    )
    tdc_prompts_json = json.load(open(tdc_prompts_filepath, "r"))
    # 修正 key
    tdc_prompts_json['SARSCoV2_3CLPro_Diamond'] = tdc_prompts_json['SARSCOV2_3CLPro_Diamond']

    # ---------------- 存放各类任务 AUROC ----------------
    # Tox
    ToxCast_AUROCS = []; herg_central_AUROCS = []; Tox21_AUROCS = []
    Skin_Reaction_AUROCS = []; hERG_AUROCS = []; AMES_AUROCS = []
    DILI_AUROCS = []; ClinTox_AUROCS = []
    # ADME
    PAMPA_NCATS_AUROCS=[]; HIA_Hou_AUROCS=[]; Bioavailability_Ma_AUROCS=[]
    BBB_Martins_AUROCS=[]; Pgp_Broccatelli_AUROCS=[]
    CYP1A2_Veith_AUROCS=[]; CYP2C19_Veith_AUROCS=[]; CYP2C9_Veith_AUROCS=[]
    CYP2D6_Veith_AUROCS=[]; CYP3A4_Veith_AUROCS=[]
    CYP2C9_Substrate_CarbonMangels_AUROCS=[]; CYP2D6_Substrate_CarbonMangels_AUROCS=[]
    CYP3A4_Substrate_CarbonMangels_AUROCS=[]
    # HTS
    HIV_AUROCS=[]; SARSCoV2_3CLPro_Diamond_AUROCS=[]; SARSCoV2_Vitro_Touret_AUROCS=[]
    butkiewicz_AUROCS=[]
    # Develop
    SAbDab_Chen_AUROCS=[]
    # PPI
    HuRI_AUROCS=[]
    # TCR
    Weber_AUROCS=[]
    # TrialOutcome
    phase1_AUROCS=[]; phase2_AUROCS=[]; phase3_AUROCS=[]
    # PeptideMHC
    MHC1_IEDB_IMGT_Nielsen_AUROCS=[]; MHC2_IEDB_Jensen_AUROCS=[]

    # ---------------- 任务主循环 ----------------
    for TASK_GROUP_NAME in TASK_GROUP_NAMEs:
        if TASK_GROUP_NAME == 'Tox':
            TASK_NAMEs = (
                ['Skin_Reaction', 'hERG', 'AMES', 'DILI','ClinTox'] +
                ['Tox21' + '_' + label.replace('-', '_') for label in retrieve_label_name_list('Tox21')] +
                ['herg_central' + '_' + retrieve_label_name_list('herg_central')[-1]]
                + ['ToxCast' + '_' + label for label in retrieve_label_name_list('Toxcast')]
            )
        elif TASK_GROUP_NAME == 'ADME':
            TASK_NAMEs = [
                'PAMPA_NCATS', 'HIA_Hou', 'Bioavailability_Ma', 'BBB_Martins', 'Pgp_Broccatelli',
                'CYP1A2_Veith', 'CYP2C19_Veith', 'CYP2C9_Veith', 'CYP2D6_Veith', 'CYP3A4_Veith',
                'CYP2C9_Substrate_CarbonMangels', 'CYP2D6_Substrate_CarbonMangels', 'CYP3A4_Substrate_CarbonMangels',
            ]
        elif TASK_GROUP_NAME == 'HTS':
            TASK_NAMEs = [
                'cav3_t-type_calcium_channels_butkiewicz',
                'tyrosyl-dna_phosphodiesterase_butkiewicz',
                'HIV',
                'SARSCoV2_3CLPro_Diamond',
                'SARSCoV2_Vitro_Touret',
                'orexin1_receptor_butkiewicz',
                'm1_muscarinic_receptor_agonists_butkiewicz',
                'm1_muscarinic_receptor_antagonists_butkiewicz',
                'potassium_ion_channel_kir2.1_butkiewicz',
                'kcnq2_potassium_channel_butkiewicz',
                'choline_transporter_butkiewicz',
                'serine_threonine_kinase_33_butkiewicz',
            ]
        elif TASK_GROUP_NAME == 'Develop':
            TASK_NAMEs = ['SAbDab_Chen']
        elif TASK_GROUP_NAME == 'PPI':
            TASK_NAMEs = ['HuRI']
        elif TASK_GROUP_NAME == 'TCREpitopeBinding':
            TASK_NAMEs = ['Weber']
        elif TASK_GROUP_NAME == 'TrialOutcome':
            TASK_NAMEs = ['phase1', 'phase2', 'phase3']
        elif TASK_GROUP_NAME == 'PeptideMHC':
            TASK_NAMEs = ['MHC1_IEDB-IMGT_Nielsen', 'MHC2_IEDB_Jensen']
        else:
            continue

        for TASK_NAME in tqdm(TASK_NAMEs, desc=f"[{TASK_GROUP_NAME}] Processing tasks ..."):
            # --------- 加载数据 ---------
            if TASK_GROUP_NAME == 'Tox':
                data = None
                for special_task in ['ToxCast', 'herg_central', 'Tox21']:
                    if TASK_NAME.startswith(special_task):
                        data = Tox(
                            name=special_task,
                            label_name=TASK_NAME.split(special_task + '_')[-1] if special_task != 'Tox21'
                            else TASK_NAME.split(special_task + '_')[-1].replace('_', '-')
                        )
                        break
                if data is None:
                    data = Tox(name=TASK_NAME)
            elif TASK_GROUP_NAME == 'ADME':
                data = ADME(name=TASK_NAME)
            elif TASK_GROUP_NAME == 'HTS':
                data = HTS(name=TASK_NAME)
            elif TASK_GROUP_NAME == 'Develop':
                data = Develop(name=TASK_NAME)
            elif TASK_GROUP_NAME == 'PPI':
                data = PPI(name=TASK_NAME).neg_sample(frac=1)
            elif TASK_GROUP_NAME == 'TCREpitopeBinding':
                data = TCREpitopeBinding(name=TASK_NAME)
            elif TASK_GROUP_NAME == 'TrialOutcome':
                data = TrialOutcome(name=TASK_NAME)
            elif TASK_GROUP_NAME == 'PeptideMHC':
                data = PeptideMHC(name=TASK_NAME)
            else:
                continue

            split = data.get_split()

            # --------- 构造输入与标签 ---------
            if TASK_NAME == 'SAbDab_Chen':
                test_inputs = split['test']['Antibody'].values  # 里边是 str(list) 需解析
            elif TASK_NAME == 'HuRI':
                test_inputs = split['test'][['Protein1', 'Protein2']].values
            elif TASK_NAME == 'Weber':
                test_inputs = split['test'][['epitope_aa', 'tcr']].values
            elif TASK_NAME in ['MHC1_IEDB-IMGT_Nielsen', 'MHC2_IEDB_Jensen']:
                test_inputs = split['test'][['Peptide', 'MHC']].values
            else:
                test_inputs = split['test']['Drug'].values

            if TASK_NAME == 'Weber':
                test_labels = split['test']['label'].values
            else:
                test_labels = split['test']['Y'].values

            logger.info(f"[{TASK_GROUP_NAME}/{TASK_NAME}] Total test samples: {len(test_inputs)}")

            # --------- 准备 prompts ---------
            INPUT_TYPE = "{Drug SMILES}"
            base_prompts = []
            for entry in test_inputs:
                if TASK_NAME.startswith('herg_central'):
                    tn = 'herg_central'
                else:
                    tn = TASK_NAME

                if tn == 'SAbDab_Chen':
                    pair = ast.literal_eval(entry)
                    user_text = tdc_prompts_json[tn.replace('-', '_')].replace(
                        '{Antibody heavy chain sequence}', f'<FASTA>{pair[0]}</FASTA>'
                    ).replace(
                        '{Antibody light chain sequence}', f'<FASTA>{pair[1]}</FASTA>'
                    )
                elif tn == 'HuRI':
                    user_text = tdc_prompts_json[tn.replace('-', '_')].replace(
                        '{Protein1 amino acid sequence}', f'<FASTA>{entry[0]}</FASTA>'
                    ).replace(
                        '{Protein2 amino acid sequence}', f'<FASTA>{entry[1]}</FASTA>'
                    )
                elif tn == 'Weber':
                    user_text = tdc_prompts_json['weber'].replace(
                        '{Epitope amino acid sequence}', f'<FASTA>{entry[0]}</FASTA>'
                    ).replace(
                        '{TCR amino acid sequence}', f'<FASTA>{entry[1]}</FASTA>'
                    )
                elif tn in ['MHC1_IEDB-IMGT_Nielsen', 'MHC2_IEDB_Jensen']:
                    user_text = tdc_prompts_json[tn.replace('-', '_')].replace(
                        '{Peptide amino acid sequence}', f'<FASTA>{entry[0]}</FASTA>'
                    ).replace(
                        '{Possible MHC pseudosequences}', f'<FASTA>{entry[1]}</FASTA>'
                    )
                else:
                    user_text = tdc_prompts_json[tn.replace('-', '_')].replace(
                        INPUT_TYPE, f'<SMILES>{entry}</SMILES>'
                    )

                if INJECT_STEPS_BEFORE_ANSWER:  # TODO: 新增的
                    user_text = user_text.replace(
                        'Answer:',
                        'Please think step by step and then put ONLY your final choice ((A) or (B)) after "Answer:"'
                    )

                base_prompts.append(to_prompt_user_block(tokenizer, user_text, ENABLE_THINKING))
                # break # TODO: for debug

            # --------- 生成 ---------
            if USE_LORA:
                outputs = llm.generate(base_prompts, sp, lora_request=lora_req)
            else:
                outputs = llm.generate(base_prompts, sp)

            # --------- 汇总每题概率 \hat p ---------
            p_scores = []
            failed_items = 0
            failed_parses = []  # 存储解析失败的样本
            total_cnt_a = 0; total_cnt_b = 0; total_cnt_none = 0

            for i, out in enumerate(tqdm(outputs, desc=f"[{TASK_GROUP_NAME}/{TASK_NAME}] Parsing outputs ...")):
                cnt_a = cnt_b = cnt_none = 0
                for j, cand in enumerate(out.outputs):
                    txt = (cand.text or "").strip()

                    # print(txt) # TODO: debug

                    ans_txt, fmt_ok = extract_answer(txt)
                    pred = parse_answer(ans_txt, fmt_ok, ENABLE_THINKING)
                    if pred == 0:
                        cnt_a += 1
                    elif pred == 1:
                        cnt_b += 1
                    else:
                        cnt_none += 1

                    # 记录解析失败的样本
                    if pred is None:
                        failed_parses.append({
                            'index': i,
                            'sample_index': j,
                            'test_input': str(test_inputs[i]),  # 截取前100个字符
                            'full_output': txt,
                            'extracted_answer': ans_txt,
                            'true_label': test_labels[i]
                        })

                    # 仅前2题展示前3个采样，便于 sanity check
                    if i < 2 and j < 3:
                        print(f'\n[Item {i} | Sample {j}]')
                        print(f'Raw: {txt}')
                        print(f'Parsed -> {pred}')

                denom = cnt_a + cnt_b
                if denom == 0:
                    p_hat = None
                    failed_items += 1
                else:
                    p_hat = cnt_b / denom

                p_scores.append(p_hat)
                total_cnt_a += cnt_a; total_cnt_b += cnt_b; total_cnt_none += cnt_none

            valid_idx = [i for i, p in enumerate(p_scores) if p is not None]
            y_true = [int(test_labels[i]) for i in valid_idx]
            y_score = [float(p_scores[i]) for i in valid_idx]

            print(f'\n{"="*80}')
            print(f'[{TASK_GROUP_NAME}/{TASK_NAME}] EVALUATION RESULTS (MC sampling, None dropped)')
            print(f'{"="*80}')
            print(f'Items with defined p-hat: {len(valid_idx)}/{len(p_scores)} (dropped={failed_items})')
            print(f'Parse counts across all samples: A={total_cnt_a}, B={total_cnt_b}, None={total_cnt_none}')

            # --------- 打印解析失败的样本 ---------
            if len(failed_parses) > 0:
                print('\n' + '=' * 80)
                print(f'FAILED PARSES (Total: {len(failed_parses)})')
                print('=' * 80)
                # 只显示前10个失败样本，避免输出过多
                for fail in failed_parses[:10]:
                    print(f'\n[Failed Sample - Item {fail["index"]} | Sample {fail["sample_index"]}]')
                    print(f'Test Input: {fail["test_input"][:100]}...' if len(fail["test_input"]) > 100 else f'Test Input: {fail["test_input"]}')
                    print(f'True Label: {fail["true_label"]}')
                    print(f'Extracted Answer: {fail["extracted_answer"]}')
                    print(f'Full Output:\n{fail["full_output"][:500]}...' if len(fail["full_output"]) > 500 else f'Full Output:\n{fail["full_output"]}')
                    print('-' * 80)
                if len(failed_parses) > 10:
                    print(f'\n... and {len(failed_parses) - 10} more failed parses (not shown)')
                print('=' * 80)
            else:
                print('\nNo failed parses!')

            # --------- 计算 AUROC 并保存到对应列表 ---------
            if len(valid_idx) > 0 and len(set(y_true)) > 1:
                auroc = roc_auc_score(y_true, y_score)
                print(f'AUROC: {auroc:.4f}')
                print(f'{"="*80}\n')
                if TASK_GROUP_NAME == 'Tox':
                    if TASK_NAME.startswith('herg_central'):
                        herg_central_AUROCS.append(auroc)
                    elif TASK_NAME.startswith('ToxCast'):
                        ToxCast_AUROCS.append(auroc)
                    elif TASK_NAME.startswith('Tox21'):
                        Tox21_AUROCS.append(auroc)
                    elif TASK_NAME.startswith('Skin_Reaction'):
                        Skin_Reaction_AUROCS.append(auroc)
                    elif TASK_NAME.startswith('hERG'):
                        hERG_AUROCS.append(auroc)
                    elif TASK_NAME.startswith('AMES'):
                        AMES_AUROCS.append(auroc)
                    elif TASK_NAME.startswith('DILI'):
                        DILI_AUROCS.append(auroc)
                    elif TASK_NAME.startswith('ClinTox'):
                        ClinTox_AUROCS.append(auroc)
                elif TASK_GROUP_NAME == 'ADME':
                    if TASK_NAME.startswith('PAMPA_NCATS'):
                        PAMPA_NCATS_AUROCS.append(auroc)
                    elif TASK_NAME.startswith('HIA_Hou'):
                        HIA_Hou_AUROCS.append(auroc)
                    elif TASK_NAME.startswith('Bioavailability_Ma'):
                        Bioavailability_Ma_AUROCS.append(auroc)
                    elif TASK_NAME.startswith('BBB_Martins'):
                        BBB_Martins_AUROCS.append(auroc)
                    elif TASK_NAME.startswith('Pgp_Broccatelli'):
                        Pgp_Broccatelli_AUROCS.append(auroc)
                    elif TASK_NAME.startswith('CYP1A2_Veith'):
                        CYP1A2_Veith_AUROCS.append(auroc)
                    elif TASK_NAME.startswith('CYP2C19_Veith'):
                        CYP2C19_Veith_AUROCS.append(auroc)
                    elif TASK_NAME.startswith('CYP2C9_Veith'):
                        CYP2C9_Veith_AUROCS.append(auroc)
                    elif TASK_NAME.startswith('CYP2D6_Veith'):
                        CYP2D6_Veith_AUROCS.append(auroc)
                    elif TASK_NAME.startswith('CYP3A4_Veith'):
                        CYP3A4_Veith_AUROCS.append(auroc)
                    elif TASK_NAME.startswith('CYP2C9_Substrate_CarbonMangels'):
                        CYP2C9_Substrate_CarbonMangels_AUROCS.append(auroc)
                    elif TASK_NAME.startswith('CYP2D6_Substrate_CarbonMangels'):
                        CYP2D6_Substrate_CarbonMangels_AUROCS.append(auroc)
                    elif TASK_NAME.startswith('CYP3A4_Substrate_CarbonMangels'):
                        CYP3A4_Substrate_CarbonMangels_AUROCS.append(auroc)
                elif TASK_GROUP_NAME == 'HTS':
                    if TASK_NAME.startswith('HIV'):
                        HIV_AUROCS.append(auroc)
                    elif TASK_NAME.startswith('SARSCoV2_3CLPro_Diamond'):
                        SARSCoV2_3CLPro_Diamond_AUROCS.append(auroc)
                    elif TASK_NAME.startswith('SARSCoV2_Vitro_Touret'):
                        SARSCoV2_Vitro_Touret_AUROCS.append(auroc)
                    elif TASK_NAME.endswith('butkiewicz'):
                        butkiewicz_AUROCS.append(auroc)
                elif TASK_GROUP_NAME == 'Develop':
                    if TASK_NAME.startswith('SAbDab_Chen'):
                        SAbDab_Chen_AUROCS.append(auroc)
                elif TASK_GROUP_NAME == 'PPI':
                    if TASK_NAME.startswith('HuRI'):
                        HuRI_AUROCS.append(auroc)
                elif TASK_GROUP_NAME == 'TCREpitopeBinding':
                    if TASK_NAME.startswith('Weber'):
                        Weber_AUROCS.append(auroc)
                elif TASK_GROUP_NAME == 'TrialOutcome':
                    if TASK_NAME == 'phase1':
                        phase1_AUROCS.append(auroc)
                    elif TASK_NAME == 'phase2':
                        phase2_AUROCS.append(auroc)
                    elif TASK_NAME == 'phase3':
                        phase3_AUROCS.append(auroc)
                elif TASK_GROUP_NAME == 'PeptideMHC':
                    if TASK_NAME.startswith('MHC1_IEDB-IMGT_Nielsen'):
                        MHC1_IEDB_IMGT_Nielsen_AUROCS.append(auroc)
                    if TASK_NAME.startswith('MHC2_IEDB_Jensen'):
                        MHC2_IEDB_Jensen_AUROCS.append(auroc)
            else:
                print(f'Cannot compute AUROC (no valid items or only one class).')
                print(f'{"="*80}\n')

    # ---------------- 汇总打印 ----------------
    print("="*80)
    print(MODEL_NAME)
    for TASK_GROUP_NAME in TASK_GROUP_NAMEs:
        if TASK_GROUP_NAME == 'Tox':
            print('herg_central', np.mean(np.array(herg_central_AUROCS)) if herg_central_AUROCS else None)
            print('ToxCast', np.mean(np.array(ToxCast_AUROCS)) if ToxCast_AUROCS else None)
            print('Tox21', np.mean(np.array(Tox21_AUROCS)) if Tox21_AUROCS else None)
            print('Skin_Reaction', np.mean(np.array(Skin_Reaction_AUROCS)) if Skin_Reaction_AUROCS else None)
            print('hERG', np.mean(np.array(hERG_AUROCS)) if hERG_AUROCS else None)
            print('AMES', np.mean(np.array(AMES_AUROCS)) if AMES_AUROCS else None)
            print('DILI', np.mean(np.array(DILI_AUROCS)) if DILI_AUROCS else None)
            print('ClinTox', np.mean(np.array(ClinTox_AUROCS)) if ClinTox_AUROCS else None)
        elif TASK_GROUP_NAME == 'ADME':
            print('PAMPA_NCATS', np.mean(np.array(PAMPA_NCATS_AUROCS)) if PAMPA_NCATS_AUROCS else None)
            print('HIA_Hou', np.mean(np.array(HIA_Hou_AUROCS)) if HIA_Hou_AUROCS else None)
            print('Bioavailability_Ma', np.mean(np.array(Bioavailability_Ma_AUROCS)) if Bioavailability_Ma_AUROCS else None)
            print('BBB_Martins', np.mean(np.array(BBB_Martins_AUROCS)) if BBB_Martins_AUROCS else None)
            print('Pgp_Broccatelli', np.mean(np.array(Pgp_Broccatelli_AUROCS)) if Pgp_Broccatelli_AUROCS else None)
            print('CYP1A2_Veith', np.mean(np.array(CYP1A2_Veith_AUROCS)) if CYP1A2_Veith_AUROCS else None)
            print('CYP2C19_Veith', np.mean(np.array(CYP2C19_Veith_AUROCS)) if CYP2C19_Veith_AUROCS else None)
            print('CYP2C9_Veith', np.mean(np.array(CYP2C9_Veith_AUROCS)) if CYP2C9_Veith_AUROCS else None)
            print('CYP2D6_Veith', np.mean(np.array(CYP2D6_Veith_AUROCS)) if CYP2D6_Veith_AUROCS else None)
            print('CYP3A4_Veith', np.mean(np.array(CYP3A4_Veith_AUROCS)) if CYP3A4_Veith_AUROCS else None)
            print('CYP2C9_Substrate_CarbonMangels', np.mean(np.array(CYP2C9_Substrate_CarbonMangels_AUROCS)) if CYP2C9_Substrate_CarbonMangels_AUROCS else None)
            print('CYP2D6_Substrate_CarbonMangels', np.mean(np.array(CYP2D6_Substrate_CarbonMangels_AUROCS)) if CYP2D6_Substrate_CarbonMangels_AUROCS else None)
            print('CYP3A4_Substrate_CarbonMangels', np.mean(np.array(CYP3A4_Substrate_CarbonMangels_AUROCS)) if CYP3A4_Substrate_CarbonMangels_AUROCS else None)
        elif TASK_GROUP_NAME == 'HTS':
            print('HIV', np.mean(np.array(HIV_AUROCS)) if HIV_AUROCS else None)
            print('SARSCoV2_3CLPro_Diamond', np.mean(np.array(SARSCoV2_3CLPro_Diamond_AUROCS)) if SARSCoV2_3CLPro_Diamond_AUROCS else None)
            print('SARSCoV2_Vitro_Touret', np.mean(np.array(SARSCoV2_Vitro_Touret_AUROCS)) if SARSCoV2_Vitro_Touret_AUROCS else None)
            print('butkiewicz', np.mean(np.array(butkiewicz_AUROCS)) if butkiewicz_AUROCS else None)
        elif TASK_GROUP_NAME == 'Develop':
            print('SAbDab_Chen', np.mean(np.array(SAbDab_Chen_AUROCS)) if SAbDab_Chen_AUROCS else None)
        elif TASK_GROUP_NAME == 'PPI':
            print('HuRI', np.mean(np.array(HuRI_AUROCS)) if HuRI_AUROCS else None)
        elif TASK_GROUP_NAME == 'TCREpitopeBinding':
            print('Weber', np.mean(np.array(Weber_AUROCS)) if Weber_AUROCS else None)
        elif TASK_GROUP_NAME == 'TrialOutcome':
            print('phase1', np.mean(np.array(phase1_AUROCS)) if phase1_AUROCS else None)
            print('phase2', np.mean(np.array(phase2_AUROCS)) if phase2_AUROCS else None)
            print('phase3', np.mean(np.array(phase3_AUROCS)) if phase3_AUROCS else None)
        elif TASK_GROUP_NAME == 'PeptideMHC':
            print('MHC1_IEDB-IMGT_Nielsen', np.mean(np.array(MHC1_IEDB_IMGT_Nielsen_AUROCS)) if MHC1_IEDB_IMGT_Nielsen_AUROCS else None)
            print('MHC2_IEDB_Jensen', np.mean(np.array(MHC2_IEDB_Jensen_AUROCS)) if MHC2_IEDB_Jensen_AUROCS else None)
    print("="*80)

if __name__ == "__main__":
    main()
