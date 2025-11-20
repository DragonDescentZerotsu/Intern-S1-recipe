import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from tdc.single_pred import ADME, Tox, HTS, Develop
from tdc.multi_pred.ppi import PPI
from tdc.multi_pred.tcr_epi import TCREpitopeBinding
from tdc.multi_pred.trialoutcome import TrialOutcome
from tdc.multi_pred.peptidemhc import PeptideMHC
from tdc.utils import retrieve_label_name_list


def get_task_list(group_name):
    """完全复刻你原始代码的任务列表生成逻辑"""
    if group_name == 'Tox':
        return (['Skin_Reaction', 'hERG', 'AMES', 'DILI', 'ClinTox'] +
                ['Tox21' + '_' + l.replace('-', '_') for l in retrieve_label_name_list('Tox21')] +
                ['herg_central' + '_' + retrieve_label_name_list('herg_central')[-1]] +
                ['ToxCast' + '_' + l for l in retrieve_label_name_list('Toxcast')])
    elif group_name == 'ADME':
        return ['CYP1A2_Veith', 'CYP2C19_Veith', 'CYP2C9_Veith', 'CYP2D6_Veith', 'CYP3A4_Veith',
                'PAMPA_NCATS', 'HIA_Hou', 'Bioavailability_Ma', 'BBB_Martins', 'Pgp_Broccatelli',
                'CYP2C9_Substrate_CarbonMangels', 'CYP2D6_Substrate_CarbonMangels', 'CYP3A4_Substrate_CarbonMangels']
    elif group_name == 'HTS':
        return ['HIV', 'SARSCoV2_3CLPro_Diamond', 'SARSCoV2_Vitro_Touret',
                'cav3_t-type_calcium_channels_butkiewicz', 'tyrosyl-dna_phosphodiesterase_butkiewicz',
                'orexin1_receptor_butkiewicz', 'm1_muscarinic_receptor_agonists_butkiewicz',
                'm1_muscarinic_receptor_antagonists_butkiewicz', 'potassium_ion_channel_kir2.1_butkiewicz',
                'kcnq2_potassium_channel_butkiewicz', 'choline_transporter_butkiewicz',
                'serine_threonine_kinase_33_butkiewicz']
    elif group_name == 'Develop':
        return ['SAbDab_Chen']
    elif group_name == 'PPI':
        return ['HuRI']
    elif group_name == 'TCREpitopeBinding':
        return ['Weber']
    elif group_name == 'TrialOutcome':
        return ['phase1', 'phase2', 'phase3']
    elif group_name == 'PeptideMHC':
        return ['MHC1_IEDB-IMGT_Nielsen', 'MHC2_IEDB_Jensen']
    return []


def load_data(group, name):
    """复刻原始代码的数据加载逻辑，特别是 Tox 和 PPI"""
    if group == 'Tox':
        # 处理 ToxCast, herg_central, Tox21 的特殊命名
        for special in ['ToxCast', 'herg_central', 'Tox21']:
            if name.startswith(special):
                label = name.split(special + '_')[-1]
                if special == 'Tox21': label = label.replace('_', '-')
                return Tox(name=special, label_name=label)
        return Tox(name=name)
    elif group == 'ADME':
        return ADME(name=name)
    elif group == 'HTS':
        return HTS(name=name)
    elif group == 'Develop':
        return Develop(name=name)
    elif group == 'PPI':
        return PPI(name=name).neg_sample(frac=1)  # 必须复刻 negative sampling
    elif group == 'TCREpitopeBinding':
        return TCREpitopeBinding(name=name)
    elif group == 'TrialOutcome':
        return TrialOutcome(name=name)
    elif group == 'PeptideMHC':
        return PeptideMHC(name=name)
    return None


def run_all_baselines():
    GROUPS = ['ADME', 'Tox', 'HTS', 'Develop', 'PPI', 'TCREpitopeBinding', 'TrialOutcome', 'PeptideMHC']

    print(f"{'Group':<15} | {'Task Name':<40} | {'Maj. Class':<10} | {'AUROC'}")
    print("-" * 85)

    results = {}  # 存储结果以便后续计算组平均值

    for group in GROUPS:
        tasks = get_task_list(group)
        group_scores = []

        for task in tqdm(tasks, desc=f"Processing {group}", leave=False):
            try:
                # 1. 加载数据
                data = load_data(group, task)
                split = data.get_split()

                # 2. 获取标签 (Weber用label, 其他用Y)
                label_col = 'label' if task == 'Weber' else 'Y'
                y_train = split['train'][label_col].values
                y_test = split['test'][label_col].values

                # 3. 多数类基线逻辑
                # 统计训练集多数类
                majority_label = 1 if np.mean(y_train) > 0.5 else 0

                # 预测：全0或全1
                y_pred = np.full(y_test.shape, majority_label)

                # 4. 计算分数
                try:
                    if len(set(y_test)) < 2:
                        score = None  # 测试集只有一种类别，无法计算AUC
                    else:
                        # Sklearn对于全0或全1的预测，通常会报错或定义为0.5
                        # 这里手动处理防止报错
                        score = 0.5
                except Exception:
                    score = 0.5

                if score is not None:
                    group_scores.append(score)
                    print(f"{group:<15} | {task[:40]:<40} | {majority_label:<10} | {score:.4f}")

            except Exception as e:
                print(f"{group:<15} | {task[:40]:<40} | {'ERROR':<10} | {str(e)}")

        if group_scores:
            results[group] = np.mean(group_scores)

    print("\n" + "=" * 30)
    print("FINAL GROUP AVERAGE AUROC")
    print("=" * 30)
    for g, s in results.items():
        print(f"{g:<20}: {s:.4f}")


if __name__ == "__main__":
    run_all_baselines()