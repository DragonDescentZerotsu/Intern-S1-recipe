import os
import json
import logging
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from tdc.single_pred import ADME, Tox, HTS, Develop
from tdc.multi_pred.ppi import PPI
from tdc.multi_pred.tcr_epi import TCREpitopeBinding
from tdc.multi_pred.trialoutcome import TrialOutcome
from tdc.multi_pred.peptidemhc import PeptideMHC
from tdc.utils import retrieve_label_name_list
import ast
import argparse

# 放在文件顶部合适位置（例如 to_prompt 后面）
SPLIT_RANDOM = {
    'SAbDab_Chen',                    # Developability
    'MHC1_IEDB-IMGT_Nielsen',         # Peptide-MHC
    'MHC2_IEDB_Jensen',
    # 以 butkiewicz 结尾的一批 HTS 任务也用 random（代码里用后缀判断）
}

SPLIT_COLD = {
    # 冷启动：HuRI（PPI）、Weber（TCR-epitope）、临床试验结局
    'HuRI', 'Weber', 'phase1', 'phase2', 'phase3',
}

def choose_split_method(task_name, data):
    # butkiewicz 系列 HTS
    if task_name.endswith('butkiewicz'):
        return "random", {}

    if task_name in SPLIT_RANDOM:
        return "random", {}

    if task_name in SPLIT_COLD:
        # ✅ multi_pred 冷启动统一用 cold_split
        if task_name == "Weber":
            return "cold_split", {"column_name": ["tcr"]}  # or only "tcr"
        if task_name == "HuRI":
            return "cold_split", {"column_name": ["Protein1", 'Protein2']}  # 也可只放一个
        # phase1/2/3 取决于 loader 实际列名（Drug / Disease / etc.）
        return "cold_split", {"column_name": "Drug"}

    # 默认 Scaffold
    return "scaffold", {}



def to_prompt(prompt_template, entry, task_name):
    """
    Generate prompt based on task type.
    Crucially, do NOT add <SMILES>/<FASTA> tags.
    """
    user_text = ""
    
    # Task-specific prompt construction
    if task_name == 'SAbDab_Chen':
        try:
            pair = ast.literal_eval(entry) if isinstance(entry, str) else entry
        except:
            pair = entry 
            
        user_text = prompt_template.replace(
            '{Antibody heavy chain sequence}', f'{pair[0]}'
        ).replace(
            '{Antibody light chain sequence}', f'{pair[1]}'
        )
    elif task_name == 'HuRI':
        user_text = prompt_template.replace(
            '{Protein1 amino acid sequence}', f'{entry[0].split("*")[0]}'
        ).replace(
            '{Protein2 amino acid sequence}', f'{entry[1].split("*")[0]}'
        )
    elif task_name == 'Weber':
        user_text = prompt_template.replace(
            '{Epitope amino acid sequence}', f'{entry[0]}'
        ).replace(
            '{TCR amino acid sequence}', f'{entry[1]}'
        )
    elif task_name in ['MHC1_IEDB-IMGT_Nielsen', 'MHC2_IEDB_Jensen']:
        user_text = prompt_template.replace(
            '{Peptide amino acid sequence}', f'{entry[0]}'
        ).replace(
            '{Possible MHC pseudosequences}', f'{entry[1]}'
        )
    else:
        # Standard SMILES task
        user_text = prompt_template.replace(
            "{Drug SMILES}", f'{entry}'
        )

    # Append instruction
    user_text = user_text.replace(
                                'Answer:',
                                'Please think step by step and then put ONLY your final choice ((A) or (B)) after "Answer:"'
                            )
    
    return user_text

def main():
    parser = argparse.ArgumentParser(description="Process TDC data for a specific split.")
    parser.add_argument("--target-split", type=str, default="train", choices=["train", "test", "valid"], help="Target split to process (train, test, valid)")
    args = parser.parse_args()
    target_split = args.target_split
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    output_dir = Path(f"DataPrepare/TDC_{target_split}_prompts_label_scaffold")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load TDC prompts
    logger.info("Downloading/Loading TDC prompts...")
    tdc_prompts_filepath = hf_hub_download(
        repo_id="google/txgemma-9b-predict",
        filename="tdc_prompts.json",
    )
    with open(tdc_prompts_filepath, "r") as f:
        tdc_prompts_json = json.load(f)
    
    # Fix known key issue
    if 'SARSCOV2_3CLPro_Diamond' in tdc_prompts_json:
        tdc_prompts_json['SARSCoV2_3CLPro_Diamond'] = tdc_prompts_json['SARSCOV2_3CLPro_Diamond']

    task_groups = ['ADME', 'Tox', 'HTS', 'Develop', 'PPI', 'TCREpitopeBinding', 'TrialOutcome', 'PeptideMHC']
    
    metadata = {}
    
    # buffers for aggregation
    aggregated_data = {
        'Tox21': [],
        'ToxCast': [],
        'butkiewicz': []
    }
    aggregated_split_methods = {}

    for group_name in task_groups:
        task_names = []
        if group_name == 'Tox':
            task_names = ['Skin_Reaction'] 
            task_names += ['hERG', 'AMES', 'DILI', 'ClinTox']
            task_names += ['Tox21_' + label for label in retrieve_label_name_list('Tox21')]
            task_names += ['herg_central_' + retrieve_label_name_list('herg_central')[-1]] 
            task_names += ['ToxCast_' + label for label in retrieve_label_name_list('Toxcast')]
            
        elif group_name == 'ADME':
            task_names = [
                'PAMPA_NCATS', 
                'HIA_Hou', 
                'Bioavailability_Ma', 
                'BBB_Martins', 
                'Pgp_Broccatelli',
                'CYP1A2_Veith', 
                'CYP2C19_Veith', 
                'CYP2C9_Veith', 
                'CYP2D6_Veith', 
                'CYP3A4_Veith',
                'CYP2C9_Substrate_CarbonMangels', 
                'CYP2D6_Substrate_CarbonMangels', 
                'CYP3A4_Substrate_CarbonMangels'
            ]
        elif group_name == 'HTS':
            task_names = [
                'HIV', 'SARSCoV2_3CLPro_Diamond', 'SARSCoV2_Vitro_Touret', 
                'cav3_t-type_calcium_channels_butkiewicz', 'tyrosyl-dna_phosphodiesterase_butkiewicz',
                'm1_muscarinic_receptor_agonists_butkiewicz', 'm1_muscarinic_receptor_antagonists_butkiewicz',
                'potassium_ion_channel_kir2.1_butkiewicz', 'kcnq2_potassium_channel_butkiewicz',
                'orexin1_receptor_butkiewicz', 'choline_transporter_butkiewicz', 'serine_threonine_kinase_33_butkiewicz'
            ]
        elif group_name == 'Develop':
            task_names = ['SAbDab_Chen']
        elif group_name == 'PPI':
            task_names = ['HuRI']
        elif group_name == 'TCREpitopeBinding':
            task_names = ['Weber']
        elif group_name == 'TrialOutcome':
            task_names = ['phase1', 'phase2', 'phase3']
        elif group_name == 'PeptideMHC':
            task_names = ['MHC1_IEDB-IMGT_Nielsen', 'MHC2_IEDB_Jensen']
        
        else:
             continue 

        for task_name in tqdm(task_names, desc=f"Processing {group_name}"):
            try:
                # 1. Load Data
                data = None
                if group_name == 'Tox':
                    if task_name.startswith('ToxCast_'):
                         label = task_name.split('ToxCast_')[-1]
                         data = Tox(name='ToxCast', label_name=label)
                    elif task_name.startswith('herg_central_'):
                         label = task_name.split('herg_central_')[-1]
                         data = Tox(name='herg_central', label_name=label)
                    elif task_name.startswith('Tox21_'):
                         label = task_name.split('Tox21_')[-1].replace('_', '-') 
                         data = Tox(name='Tox21', label_name=label)
                    else:
                         data = Tox(name=task_name)
                elif group_name == 'ADME':
                    data = ADME(name=task_name)
                elif group_name == 'HTS':
                    data = HTS(name=task_name)
                elif group_name == 'Develop':
                    data = Develop(name=task_name)
                elif group_name == 'PPI':
                    data = PPI(name=task_name).neg_sample(frac=1)
                elif group_name == 'TCREpitopeBinding':
                    data = TCREpitopeBinding(name=task_name)
                elif group_name == 'TrialOutcome':
                    data = TrialOutcome(name=task_name)
                elif group_name == 'PeptideMHC':
                    data = PeptideMHC(name=task_name)
                
                if data is None:
                    continue
                
                # 原始固定的 split 的方法
                # split = data.get_split(method='scaffold')

                # 新的 split 的方法
                method, kwargs = choose_split_method(task_name, data)
                split = None
                _last_err = None
                used_split_method = None

                # 先按规则尝试；不行则回退到 scaffold -> random，保证能跑通
                for (m, kw) in [(method, kwargs), ("scaffold", {}), ("random", {})]:
                    try:
                        split = data.get_split(method=m, **kw)
                        used_split_method = f"{m}:{kw}" if kw else m
                        break
                    except Exception as e:
                        _last_err = e

                if split is None:
                    logger.warning(f"get_split failed for {task_name} (tried: {method}, scaffold, random). Last error: {_last_err}. Skipping.")
                    continue

                
                if target_split not in split:
                    logger.warning(f"No {target_split} split for {task_name}, skipping.")
                    continue
                
                split_df = split[target_split]
                    
                # 2. Extract inputs and labels
                inputs = []
                labels = []
                
                if task_name == 'SAbDab_Chen':
                     inputs = split_df['Antibody'].values
                     labels = split_df['Y'].values
                elif task_name == 'HuRI':
                     inputs = split_df[['Protein1', 'Protein2']].values
                     labels = split_df['Y'].values
                elif task_name == 'Weber':
                     inputs = split_df[['epitope_aa', 'tcr']].values
                     labels = split_df['label'].values
                elif task_name in ['MHC1_IEDB-IMGT_Nielsen', 'MHC2_IEDB_Jensen']:
                     inputs = split_df[['Peptide', 'MHC']].values
                     labels = split_df['Y'].values
                else:
                     inputs = split_df['Drug'].values
                     labels = split_df['Y'].values

                # 3. Generate Prompts
                processed_data = []
                
                prompt_key = task_name
                if task_name.startswith('herg_central'):
                    prompt_key = 'herg_central'
                
                clean_key = prompt_key.replace('-', '_')
                
                if clean_key not in tdc_prompts_json:
                    base_key = clean_key.split('_')[0]
                    lower_base_key = base_key.lower()
                    if base_key in tdc_prompts_json:
                         clean_key = base_key
                    elif lower_base_key in tdc_prompts_json:
                         clean_key = lower_base_key
                    else:
                         # Attempt to fall back for Tox21/ToxCast if needed, though they shoud use 'Tox21'/'ToxCast' keys ideally?
                         # TDC prompts json typically has keys like 'Tox21', 'ToxCast'.
                         # If task_name is 'Tox21_AhR', clean_key is 'Tox21_AhR'.
                         # If json has 'Tox21', we need to match it.
                         if 'Tox21' in task_name and 'Tox21' in tdc_prompts_json:
                             clean_key = 'Tox21'
                         elif 'ToxCast' in task_name and 'ToxCast' in tdc_prompts_json:
                             clean_key = 'ToxCast'
                         else:
                             # Try partial match or skip
                             logger.warning(f"Prompt key {clean_key} not found for {task_name}. Skipping.")
                             continue

                template = tdc_prompts_json[clean_key]

                for x, y in zip(inputs, labels):
                    text = to_prompt(template, x, task_name)
                    processed_data.append({
                        "text": text,
                        "Y": int(y) if not isinstance(y, (list, tuple)) else y
                    })
                
                # 4. Save or Aggregate
                agg_target = None
                if task_name.startswith('Tox21'):
                    agg_target = 'Tox21'
                elif task_name.startswith('ToxCast'):
                    agg_target = 'ToxCast'
                elif task_name.endswith('butkiewicz'):
                    agg_target = 'butkiewicz'
                
                if agg_target:
                    aggregated_data[agg_target].extend(processed_data)
                    aggregated_split_methods[agg_target] = used_split_method
                else:
                    out_filename = f"{task_name}.jsonl"
                    out_path = output_dir / out_filename
                    with open(out_path, 'w', encoding='utf-8') as f:
                        for entry in processed_data:
                            f.write(json.dumps(entry) + '\n')
                    metadata[task_name] = {
                        "num_samples": len(processed_data),
                        "split_method": used_split_method
                    }
                
            except Exception as e:
                logger.error(f"Error processing {task_name}: {e}")
    
    # Save Aggregated Data
    for key, data_list in aggregated_data.items():
        if data_list:
            out_filename = f"{key}.jsonl"
            out_path = output_dir / out_filename
            logger.info(f"Saving aggregated {key} to {out_path} ({len(data_list)} entries)")
            with open(out_path, 'w', encoding='utf-8') as f:
                for entry in data_list:
                    f.write(json.dumps(entry) + '\n')
            metadata[key] = {
                "num_samples": len(data_list),
                "split_method": aggregated_split_methods.get(key, 'unknown')
            }

    # Save metadata
    with open(output_dir / "metadata.json", "w", encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
        
if __name__ == "__main__":
    main()
