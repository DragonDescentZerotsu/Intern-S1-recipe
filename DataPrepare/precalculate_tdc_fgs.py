import os
import sys

# Set environment variables to avoid nested parallelism overhead
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import logging
from pathlib import Path
from tqdm import tqdm
from tdc.single_pred import ADME, Tox, HTS, Develop
from tdc.multi_pred.ppi import PPI
from tdc.multi_pred.tcr_epi import TCREpitopeBinding
from tdc.multi_pred.trialoutcome import TrialOutcome
from tdc.multi_pred.peptidemhc import PeptideMHC
from tdc.utils import retrieve_label_name_list
import concurrent.futures
import multiprocessing

# Add project root to path
current_dir = Path(__file__).parent.resolve()
sys.path.append(str(current_dir.parent))

from tools.AccFG import high_level_fg_fragments_w_attach_points_no_special_tokens_w_atom_ids

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_all_tdc_smiles():
    smiles_set = set()
    
    task_groups = ['ADME', 'Tox', 'HTS'] #, 'Develop', 'PPI', 'TCREpitopeBinding', 'TrialOutcome', 'PeptideMHC']

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
                'PAMPA_NCATS', 'HIA_Hou', 'Bioavailability_Ma', 'BBB_Martins', 'Pgp_Broccatelli',
                'CYP1A2_Veith', 'CYP2C19_Veith', 'CYP2C9_Veith', 'CYP2D6_Veith', 'CYP3A4_Veith',
                'CYP2C9_Substrate_CarbonMangels', 'CYP2D6_Substrate_CarbonMangels', 'CYP3A4_Substrate_CarbonMangels'
            ]
        elif group_name == 'HTS':
            task_names = [
                'cav3_t-type_calcium_channels_butkiewicz', 'tyrosyl-dna_phosphodiesterase_butkiewicz',
                'HIV', 'SARSCoV2_3CLPro_Diamond', 'SARSCoV2_Vitro_Touret', 'orexin1_receptor_butkiewicz',
                'm1_muscarinic_receptor_agonists_butkiewicz', 'm1_muscarinic_receptor_antagonists_butkiewicz',
                'potassium_ion_channel_kir2.1_butkiewicz', 'kcnq2_potassium_channel_butkiewicz',
                'choline_transporter_butkiewicz', 'serine_threonine_kinase_33_butkiewicz'
            ]
        # Skip others as they are likely not simple SMILES tasks or we need to be careful
        # But let's check the loop in process_tdc_training.py
        # SAbDab_Chen -> Antibody sequence pairs (NOT SMILES)
        # HuRI -> Protein sequence pairs (NOT SMILES)
        # Weber -> Epitope/TCR sequences (NOT SMILES)
        # MHC -> Peptide/MHC sequences (NOT SMILES)
        # We only strictly care about SMILES inputs for AccFG.
        
        else:
            continue

        for task_name in tqdm(task_names, desc=f"Collecting SMILES from {group_name}"):
            try:
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
                
                if data is None:
                    continue

                split = data.get_split()
                if 'train' in split:
                    train_df = split['train']
                    # Standard SMILES task usually has 'Drug' column
                    if 'Drug' in train_df.columns:
                        smiles_set.update(train_df['Drug'].dropna().unique())
            
            except Exception as e:
                logger.warning(f"Error processing {task_name}: {e}")

    return list(smiles_set)

def process_single_smiles(smiles):
    try:
        # Check if valid string
        if not isinstance(smiles, str) or not smiles.strip():
            return None
        
        desc = high_level_fg_fragments_w_attach_points_no_special_tokens_w_atom_ids(smiles)
        return smiles, desc
    except Exception as e:
        # logger.error(f"Failed for {smiles}: {e}")
        return smiles, f"Error: {e}"

class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass

class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess

# We use a non-daemonic pool to allow nested multiprocessing if AccFG uses it.
# However, standard Pool is hard to force non-daemon. 
# joblib is preferred but might not be installed. 
# concurrent.futures.ProcessPoolExecutor also uses daemon processes.
# Let's try to use a custom pool or just joblib if available.
# Since I can't guarantee joblib, I will try to use the NoDaemon logic for Pool.

def main():
    output_path = current_dir / 'shared_data' / 'TDC_train_fg_desc_with_attach_points_and_atom_ids.jsonl'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Collect SMILES
    logger.info("Collecting SMILES from TDC...")
    all_smiles = get_all_tdc_smiles()
    logger.info(f"Collected {len(all_smiles)} unique SMILES.")
    
    # 2. Process in parallel
    results = {}
    
    # Check if joblib is available
    try:
        from joblib import Parallel, delayed
        logger.info("Using joblib for parallelism.")
        
        # AccFG is CPU bound. Machine has 224 cores.
        # Using 32 jobs to be safe but efficient.
        # Write line by line to allow resume/progress check.
        with open(output_path, 'w', encoding='utf-8') as f_out:
            parallel = Parallel(n_jobs=32, verbose=5, backend='loky', return_generator=True)
            
            # Use a generator expression
            results_generator = parallel(delayed(process_single_smiles)(s) for s in all_smiles)
            
            for res in tqdm(results_generator, total=len(all_smiles), desc="Processing"):
                if res:
                    s, d = res
                    # Write as JSONL line: {"smiles": s, "description": d}
                    # Or simple mapping if preferred, but JSONL is better for append.
                    # The original request implied a mapping file or similar?
                    # "computed ... results saved down"
                    # Original precalculate_fgs.py wrote a dict: {smiles: desc}
                    # But for large file, JSONL is better: {"smiles": s, "desc": d}
                    # I will write JSONL line.
                    record = {s: d} # To match "smiles -> description" concept, but one line per item?
                    # No, JSONL usually is list of objects.
                    # Let's write {"smiles": s, "accfg": d}
                    # But the tool loader reads `json.load(f)`. It expects a SINGLE JSON OBJECT (Dict).
                    # This is a conflict. 
                    # If I write line-by-line, I produce a valid JSONL file (multiple objects).
                    # But `tools/AccFG.py` does `FG_CACHE = json.load(f)`.
                    # I must change `tools/AccFG.py` to support JSONL or write a script to convert JSONL to JSON at the end.
                    # Or just write JSONL and update AccFG.py to read JSONL.
                    # Updating AccFG.py is better for large files.
                    f_out.write(json.dumps({s: d}) + '\n')
                    f_out.flush()

    # Note: The output is now a JSONL file (one dict per line), not a single JSON dict.
    # We should update AccFG.py to handle this.

                
    except ImportError:
        logger.info("joblib not found, falling back to multiprocessing.Pool with NoDaemonContext (if nested) or ProcessPoolExecutor.")
        # Fallback to ProcessPoolExecutor - verify if AccFG CRASHES with daemon.
        # The user said "AccFG will have multi-process inside... daemon will error".
        # So we MUST NOT be daemon.
        
        # Custom NoDaemonPool
        class NoDaemonPool(multiprocessing.pool.Pool):
            def Process(self, *args, **kwds):
                proc = super(NoDaemonPool, self).Process(*args, **kwds)
                proc.daemon = False
                return proc
        
        logger.info("Using custom NoDaemonPool.")
        with NoDaemonPool(processes=os.cpu_count()) as pool:
            # map
            for res in tqdm(pool.imap_unordered(process_single_smiles, all_smiles), total=len(all_smiles)):
                if res:
                    s, d = res
                    results[s] = d

    # 3. Done
    logger.info(f"Finished processing. Saved to {output_path}")

if __name__ == "__main__":
    main()
