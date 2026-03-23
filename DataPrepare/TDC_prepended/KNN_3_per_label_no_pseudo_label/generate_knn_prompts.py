import os
import json
import logging
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from tqdm import tqdm
from accfg import compare_mols
from huggingface_hub import hf_hub_download
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

# Local imports
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from tools import BASIC_TOOLS, get_function_by_name
from tools.RDKit_tools import TDC_RDKIT_SPECIFIC_OPENAI_TOOLS_MAP
from tools.AccFG import find_mcs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EXCLUDED_PROPERTY_TOOL_NAMES = {"remove_salts"}


def configure_single_thread_runtime():
    # Avoid nested parallelism inside BLAS/RDKit-adjacent dependencies for each worker.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Keep this pipeline on CPU to avoid multi-process GPU contention.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


configure_single_thread_runtime()


def normalize_smiles_for_lookup(smiles: str) -> str:
    """Match the desalted canonical SMILES used in fingerprint generation."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles

    try:
        lfc = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
        cleaned = lfc.choose(mol)
        if cleaned is not None:
            return Chem.MolToSmiles(cleaned, canonical=True, isomericSmiles=True)
    except Exception:
        pass

    return smiles

def get_tools_for_task(task_name):
    specific_tools = TDC_RDKIT_SPECIFIC_OPENAI_TOOLS_MAP.get(task_name, [])
    return [
        tool
        for tool in (BASIC_TOOLS + specific_tools)
        if tool["function"]["name"] not in EXCLUDED_PROPERTY_TOOL_NAMES
    ]

def compute_properties(smiles, tools_list):
    output = []
    for tool in tools_list:
        name = tool['function']['name']
        func = get_function_by_name(name)
        if func is None:
            continue
        
        # We only compute zero-argument string/float returning functions
        # Skip tools that need additional arguments like patterns or reference_smiles
        if name in ['match_substructure', 'find_mcs', 'evaluate_arithmetic', 'score_structural_alerts']:
            continue
            
        try:
            # Most tools just take smiles as the single positional/keyword arg
            res = func(smiles=smiles)
            if isinstance(res, str):
                pass
            
            # Extract just the result logic (typically colon separated)
            str_res = str(res)
            # if ":" in str_res:
            #    str_res = str_res.split(":", 1)[1].strip()
            
            output.append(f"- {str_res}")
        except Exception as e:
            pass # Skip silently on error
            
    return "\n".join(output)

def format_compare_mols(query_smiles, ref_smiles):
    try:
        try:
            # First attempt with default similarity
            res = compare_mols(query_smiles, ref_smiles, similarityThreshold=0.2)
        except ValueError:
            # If MCES fails (ValueError: Error on [...]), retry with low similarityThreshold
            res = compare_mols(query_smiles, ref_smiles, similarityThreshold=0.1)
            
        (q_rings, q_fgs), (r_rings, r_fgs) = res
        output = "Differences in Functional Groups (via compare_mols):"
        output += "\n  Unique to Query Molecule:"
        if not q_rings and not q_fgs:
            output += " None"
        else:
            for name, count, _ in q_rings + q_fgs:
                output += f" {count}x {name},"
        
        output += "\n  Unique to Reference Molecule:"
        if not r_rings and not r_fgs:
            output += " None"
        else:
            for name, count, _ in r_rings + r_fgs:
                output += f" {count}x {name},"
                
        return output
    except Exception as e:
        return f"Differences in Functional Groups: Error computing ({str(e)})"


def load_true_labels(task_name, split):
    labels_map_path = (
        project_root
        / "DataPrepare"
        / "TDC_mol_fingerprints"
        / "Label_maps"
        / "by_task"
        / task_name
        / f"{split}_labels.jsonl"
    )
    if labels_map_path.exists():
        labels = {}
        with open(labels_map_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                canonical_smiles = record.get("canonical_smiles")
                if canonical_smiles and record.get("Y") is not None:
                    labels[canonical_smiles] = record.get("Y")
        logger.info(f"Loaded true labels from canonicalized label map: {labels_map_path}")
        return labels

    labels_path = project_root / f"DataPrepare/TDC_{split}_prompts_label_scaffold" / f"{task_name}.jsonl"
    if not labels_path.exists():
        logger.warning(f"Missing true-label file for {task_name} {split}: {labels_path}")
        return {}

    labels = {}
    with open(labels_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record.get("text", "")
            if "Drug SMILES: " not in text:
                continue
            drug_smiles = text.split("Drug SMILES: ", 1)[1].splitlines()[0].strip()
            labels[normalize_smiles_for_lookup(drug_smiles)] = record.get("Y")
    return labels


def build_prompt_for_query(
    index,
    query_smiles,
    m_data,
    f_data,
    pseudo_label,
    prompt_template,
    tools_list,
    true_label,
    top_k,
    retrieval_mode,
    include_pseudo_label,
):
    def parse_to_dict(sim_data):
        ref_dict = {}
        for label_name, label_val in [("label_0", 0), ("label_1", 1)]:
            for score, ref_sm in sim_data[label_name]:
                ref_dict[ref_sm] = {"score": score, "label": label_val}
        return ref_dict

    m_refs = parse_to_dict(m_data)
    f_refs = parse_to_dict(f_data)

    all_ref_smiles = set(list(m_refs.keys()) + list(f_refs.keys()))
    weighted_scores = []
    for ref_smiles in all_ref_smiles:
        m_score = m_refs.get(ref_smiles, {}).get("score", 0.0)
        f_score = f_refs.get(ref_smiles, {}).get("score", 0.0)
        ref_label = m_refs.get(ref_smiles, {}).get("label")
        if ref_label is None:
            ref_label = f_refs.get(ref_smiles, {}).get("label")

        final_score = (0.8 * m_score) + (0.2 * f_score)
        weighted_scores.append((final_score, ref_label, ref_smiles))

    weighted_scores = sorted(weighted_scores, key=lambda x: x[0], reverse=True)
    label_1_refs = [x for x in weighted_scores if x[1] == 1][:top_k]
    label_0_refs = [x for x in weighted_scores if x[1] == 0][:top_k]
    global_top_refs = weighted_scores[: 2 * top_k]

    base_instruction = prompt_template.replace("{Drug SMILES}", query_smiles)
    base_instruction = base_instruction.replace(
        'Answer:',
        'Please think step by step and then put ONLY your final choice ((A) or (B)) after "Answer:"'
    )

    prompt = f"{base_instruction.split('Drug SMILES:')[0]}Drug SMILES: {query_smiles}\n\n"

    query_properties = compute_properties(query_smiles, tools_list)

    prompt += "Properties of the query molecule:\n"
    prompt += f"{query_properties}\n\n"
    prompt += "Here are some of the molecules that are similar to the query molecule:\n\n"

    if retrieval_mode == "per_label":
        prompt += f"--- Top {top_k} similar molecules with label (B) ---\n"
        for idx, (score, label, ref_sm) in enumerate(label_1_refs, start=1):
            ref_props = compute_properties(ref_sm, [t for t in tools_list if t['function']['name'] != 'describe_high_level_fg_fragments'])
            mcs_result = find_mcs(query_smiles, [ref_sm])
            fg_diff = format_compare_mols(query_smiles, ref_sm)

            prompt += f"Reference {idx} SMILES: {ref_sm}\n"
            prompt += f"Computed Weighted Similarity: {score:.4f}\n"
            prompt += f"Properties:\n{ref_props}\n"
            prompt += f"{mcs_result}\n"
            prompt += f"{fg_diff}\n\n"

        prompt += f"--- Top {top_k} similar molecules with label (A) ---\n"
        for idx, (score, label, ref_sm) in enumerate(label_0_refs, start=1):
            ref_props = compute_properties(ref_sm, [t for t in tools_list if t['function']['name'] != 'describe_high_level_fg_fragments'])
            mcs_result = find_mcs(query_smiles, [ref_sm])
            fg_diff = format_compare_mols(query_smiles, ref_sm)

            prompt += f"Reference {idx} SMILES: {ref_sm}\n"
            prompt += f"Computed Weighted Similarity: {score:.4f}\n"
            prompt += f"Properties:\n{ref_props}\n"
            prompt += f"{mcs_result}\n"
            prompt += f"{fg_diff}\n\n"
    else:
        prompt += f"--- Top {2 * top_k} most similar molecules overall ---\n"
        for idx, (score, label, ref_sm) in enumerate(global_top_refs, start=1):
            ref_props = compute_properties(ref_sm, [t for t in tools_list if t['function']['name'] != 'describe_high_level_fg_fragments'])
            mcs_result = find_mcs(query_smiles, [ref_sm])
            fg_diff = format_compare_mols(query_smiles, ref_sm)
            label_str = "(B)" if label == 1 else "(A)"

            prompt += f"Reference {idx} SMILES: {ref_sm}\n"
            prompt += f"Reference Label: {label_str}\n"
            prompt += f"Computed Weighted Similarity: {score:.4f}\n"
            prompt += f"Properties:\n{ref_props}\n"
            prompt += f"{mcs_result}\n"
            prompt += f"{fg_diff}\n\n"

    if include_pseudo_label:
        pseudo_label_str = "(B)" if pseudo_label == 1 else "(A)"
        prompt += f"The pseudo label from naive Morgan fingerprint KNN prediction is {pseudo_label_str}. You should compare the molecules carefully and decide to agree with the pseudo label or not.\n\n"
        prompt += 'Please think step by step and then put ONLY your final choice ((A) or (B)) after "Answer:"'
    else:
        prompt += 'Please think step by step, compare the molecules carefully, and then put ONLY your final choice ((A) or (B)) after "Answer:"'

    return index, {
        "text": prompt,
        "drug": query_smiles,
        "Y": true_label,
    }

def process_single_task(
    task_name,
    split,
    n_examples=None,
    num_workers=1,
    top_k=3,
    retrieval_mode="per_label",
    include_pseudo_label=True,
):
    base_dir = project_root / "DataPrepare" / "TDC_mol_fingerprints"
    output_dir_name = f"KNN_{top_k}"
    if retrieval_mode != "per_label":
        output_dir_name += "_global_top2k"
    if not include_pseudo_label:
        output_dir_name += "_no_pseudo_label"

    out_root = project_root / "DataPrepare" / "TDC_prepended" / output_dir_name / split
    out_root.mkdir(parents=True, exist_ok=True)
    
    # Load similarities
    morgan_sim_path = base_dir / "Morgan_similarity" / "by_task" / task_name / f"{split}_similarity.pkl"
    feat_morgan_sim_path = base_dir / "Feature_Morgan_similarity" / "by_task" / task_name / f"{split}_similarity.pkl"
    pseudo_labels_path = (
        base_dir
        / "KNN_pesudo_labels"
        / split
        / "by_task"
        / task_name
        / f"{split}_knn_labels.json"
    )
    
    required_paths = [morgan_sim_path, feat_morgan_sim_path]
    if include_pseudo_label:
        required_paths.append(pseudo_labels_path)

    if not all(path.exists() for path in required_paths):
        logger.warning(f"Missing required files for {task_name} {split}. Skipping.")
        return
        
    with open(morgan_sim_path, "rb") as f:
        morgan_sims = pickle.load(f)
    with open(feat_morgan_sim_path, "rb") as f:
        feat_morgan_sims = pickle.load(f)
    pseudo_labels = {}
    if include_pseudo_label:
        with open(pseudo_labels_path, "r") as f:
            pseudo_labels = json.load(f)

    true_labels = load_true_labels(task_name, split)
        
    # Load TDC prompts
    tdc_prompts_filepath = hf_hub_download(
        repo_id="google/txgemma-9b-predict",
        filename="tdc_prompts.json",
    )
    with open(tdc_prompts_filepath, "r") as f:
        tdc_prompts_json = json.load(f)
        
    if 'SARSCOV2_3CLPro_Diamond' in tdc_prompts_json:
        tdc_prompts_json['SARSCoV2_3CLPro_Diamond'] = tdc_prompts_json['SARSCOV2_3CLPro_Diamond']
        
    prompt_template = tdc_prompts_json.get(task_name, "")
    if not prompt_template:
        logger.warning(f"No prompt template found for {task_name}")
        return
        
    tools_list = get_tools_for_task(task_name)
    
    selected_queries = list(morgan_sims.keys())
    if n_examples is not None:
        selected_queries = selected_queries[:n_examples]

    missing_true_labels = [
        query_smiles
        for query_smiles in selected_queries
        if true_labels.get(normalize_smiles_for_lookup(query_smiles)) is None
    ]
    if missing_true_labels:
        logger.warning(
            f"{task_name} {split}: {len(missing_true_labels)} queries still have no true label after normalization. "
            f"Example: {missing_true_labels[0]}"
        )

    jobs = []
    skipped_missing_labels = 0
    for query_smiles in selected_queries:
        true_label = true_labels.get(normalize_smiles_for_lookup(query_smiles))
        if true_label is None:
            skipped_missing_labels += 1
            logger.warning(f"Skip {task_name} {split} query with missing true label: {query_smiles}")
            continue

        job_index = len(jobs)
        jobs.append(
            (
                job_index,
                query_smiles,
                morgan_sims[query_smiles],
                feat_morgan_sims[query_smiles],
                pseudo_labels.get(query_smiles, "Unknown"),
                prompt_template,
                tools_list,
                true_label,
                top_k,
                retrieval_mode,
                include_pseudo_label,
            )
        )

    if skipped_missing_labels:
        logger.warning(f"{task_name} {split}: skipped {skipped_missing_labels} queries with missing true labels.")

    output_records = [None] * len(jobs)
    progress = tqdm(total=len(jobs), desc=f"Processing {task_name} {split}")

    if num_workers <= 1:
        for job in jobs:
            index, record = build_prompt_for_query(*job)
            output_records[index] = record
            progress.update(1)
    else:
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=get_context("spawn"),
            initializer=configure_single_thread_runtime,
        ) as executor:
            futures = [executor.submit(build_prompt_for_query, *job) for job in jobs]
            for future in as_completed(futures):
                index, record = future.result()
                output_records[index] = record
                progress.update(1)

    progress.close()

    out_jsonl = out_root / f"{task_name}.jsonl"
    with open(out_jsonl, "w") as f:
        for record in output_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate KNN augmented Prompts for TDC")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[
            # "BBB_Martins",
            # "DILI",
            # "Pgp_Broccatelli",
            "PAMPA_NCATS",
            "HIA_Hou",
            "Bioavailability_Ma",
            "hERG",
            "AMES",
            "Skin_Reaction",
            "ClinTox",
            "Carcinogens_Lagunin",
            "CYP2C9_Substrate_CarbonMangels",
            "CYP2D6_Substrate_CarbonMangels",
            "CYP3A4_Substrate_CarbonMangels",
            "SARSCoV2_3CLPro_Diamond",
            "SARSCoV2_Vitro_Touret",
        ],
    )
    parser.add_argument("--splits", nargs="+", default=["train", "valid"])
    parser.add_argument("--n-examples", type=int, default=None, help="Number of examples to process (default: all), for quick testing")
    parser.add_argument("--num-workers", type=int, default=min(4, os.cpu_count() or 1), help="Number of worker processes for parallel prompt generation")
    parser.add_argument("--top-k", type=int, default=3, help="Number of most similar molecules to keep for each label")
    parser.add_argument(
        "--retrieval-mode",
        choices=["per_label", "global_top2k"],
        default="per_label",
        help="Choose whether to keep top-k molecules per label or the overall top-2*top-k most similar molecules",
    )
    parser.add_argument(
        "--include-pseudo-label",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the naive Morgan fingerprint KNN pseudo label in the prompt (use --no-include-pseudo-label to disable)",
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be a positive integer")
    
    for task in args.tasks:
        for split in args.splits:
            process_single_task(
                task,
                split,
                n_examples=args.n_examples,
                num_workers=args.num_workers,
                top_k=args.top_k,
                retrieval_mode=args.retrieval_mode,
                include_pseudo_label=args.include_pseudo_label,
            )

if __name__ == "__main__":
    main()
