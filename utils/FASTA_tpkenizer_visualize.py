from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "internlm/Intern-S1-mini-FP8"

# 1. 加载
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto",
)

seqs = [
    "MVPSAGQLALFALGIVLAACQALENSTSPLSADPPVAAAVVSHFNDCPDSHTQFCFHGTCRFLVQEDKPACVCHSGYVGARCEHADLLAVVAASQKKQAITALVVVSIVALAVLIITCVLIHCCQVRKHCEWCRALICRHEKPSALLKGRTACCHSETVV*MVPSAGQLALFALGIVLAACQALENSTSPLSDPPVAAAVVSHFNDCPDSHTQFCFHGTCRFLVQEDKPACVCHSGYVGARCEHADLLAVVAASQKKQAITALVVVSIVALAVLIITCVLIHCCQVRKHCEWCRALICRHEKPSALLKGRTACCHSETVV*XPPVAAAVVSHFNDCPDSHTQFCFHGTCRFLVQEDKPACVCHSGYVGARCEHADLLAVVAASQKKQAITALVVVSIVALAVLIITCVLIHCCQVRKHCEWCRALICRHEKPSALLKGRTACCHSETGCRLY*MVPSAGQLALFALGIVLAACQALENSTSPLSDPPVAAAVVSHFNDCPDSHTQFCFHGTCRFLVQEDKPACVCHSGYVGARCEHADLLAVVAASQKKQAITALVVVSIVALAVLIITCVLIMFKIGRGALDLFSELLSFGGIVLAACQALENSTSPLSADPPVAAAVVSHFNDCPDSHTQFCFHGTCRFLVQEDKPACVCHSGYVGARCEHADLLAVVAASQKKQAITALVVVSIVALAVLIITCVLIHCCQVRKHCEWCRALICRHEKPSALLKGRTACCHSETVV*MVPSAGQLALFALGIVLAACQALENSTSPLSDPPVAAAVVSHFNDCPDSHTQFCFHGTCRFLVQEDKPACVCHSGYVGARCEHADLLAVVAASQKKQAITALVVVSIVALAVLIITCVLIHCCQVRKHCEWCRALICRHEKPSALLKGRTACCHSETATLG*MFKIGRGALDLFSELLSFGGIVLAACQALENSTSPLSDPPVAAAVVSHFNDCPDSHTQFCFHGTCRFLVQEDKPACVCHSGYVGARCEHADLLAVVAASQKKQAITALVVVSIVALAVLIITCVLIHCCQVRKHCEWCRALICRHEKPSALLKGRTACCHSETVV*",   # 34 aa
    "ACDEFGHIKLMNPQRST"                    # 17 aa
]

# 想看的几个 FASTA 子词
target_tokens = ["AC", "DE", "FG", "HI"]

# 拿到整张 embedding 表
emb_table = model.get_input_embeddings().weight  # (vocab_size, hidden_size)

for seq in seqs:
    print("\nseq:", seq)
    # 2. 先 tokenize
    tokens = tokenizer.tokenize(seq)
    print("tokens:", tokens)

    # 3. 转成 id（必须整体转，这样 auto-detect 的逻辑才生效）
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print("token_ids:", token_ids)

    # 4. 针对每一个想看的 token，去这条序列里找
    for tt in target_tokens:
        positions = [i for i, t in enumerate(tokens) if t == tt]
        if not positions:
            print(f"  token '{tt}' not found in this seq.")
            continue

        # 这条序列里可能出现多次，我们就都打印一下
        for pos in positions:
            tok_id = token_ids[pos]
            emb = emb_table[tok_id].detach().cpu()

            print(f"\n  token '{tt}' at position {pos}:")
            print("    id:", tok_id)
            print("    shape:", emb.shape)
            # 打印前 100 维，够看分布了
            print("    first 100 dims:", emb[:10])
            print("    L2 norm:", float(emb.norm()))
