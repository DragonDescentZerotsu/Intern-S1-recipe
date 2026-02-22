import os
import glob

def merge_train_data():
    input_dir = "DataPrepare/Slime_RL_data/by_task/train"
    output_file = "DataPrepare/Slime_RL_data/merged/train.jsonl"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    count = 0
    with open(output_file, "w") as outfile:
        for filename in sorted(glob.glob(os.path.join(input_dir, "*.jsonl"))):
            with open(filename, "r") as infile:
                for line in infile:
                    outfile.write(line)
            count += 1
    print(f"Merged {count} files into {output_file}")

if __name__ == "__main__":
    merge_train_data()
