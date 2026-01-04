from tdc.single_pred import ADME

def main():
    task_name = 'CYP2C9_Substrate_CarbonMangels'
    print(f"Loading task: {task_name}")
    # Initialize the ADME task
    data = ADME(name=task_name)
    # Get the data split (train/val/test)
    split = data.get_split()
    
    # Check for test set
    if 'test' in split:
        test_df = split['test']
        # The 'Drug' column usually contains the SMILES or molecule identifiers
        num_molecules = len(test_df)
        print(f"Number of molecules in '{task_name}' test set: {num_molecules}")
    else:
        print("Test set not found for this task.")

if __name__ == "__main__":
    main()
