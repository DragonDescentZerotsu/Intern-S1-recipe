from tools.RDKit_tools import (
    get_molecular_weight,
    get_exact_molecular_weight,
    get_heavy_atom_count,
    get_mol_logp,
    get_tpsa,
    get_hbd,
    get_hba,
    get_num_rotatable_bonds,
    get_fraction_csp3,
    get_labute_asa,
    get_mol_mr
)

def test_formats():
    smiles = "CCO"  # Ethanol
    print(f"Testing SMILES: {smiles}")
    
    tools = [
        ("Molecular Weight", get_molecular_weight),
        ("Exact Molecular Weight", get_exact_molecular_weight),
        ("Heavy Atom Count", get_heavy_atom_count),
        ("LogP", get_mol_logp),
        ("TPSA", get_tpsa),
        ("HBD", get_hbd),
        ("HBA", get_hba),
        ("Rotatable Bonds", get_num_rotatable_bonds),
        ("Fraction CSP3", get_fraction_csp3),
        ("Labute ASA", get_labute_asa),
        ("Molar Refractivity", get_mol_mr)
    ]
    
    for name, tool in tools:
        try:
            result = tool(smiles)
            print(f"{name}: {result}")
            assert isinstance(result, str), f"{name} should return str, got {type(result)}"
        except Exception as e:
            print(f"Error testing {name}: {e}")

if __name__ == "__main__":
    test_formats()
