
from transformers import AutoTokenizer
import json

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("internlm/Intern-S1-mini", trust_remote_code=True)

# Define tools as in the script
tools = [{
    'type': 'function',
    'function': {
        'name': 'describe_high_level_fg_fragments',
        'description': 'Parse SMILES string into functional groups with attachment points and mark atom ids in the SMILES string for better structure description.',
        'parameters': {
            'type': 'object',
            'properties': {
                'smiles': {
                    'type': 'string',
                    'description': 'The SMILES string to parse.'
                }
            },
            'required': [
                'smiles'
            ]
        }
    }
}]

# Mock messages
messages = [
    {'role': 'user', 'content': 'Test message'},
    {'role': 'assistant', 'content': None, 'reasoning_content': 'I am thinking...'}
]

# Attempt to apply chat template
try:
    print('\n\n\n'+tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=True, tools=tools))
except Exception as e:
    import traceback
    traceback.print_exc()
