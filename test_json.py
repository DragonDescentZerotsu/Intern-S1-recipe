import json
import torch
print(json.dumps({'shapes': [torch.Size([2, 3])]}))
