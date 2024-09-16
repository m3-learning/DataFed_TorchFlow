import json
import torch

class UniversalEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle PyTorch objects
        if isinstance(obj, torch.nn.Module):
            return str(obj)  # Convert PyTorch models or layers to string
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()  # Convert tensors to lists
        elif callable(obj):
            return str(obj)  # Convert functions or callables to strings
        elif isinstance(obj, set):
            return list(obj)  # Convert sets to lists
        elif hasattr(obj, '__dict__'):
            return obj.__dict__  # Serialize object attributes
        else:
            # Call the default method for other cases
            return super().default(obj)