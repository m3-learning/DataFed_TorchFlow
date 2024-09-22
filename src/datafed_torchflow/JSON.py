import json
import numpy as np


class UniversalEncoder(json.JSONEncoder):
    def default(self, obj):
        # Convert numpy types to their Python equivalents
        if isinstance(obj, np.integer):
            return int(obj)  # Convert numpy integers to Python int
        elif isinstance(obj, np.floating):
            return float(obj)  # Convert numpy floats to Python float
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy arrays to lists
        elif isinstance(obj, set):
            return list(obj)  # Convert sets to lists
        elif hasattr(obj, "__dict__"):
            return obj.__dict__  # Serialize object attributes
        else:
            # Call the default method for other cases
            return super().default(obj)
