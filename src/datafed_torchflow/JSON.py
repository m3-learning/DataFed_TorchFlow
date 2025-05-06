import json

import numpy as np


class UniversalEncoder(json.JSONEncoder):
    """
    A custom JSON encoder that can handle numpy data types, sets, and objects with __dict__ attributes.
    """

    def default(self, o):
        """
        Override the default method to provide custom serialization for unsupported data types.

        Parameters:
        obj (any): The object to serialize.

        Returns:
        any: The serialized form of the object.
        """
        # Convert numpy types to their Python equivalents
        if isinstance(o, np.integer):
            return int(o)  # Convert numpy integers to Python int
        elif isinstance(o, np.floating):
            return float(o)  # Convert numpy floats to Python float
        elif isinstance(o, np.ndarray):
            return o.tolist()  # Convert numpy arrays to lists
        elif isinstance(o, set):
            return list(o)  # Convert sets to lists
        elif hasattr(o, "__dict__"):
            return o.__dict__  # Serialize object attributes
        else:
            # Call the default method for other cases
            return super().default(o)
