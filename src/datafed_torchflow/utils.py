from m3util.notebooks.checksum import calculate_notebook_checksum
import torch


def extract_instance_attributes(obj=dict()):
    """
    Recursively extracts attributes from class instances, ignoring keys that start with '_'.

    This helper function traverses the attributes of a given object and returns a dictionary
    representation of those attributes. If the object has a `__dict__` attribute, it means
    the object is likely an instance of a class, and its attributes are stored in `__dict__`.
    The function will recursively call itself to extract attributes from nested objects.

    Args:
        obj (object): The object from which to extract attributes. Defaults to an empty dictionary.

    Returns:
        dict: A dictionary containing the extracted attributes, excluding those whose keys start with '_'.
    """
    if hasattr(obj, "__dict__"):
        return {
            key: extract_instance_attributes(value)
            for key, value in obj.__dict__.items()
            if not key.startswith("_")
        }
    else:
        return obj


def getNotebookMetadata(file):
    """
    Calculates the checksum of the script or notebook file and includes it in the metadata.

    Returns:
        dict: A dictionary containing the path and checksum of the script or notebook file.
    """

    # If the script path is provided, calculate and include its checksum
    if file is not None:
        script_checksum = calculate_notebook_checksum(file)
        file_info = {"script": {"path": file, "checksum": script_checksum}}
        return file_info

def serialize_model(model_block):
    """
    Serializes the model architecture into a dictionary format with detailed layer information.

    Returns:
        dict: A dictionary containing the model's architecture with layer types,
                names, and configurations.
    """
    model_info = {}
    model_info["layers"] = {}

    top_level_count = 0  # Counter for the top-level layers
    num_top_level_layers = len(
        set(
            [
                layer_name.split(".")[0]
                for layer_name, _ in model_block.named_modules()
                if layer_name != ""
            ]
        )
    )
    pad_length = len(
        str(num_top_level_layers)
    )  # Padding length for top-level numbering

    for layer_name, layer in model_block.named_modules():
        # Skip the top layer which is the entire model itself
        if layer_name != "":
            # Split the layer name at the first '.'
            parts = layer_name.split(".", 1)
            top_level_name = parts[0]
            sub_name = parts[1] if len(parts) > 1 else None

            # Check if this is a new top-level layer (ignoring numbering)
            if top_level_name not in [
                key.split("-", 1)[1] for key in model_info["layers"].keys()
            ]:
                # Increment the top-level counter
                top_level_count += 1
                padded_count = str(top_level_count).zfill(pad_length)

                # Create a top-level entry with zero-padded numbering
                model_info["layers"][f"{padded_count}-{top_level_name}"] = {}

            # Reference the top-level dictionary without the number prefix
            current_level_key = [
                key
                for key in model_info["layers"].keys()
                if key.endswith(f"-{top_level_name}")
            ][0]
            current_level = model_info["layers"][current_level_key]

            # If there is a sub_name (nested), create nested entries
            if sub_name:
                sub_parts = sub_name.split(".")
                for sub in sub_parts[:-1]:
                    if sub not in current_level:
                        current_level[sub] = {}
                    current_level = current_level[sub]
                layer_key = sub_parts[-1]
            else:
                layer_key = top_level_name

            # Collect the layer information
            layer_descriptor = {
                "type": layer.__class__.__name__,
                "layer_name": layer_name,
                "config": {},
            }

            # Automatically collect layer parameters
            for param, value in layer.__dict__.items():
                # Filter out unnecessary attributes
                if not param.startswith("_") and not callable(value):
                    layer_descriptor["config"][param] = value

            # Add the layer descriptor under the correct key
            current_level[layer_key] = layer_descriptor

    return model_info


def serialize_pytorch_optimizer(optimizer):
    """
    Serializes the optimizer's state dictionary, converting tensors to lists for JSON compatibility.

    Returns:
        dict: A dictionary containing the optimizer's serialized parameters.
    """
    state_dict = optimizer.state_dict()
    state_dict_serializable = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            state_dict_serializable[key] = value.tolist()
        elif isinstance(value, list):
            # Convert tensors within lists to lists
            state_dict_serializable[key] = [
                v.tolist() if isinstance(v, torch.Tensor) else v for v in value
            ]
        else:
            state_dict_serializable[key] = value
    return state_dict_serializable["param_groups"][0]
