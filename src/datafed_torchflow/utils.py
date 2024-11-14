from m3util.notebooks.checksum import calculate_notebook_checksum
import torch
import numpy as np
import json 
import ast
import inspect

def is_jsonable(x):
    """
    Checks whether input is JSON serializable or not
    
    Args:
        x (any data type): the object to check for JSON serializability
    
    Returns: 
        bool: `True` if the object is JSON serializable; `False` otherwise.
    """
    try:
        json.dumps(x)
        return True
    except:
        return False 
def extract_instance_attributes(obj=dict()):
    """
    Recursively extracts attributes from class instances, converting NumPy integers to Python int,
    NumPy arrays and Torch tensors to lists, while ignoring keys that start with '_'.

    This helper function traverses the attributes of a given object and returns a dictionary
    representation of those attributes. If the object has a `__dict__` attribute, it means
    the object is likely an instance of a class, and its attributes are stored in `__dict__`.
    The function will recursively call itself to extract attributes from nested objects, 
    convert any NumPy integers to Python int, and convert NumPy arrays and Torch tensors to lists.

    Args:
        obj (object): The object from which to extract attributes. Defaults to an empty dictionary.

    Returns:
        dict: A dictionary containing the extracted attributes, excluding those whose keys start with '_'.
    """
    if hasattr(obj, "__dict__"):
        return {
            key: extract_instance_attributes(
                int(value) if isinstance(value, np.integer)
                #else value.tolist() if isinstance(value, (np.ndarray, torch.Tensor))
                else value
            )
            for key, value in obj.__dict__.items()
            if not key.startswith("_") and is_jsonable(value)
        }
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, (np.ndarray, torch.Tensor)):
        return obj.tolist()
    else:
        return obj

def get_return_variables(func):
    """
    Recursively extracts variable names from the return statement of a given function.

    This function takes another function as input, parses its source code into an 
    Abstract Syntax Tree (AST) and returns the variable names from the other function's
    return statement as a list
    
    Args: 
        func (function): The function whose return variables are to be extracted.

    Returns: 
        list of str: A list of variable names returned by the input function. If the function has 
        no return statement or returns a constant (not a variable), the list will be empty.
    
    """

    # Get the source code of the function
    source = inspect.getsource(func)
    
    # Parse the source code into an AST
    tree = ast.parse(source)
    
    # Navigate to the function definition in the AST
    function_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    
    # Extract the return statement
    return_vars = []
    for node in ast.walk(function_node):
        if isinstance(node, ast.Return):
            # Check if the return value is a tuple or a single value
            if isinstance(node.value, ast.Tuple):
                return_vars = [elt.id for elt in node.value.elts if isinstance(elt, ast.Name)]
            elif isinstance(node.value, ast.Name):
                return_vars = [node.value.id]
            break
            
    return return_vars

def clean_empty(data):
    """
    Recursively removes entries with empty values in the nested metadata dictionary. 
    Empty is defined as empty strings, "" , dictionaries, {} , lists , [], or tuples, ()
    but NOT the number zero or the boolean False. This is defined in the ``is_empty"
    helper function. 
    
    Args: 
        data (dict, list, or tuple): 

    Returns:
        The same datatype and data as the input `data`, just without empty values   
    """
    if isinstance(data, dict):
        # Recursively clean each item in the dictionary
        return {k: clean_empty(v) for k, v in data.items() if not is_empty(v)}
    elif isinstance(data, list):
        # Recursively clean each item in the list and remove any empty entries
        cleaned_list = [clean_empty(item) for item in data if not is_empty(item)]
        return cleaned_list if not is_empty(cleaned_list) else []
    elif isinstance(data, tuple):
        # Recursively clean each item in the tuple and remove any empty entries
        cleaned_tuple = tuple(clean_empty(item) for item in data if not is_empty(item))
        return cleaned_tuple if not is_empty(cleaned_tuple) else ()
    else:
        # Return the item as it is if it's not a list, dict, or tuple
        return data

def is_empty(value):
    """Helper function to determine if a value should be considered 'empty'.
    Args: 
        value (string, dict, list, or tuple): the object from which to remove empty values

    Returns:
        The same datatype and data as the input `data`, just without empty values   

    """
    if value == "" or value == {} or value == [] or value == ():
        return True
    elif isinstance(value, list) or isinstance(value, tuple):
        # Check if all elements in a list or tuple are empty
        return all(is_empty(item) for item in value)
    elif isinstance(value, dict):
        # Check if all values in a dictionary are empty
        return all(is_empty(v) for v in value.values())
    return False

def getNotebookMetadata(file):
    """
    Calculates the checksum of the script or notebook file and includes it in the metadata.

    Args: 
        file (string) the script or notebook file path
    
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

    Args: 
        model_block (custom class): the model architecture block (i.e. `encoder`, `decoder`, etc.) 
        to be serialized.
    
    Returns:
        dict: A dictionary containing the model block's architecture with layer types,
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

    Args: 
        optimizer (torch.optim): the model optimizer to be serialized.
    
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
