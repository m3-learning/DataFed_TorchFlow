import psutil
import json
import platform
import GPUtil
import pkg_resources


def get_system_info():
    """
    Extracts CPU, memory, GPU, and Python environment details.

    Returns:
        dict: A dictionary containing CPU, memory, GPU, and Python details.
    """
    system_info = {
        "cpu": get_cpu_info(),
        "memory": get_memory_info(),
        "gpu": get_gpu_info(),
        "python": get_python_info(),
    }

    return system_info


def get_cpu_info():
    """
    Retrieves CPU information.

    Returns:
        dict: CPU details including physical cores, total cores, frequency, and usage.
    """
    return {
        "physical_cores": psutil.cpu_count(logical=False),
        "total_cores": psutil.cpu_count(logical=True),
        "cpu_frequency": {
            "current": psutil.cpu_freq().current,
            "min": psutil.cpu_freq().min,
            "max": psutil.cpu_freq().max,
        },
        "cpu_usage_per_core": psutil.cpu_percent(percpu=True),
        "total_cpu_usage": psutil.cpu_percent(),
    }


def get_memory_info():
    """
    Retrieves memory information.

    Returns:
        dict: Memory details including total, available, used, and percentage used.
    """
    mem = psutil.virtual_memory()
    return {
        "total": f"{mem.total / (1024 ** 3):.2f} GB",
        "available": f"{mem.available / (1024 ** 3):.2f} GB",
        "used": f"{mem.used / (1024 ** 3):.2f} GB",
        "percent": mem.percent,
    }


def get_gpu_info():
    """
    Retrieves GPU information using GPUtil.

    Returns:
        dict: GPU details such as model, memory, and load.
    """
    try:
        gpus = GPUtil.getGPUs()
        gpu_info = []
        for gpu in gpus:
            gpu_info.append(
                {
                    "id": gpu.id,
                    "name": gpu.name,
                    "driver_version": gpu.driver,
                    "memory_total": f"{gpu.memoryTotal:.2f} MB",
                    "memory_used": f"{gpu.memoryUsed:.2f} MB",
                    "memory_free": f"{gpu.memoryFree:.2f} MB",
                    "load": f"{gpu.load * 100:.2f}%",
                    "temperature": f"{gpu.temperature} °C",
                }
            )
        return gpu_info if gpu_info else "No GPUs detected"
    except Exception as e:
        return {"error": str(e)}


def get_python_info():
    """
    Retrieves Python environment details, including version and installed packages.

    Returns:
        dict: Python details including version, implementation, and installed packages.
    """
    packages = {}
    installed_packages = pkg_resources.working_set
    for package in installed_packages:
        packages[package.key] = package.version

    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_build": platform.python_build(),
        "packages": packages,
    }


def save_to_json(data, filename="system_info.json"):
    """
    Saves the given data to a JSON file.

    Args:
        data (dict): The data to be saved.
        filename (str): The filename for the JSON file.
    """
    with open(filename, "w") as json_file:
        json.dump(data, json_file, indent=4)
