# check_environment.py
import sys
import json
import platform
import os

# Base info
env_info = {
    "python_version": platform.python_version(),
    "system": platform.system(),
    "platform": platform.platform(),
    "path": sys.path,
    "executable": sys.executable,
    "modules": {},
    "environment_variables": {
        "PYTHONPATH": os.environ.get("PYTHONPATH", "Not set"),
        "PATH": os.environ.get("PATH", "Not set")
    }
}

# Check required modules
required_modules = [
    "numpy", "pandas", "sklearn", "matplotlib", 
    "weaviate", "json", "datetime", "argparse"
]

for module in required_modules:
    try:
        __import__(module)
        version = getattr(__import__(module), "__version__", "Unknown")
        env_info["modules"][module] = {"status": "installed", "version": version}
    except ImportError:
        env_info["modules"][module] = {"status": "not installed"}

# Check if the agentic_rag.py file is accessible
script_path = "agentic_rag.py"
env_info["file_checks"] = {
    "agentic_rag.py": {
        "exists": os.path.exists(script_path),
        "is_file": os.path.isfile(script_path) if os.path.exists(script_path) else False,
        "size": os.path.getsize(script_path) if os.path.exists(script_path) and os.path.isfile(script_path) else None,
        "permissions": oct(os.stat(script_path).st_mode & 0o777) if os.path.exists(script_path) else None
    }
}

# Check project directory structure
current_dir = os.getcwd()
env_info["project_directory"] = {
    "current_directory": current_dir,
    "parent_directory": os.path.dirname(current_dir),
    "visible_files": os.listdir(current_dir)[:10]  # Only show first 10 files for brevity
}

# Print as JSON
print(json.dumps(env_info, indent=2))