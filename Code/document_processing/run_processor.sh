#!/bin/bash
echo "Setting up Python path..."

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Script directory: $SCRIPT_DIR"

# Navigate up to project root (assuming this is in Code/document_processing)
cd "$SCRIPT_DIR"
cd ../..
PROJECT_ROOT="$(pwd)"
echo "Project root: $PROJECT_ROOT"

# Create Sample_Data and vector_store directories if they don't exist
mkdir -p "$PROJECT_ROOT/Sample_Data/vector_store"

# Create __init__.py files to make them proper Python packages
if [ ! -f "$PROJECT_ROOT/Sample_Data/__init__.py" ]; then
    echo "Creating Sample_Data/__init__.py"
    echo "# This file makes Sample_Data a package" > "$PROJECT_ROOT/Sample_Data/__init__.py"
fi

if [ ! -f "$PROJECT_ROOT/Sample_Data/vector_store/__init__.py" ]; then
    echo "Creating Sample_Data/vector_store/__init__.py"
    echo "# This file makes vector_store a package" > "$PROJECT_ROOT/Sample_Data/vector_store/__init__.py"
fi

# Create a symlink to the fixed script
echo "Setting up fixed process_documents.py..."
cp "$SCRIPT_DIR/process_documents.py" "$SCRIPT_DIR/process_documents.py.bak"
cp "$SCRIPT_DIR/fixed_process_documents.py" "$SCRIPT_DIR/process_documents.py"

# Return to the document processing directory
cd "$SCRIPT_DIR"

# Print directory structure for debugging
echo "Directory structure:"
find "$PROJECT_ROOT/Sample_Data" -type d | sort

# Run the processor script with all arguments passed to this script
echo "Running document processor..."
python process_documents.py "$@"

echo "Done."