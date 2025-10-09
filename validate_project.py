#!/usr/bin/env python3
"""
Test script to validate project structure and code syntax.
This script checks if all required files exist and Python scripts compile correctly.
"""

import os
import sys
import py_compile
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists."""
    if os.path.exists(filepath):
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description} NOT FOUND: {filepath}")
        return False

def check_python_syntax(filepath):
    """Check if Python file has valid syntax."""
    try:
        py_compile.compile(filepath, doraise=True)
        print(f"✓ Valid syntax: {filepath}")
        return True
    except py_compile.PyCompileError as e:
        print(f"✗ Syntax error in {filepath}: {e}")
        return False

def main():
    """Run validation tests."""
    print("="*60)
    print("PROJECT STRUCTURE VALIDATION")
    print("="*60)
    
    project_root = Path(__file__).parent
    os.chdir(project_root)
    print(f"\nProject root: {project_root.absolute()}\n")
    
    all_passed = True
    
    # Check folders
    print("\n--- Checking Folder Structure ---")
    folders = [
        'data/raw',
        'data/processed',
        'notebooks',
        'scripts'
    ]
    
    for folder in folders:
        if os.path.isdir(folder):
            print(f"✓ Folder exists: {folder}/")
        else:
            print(f"✗ Folder missing: {folder}/")
            all_passed = False
    
    # Check required files
    print("\n--- Checking Required Files ---")
    required_files = [
        ('README.md', 'Main README'),
        ('requirements.txt', 'Dependencies file'),
        ('LICENSE', 'License file'),
        ('data/raw/README.md', 'Raw data README'),
        ('data/processed/README.md', 'Processed data README'),
        ('notebooks/README.md', 'Notebooks README'),
        ('notebooks/example_workflow.ipynb', 'Example notebook'),
    ]
    
    for filepath, description in required_files:
        if not check_file_exists(filepath, description):
            all_passed = False
    
    # Check Python scripts
    print("\n--- Checking Python Scripts ---")
    scripts = [
        'scripts/__init__.py',
        'scripts/data_preprocessing.py',
        'scripts/causal_inference.py',
        'scripts/irt_analysis.py',
        'scripts/classification_clustering.py',
        'scripts/nlp_analysis.py',
    ]
    
    for script in scripts:
        if not check_file_exists(script, f"Script"):
            all_passed = False
        elif not check_python_syntax(script):
            all_passed = False
    
    # Check README content
    print("\n--- Checking README Content ---")
    with open('README.md', 'r') as f:
        readme_content = f.read()
        
    required_sections = [
        'Overview',
        'Installation',
        'Usage',
        'Project Structure',
        'Key Features',
        'Dependencies'
    ]
    
    for section in required_sections:
        if section.lower() in readme_content.lower():
            print(f"✓ README contains '{section}' section")
        else:
            print(f"⚠ README might be missing '{section}' section")
    
    # Check requirements.txt
    print("\n--- Checking Requirements ---")
    with open('requirements.txt', 'r') as f:
        requirements = f.read()
        
    critical_packages = [
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'econml',
        'jupyter'
    ]
    
    for package in critical_packages:
        if package.lower() in requirements.lower():
            print(f"✓ Requirements include: {package}")
        else:
            print(f"⚠ Requirements might be missing: {package}")
    
    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL VALIDATION CHECKS PASSED!")
        print("="*60)
        print("\nProject structure is complete and valid.")
        print("To use the project:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run scripts from the scripts/ directory")
        print("  3. Open notebooks with: jupyter notebook")
        return 0
    else:
        print("✗ SOME VALIDATION CHECKS FAILED")
        print("="*60)
        print("\nPlease review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
