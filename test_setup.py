#!/usr/bin/env python3
"""
Test script to verify the figure caption and query generator setup.
"""

import os
import sys
import json
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import openai
        print("✓ OpenAI package available")
    except ImportError:
        print("✗ OpenAI package not found. Run: pip install openai")
        return False
    
    try:
        from PIL import Image
        print("✓ Pillow package available")
    except ImportError:
        print("✗ Pillow package not found. Run: pip install Pillow")
        return False
    
    try:
        from config import OPENAI_CONFIG, PROMPTS
        from openai_handler import OpenAIHandler
        from paper_processor import PaperProcessor
        from batch_processor import BatchProcessor
        print("✓ All local modules imported successfully")
    except ImportError as e:
        print(f"✗ Error importing local modules: {e}")
        return False
    
    return True

def test_openai_config():
    """Test OpenAI configuration."""
    print("\nTesting OpenAI configuration...")
    
    from config import OPENAI_CONFIG
    
    api_key = OPENAI_CONFIG.get("api_key")
    if not api_key:
        print("✗ OPENAI_API_KEY environment variable not set")
        print("  Please set it with: export OPENAI_API_KEY='your-key-here'")
        return False
    
    if api_key.startswith("sk-"):
        print("✓ OpenAI API key format looks correct")
    else:
        print("⚠ OpenAI API key format may be incorrect (should start with 'sk-')")
    
    return True

def test_paper_structure():
    """Test that the expected paper structure exists."""
    print("\nTesting paper structure...")
    
    base_path = "./extracted_pdfs"
    if not os.path.exists(base_path):
        print(f"✗ Base path not found: {base_path}")
        return False
    
    print(f"✓ Base path exists: {base_path}")
    
    # Look for at least one paper
    paper_found = False
    for root, dirs, files in os.walk(base_path):
        if "result.json" in files and "images" in dirs:
            print(f"✓ Found paper structure at: {root}")
            
            # Check if images directory has content
            images_dir = os.path.join(root, "images")
            if os.listdir(images_dir):
                print(f"✓ Images directory has content: {len(os.listdir(images_dir))} files")
                paper_found = True
                break
            else:
                print(f"⚠ Images directory is empty: {images_dir}")
    
    if not paper_found:
        print("✗ No valid paper structure found")
        return False
    
    return True

def test_json_structure():
    """Test that we can read a JSON file and it has expected structure."""
    print("\nTesting JSON structure...")
    
    base_path = "./extracted_pdfs"
    
    # Find first JSON file
    for root, dirs, files in os.walk(base_path):
        if "result.json" in files:
            json_path = os.path.join(root, "result.json")
            
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                print(f"✓ Successfully loaded JSON: {json_path}")
                
                # Check expected fields
                expected_fields = ["id", "pdf_path", "marker"]
                for field in expected_fields:
                    if field in data:
                        print(f"✓ Found expected field: {field}")
                    else:
                        print(f"⚠ Missing expected field: {field}")
                
                # Check if markdown content exists
                marker = data.get("marker", {})
                if isinstance(marker, dict) and "markdown" in marker:
                    content_length = len(marker["markdown"])
                    print(f"✓ Found markdown content ({content_length} characters)")
                else:
                    print("⚠ No markdown content found in marker field")
                
                return True
                
            except Exception as e:
                print(f"✗ Error reading JSON {json_path}: {e}")
                return False
    
    print("✗ No JSON files found")
    return False

def main():
    """Run all tests."""
    print("Figure Caption and Query Generator - Setup Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_openai_config,
        test_paper_structure,
        test_json_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()  # Empty line between tests
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("✓ All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Test with a single paper:")
        print("   python main.py process-single --path ./extracted_pdfs/arxiv/superconductors/arXiv:cond-mat_9909358")
        print("2. Check current status:")
        print("   python main.py stats --path ./extracted_pdfs")
        return 0
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 