#!/usr/bin/env python3
"""
Script to combine all figure_analysis.json files into a single comprehensive dataset.

This script:
1. Finds all figure_analysis.json files in the extracted_pdfs directory
2. Combines them into a single dataset with metadata
3. Provides statistics about the combined dataset
4. Saves the result as combined_figure_analysis.json
"""

import json
import os
import glob
from datetime import datetime
from typing import Dict, List, Any
import argparse


def find_figure_analysis_files(base_dir: str = "extracted_pdfs") -> List[str]:
    """Find all figure_analysis.json files recursively."""
    pattern = os.path.join(base_dir, "**", "figure_analysis.json")
    files = glob.glob(pattern, recursive=True)
    return sorted(files)


def load_figure_analysis(file_path: str) -> Dict[str, Any]:
    """Load and return a figure_analysis.json file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def extract_paper_info(file_path: str) -> Dict[str, str]:
    """Extract paper information from the file path."""
    parts = file_path.split(os.sep)
    
    # Try to find relevant path components
    paper_dir = os.path.dirname(file_path)
    paper_name = os.path.basename(paper_dir)
    
    # Extract category and subcategory if available
    category = ""
    subcategory = ""
    
    if len(parts) >= 3:
        if "arxiv" in parts or "apl" in parts:
            source = parts[-4] if len(parts) >= 4 else "unknown"
            category = parts[-3] if len(parts) >= 3 else "unknown"
            subcategory = parts[-2] if len(parts) >= 2 else "unknown"
        else:
            source = "unknown"
            category = parts[-3] if len(parts) >= 3 else "unknown"
            subcategory = parts[-2] if len(parts) >= 2 else "unknown"
    
    return {
        "source": source,
        "category": category,
        "subcategory": subcategory,
        "paper_name": paper_name,
        "file_path": file_path
    }


def combine_figure_analyses(base_dir: str = "extracted_pdfs") -> Dict[str, Any]:
    """Combine all figure_analysis.json files into a single dataset."""
    
    print("Finding figure_analysis.json files...")
    files = find_figure_analysis_files(base_dir)
    print(f"Found {len(files)} figure_analysis.json files")
    
    if not files:
        print("No figure_analysis.json files found!")
        return {"papers": [], "summary": {}}
    
    combined_data = {
        "metadata": {
            "combination_timestamp": datetime.now().isoformat(),
            "total_papers": 0,
            "total_figures": 0,
            "base_directory": base_dir,
            "source_files": []
        },
        "papers": [],
        "summary": {
            "by_source": {},
            "by_category": {},
            "by_method": {},
            "by_context": {}
        }
    }
    
    total_figures = 0
    
    print("\nProcessing files...")
    for i, file_path in enumerate(files, 1):
        print(f"Processing {i}/{len(files)}: {file_path}")
        
        # Load the analysis data
        analysis_data = load_figure_analysis(file_path)
        if not analysis_data:
            continue
            
        # Extract paper information from path
        paper_info = extract_paper_info(file_path)
        
        # Count figures in this paper
        figure_count = len(analysis_data.get("figure_analysis", {}))
        total_figures += figure_count
        
        # Create combined paper entry
        paper_entry = {
            "paper_info": paper_info,
            "original_data": analysis_data,
            "figure_count": figure_count
        }
        
        combined_data["papers"].append(paper_entry)
        combined_data["metadata"]["source_files"].append(file_path)
        
        # Update summary statistics
        source = paper_info["source"]
        category = paper_info["category"]
        
        # Count by source
        if source not in combined_data["summary"]["by_source"]:
            combined_data["summary"]["by_source"][source] = {"papers": 0, "figures": 0}
        combined_data["summary"]["by_source"][source]["papers"] += 1
        combined_data["summary"]["by_source"][source]["figures"] += figure_count
        
        # Count by category
        if category not in combined_data["summary"]["by_category"]:
            combined_data["summary"]["by_category"][category] = {"papers": 0, "figures": 0}
        combined_data["summary"]["by_category"][category]["papers"] += 1
        combined_data["summary"]["by_category"][category]["figures"] += figure_count
        
        # Count by method and context (from figure analysis)
        for figure_name, figure_data in analysis_data.get("figure_analysis", {}).items():
            method = figure_data.get("method", "unknown")
            context = figure_data.get("context_used", "unknown")
            
            if method not in combined_data["summary"]["by_method"]:
                combined_data["summary"]["by_method"][method] = 0
            combined_data["summary"]["by_method"][method] += 1
            
            if context not in combined_data["summary"]["by_context"]:
                combined_data["summary"]["by_context"][context] = 0
            combined_data["summary"]["by_context"][context] += 1
    
    # Update metadata
    combined_data["metadata"]["total_papers"] = len(combined_data["papers"])
    combined_data["metadata"]["total_figures"] = total_figures
    
    return combined_data


def print_summary(data: Dict[str, Any]):
    """Print a summary of the combined dataset."""
    metadata = data["metadata"]
    summary = data["summary"]
    
    print("\n" + "="*60)
    print("COMBINED FIGURE ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"Combination timestamp: {metadata['combination_timestamp']}")
    print(f"Total papers: {metadata['total_papers']}")
    print(f"Total figures: {metadata['total_figures']}")
    
    if metadata['total_papers'] > 0:
        avg_figures = metadata['total_figures'] / metadata['total_papers']
        print(f"Average figures per paper: {avg_figures:.1f}")
    
    print("\nBy Source:")
    for source, stats in summary["by_source"].items():
        print(f"  {source}: {stats['papers']} papers, {stats['figures']} figures")
    
    print("\nBy Category:")
    for category, stats in summary["by_category"].items():
        print(f"  {category}: {stats['papers']} papers, {stats['figures']} figures")
    
    print("\nBy Processing Method:")
    for method, count in summary["by_method"].items():
        print(f"  {method}: {count} figures")
    
    print("\nBy Context Used:")
    for context, count in summary["by_context"].items():
        print(f"  {context}: {count} figures")


def main():
    parser = argparse.ArgumentParser(description="Combine all figure_analysis.json files")
    parser.add_argument("--base-dir", default="extracted_pdfs", 
                       help="Base directory to search for figure_analysis.json files")
    parser.add_argument("--output", default="combined_figure_analysis.json",
                       help="Output file name")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save the combined file, just show summary")
    
    args = parser.parse_args()
    
    # Combine all figure analyses
    combined_data = combine_figure_analyses(args.base_dir)
    
    if not combined_data["papers"]:
        print("No data to combine!")
        return
    
    # Print summary
    print_summary(combined_data)
    
    # Save combined data
    if not args.no_save:
        print(f"\nSaving combined data to {args.output}...")
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)
        
        file_size = os.path.getsize(args.output)
        print(f"Combined data saved: {args.output} ({file_size:,} bytes)")
    else:
        print("\nSkipping save (--no-save flag specified)")


if __name__ == "__main__":
    main() 