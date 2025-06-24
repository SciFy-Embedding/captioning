#!/usr/bin/env python3
"""
Script to process markdown content from JSON files by adding figure captions.

This script:
1. Finds all result.json files with embedded markdown content
2. Looks for corresponding figure_analysis.json files with captions
3. Extracts markdown from the JSON files
4. Adds captions to the appropriate figure locations in markdown
5. Creates separate markdown files for each paper in the output directory
6. Does NOT modify original files - creates new output files
"""

import json
import os
import re
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from datetime import datetime


def find_result_json_files(base_dir: str = "extracted_pdfs_captioned") -> List[str]:
    """Find all result.json files recursively."""
    pattern = os.path.join(base_dir, "**", "result.json")
    files = glob.glob(pattern, recursive=True)
    return sorted(files)


def load_json_file(file_path: str) -> Optional[Dict]:
    """Load data from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def find_figure_analysis_file(paper_dir: str) -> Optional[str]:
    """Find figure_analysis.json file in the paper directory."""
    analysis_file = os.path.join(paper_dir, "figure_analysis.json")
    if os.path.exists(analysis_file):
        return analysis_file
    return None


def extract_markdown_from_json(json_data: Dict) -> Optional[str]:
    """Extract markdown content from JSON data."""
    try:
        # Try different possible paths for markdown content
        if 'marker' in json_data and 'markdown' in json_data['marker']:
            return json_data['marker']['markdown']
        elif 'markdown' in json_data:
            return json_data['markdown']
        elif 'content' in json_data:
            return json_data['content']
        else:
            print("  No markdown content found in expected fields")
            return None
    except Exception as e:
        print(f"  Error extracting markdown: {e}")
        return None


def extract_figure_references_from_markdown(content: str) -> List[Tuple[int, str, str, str]]:
    """Find figure references in markdown content.
    
    Returns list of (line_number, image_filename, figure_ref, full_line) tuples.
    """
    lines = content.split('\n')
    figure_refs = []
    
    for i, line in enumerate(lines):
        # Look for markdown image syntax: ![alt](filename)
        img_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        matches = re.finditer(img_pattern, line)
        
        for match in matches:
            alt_text = match.group(1)
            filename = match.group(2)
            figure_refs.append((i, filename, match.group(), line))
    
    return figure_refs


def create_captions_map(figure_analysis: Dict) -> Dict[str, str]:
    """Create a mapping from image filenames to captions."""
    captions_map = {}
    
    if 'figure_analysis' in figure_analysis:
        for image_name, data in figure_analysis['figure_analysis'].items():
            if isinstance(data, dict) and 'caption' in data and data['caption']:
                captions_map[image_name] = data['caption']
                # Also add without extension for matching
                name_without_ext = os.path.splitext(image_name)[0]
                captions_map[name_without_ext] = data['caption']
    
    return captions_map


def add_captions_to_markdown(content: str, captions_map: Dict[str, str]) -> Tuple[str, int]:
    """Add captions after figure references in markdown content.
    
    Returns tuple of (updated_content, num_captions_added).
    """
    if not captions_map:
        return content, 0
    
    lines = content.split('\n')
    figure_refs = extract_figure_references_from_markdown(content)
    
    if not figure_refs:
        return content, 0
    
    captions_added = 0
    offset = 0  # Track line insertions
    
    for line_num, filename, figure_ref, full_line in figure_refs:
        # Try to find matching caption
        caption = None
        
        # Try exact filename match first
        if filename in captions_map:
            caption = captions_map[filename]
        else:
            # Try filename without extension
            name_without_ext = os.path.splitext(filename)[0]
            if name_without_ext in captions_map:
                caption = captions_map[name_without_ext]
            else:
                # Try partial matches
                for caption_key in captions_map:
                    if (filename in caption_key or 
                        caption_key in filename or
                        name_without_ext in caption_key):
                        caption = captions_map[caption_key]
                        break
        
        # If we found a caption, insert it
        if caption:
            insert_line = line_num + offset + 1
            
            # Create formatted caption
            caption_lines = [
                "",  # Empty line before caption
                f"**Caption:** {caption}",
                ""   # Empty line after caption
            ]
            
            # Insert caption lines
            for i, caption_line in enumerate(caption_lines):
                lines.insert(insert_line + i, caption_line)
            
            offset += len(caption_lines)
            captions_added += 1
            print(f"    Added caption for {filename}")
    
    return '\n'.join(lines), captions_added


def get_paper_info(json_data: Dict, paper_dir: str) -> Dict[str, str]:
    """Extract paper information for the output."""
    info = {
        'id': json_data.get('id', 'unknown'),
        'pdf_path': json_data.get('pdf_path', 'unknown'),
        'paper_dir': os.path.basename(paper_dir),
        'status': 'unknown'
    }
    
    if 'marker' in json_data:
        info['status'] = json_data['marker'].get('status', 'unknown')
    
    return info


def create_paper_header(paper_info: Dict[str, str], captions_added: int) -> str:
    """Create a header for the markdown file."""
    header = f"""# {paper_info['paper_dir']}

**Paper ID:** {paper_info['id']}

**PDF Path:** {paper_info['pdf_path']}

**Processing Status:** {paper_info['status']}

**Captions Added:** {captions_added}

**Generated:** {datetime.now().isoformat()}

---

"""
    return header


def process_single_paper(result_json_path: str, output_dir: str) -> Optional[str]:
    """Process a single paper's JSON file and save as separate markdown file.
    
    Returns output file path if successful, None if failed.
    """
    paper_dir = os.path.dirname(result_json_path)
    paper_name = os.path.basename(paper_dir)
    
    print(f"  Processing {paper_name}...")
    
    # Load result.json
    json_data = load_json_file(result_json_path)
    if not json_data:
        return None
    
    # Extract markdown content
    markdown_content = extract_markdown_from_json(json_data)
    if not markdown_content:
        print(f"    No markdown content found")
        return None
    
    print(f"    Found markdown content ({len(markdown_content):,} characters)")
    
    # Look for figure analysis
    analysis_file = find_figure_analysis_file(paper_dir)
    captions_map = {}
    
    if analysis_file:
        figure_analysis = load_json_file(analysis_file)
        if figure_analysis:
            captions_map = create_captions_map(figure_analysis)
            print(f"    Found {len(captions_map)} captions")
    
    if not captions_map:
        print(f"    No captions available")
    
    # Add captions to markdown
    processed_markdown, captions_added = add_captions_to_markdown(markdown_content, captions_map)
    
    # Get paper info
    paper_info = get_paper_info(json_data, paper_dir)
    
    print(f"    Added {captions_added} captions")
    
    # Create output file path
    safe_paper_name = re.sub(r'[^\w\-_.]', '_', paper_name)
    output_file = os.path.join(output_dir, f"{safe_paper_name}.md")
    
    # Create the complete markdown content with header
    header = create_paper_header(paper_info, captions_added)
    complete_content = header + processed_markdown
    
    # Save to file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(complete_content)
        
        file_size = os.path.getsize(output_file)
        print(f"    Created: {output_file} ({file_size:,} bytes)")
        return output_file
        
    except Exception as e:
        print(f"    Error saving {output_file}: {e}")
        return None


def create_index_file(processed_files: List[Tuple[str, Dict[str, str], int]], output_dir: str):
    """Create an index file listing all processed papers."""
    
    total_papers = len(processed_files)
    total_captions = sum(captions for _, _, captions in processed_files)
    
    index_file = os.path.join(output_dir, "README.md")
    
    print(f"\nCreating index file: {index_file}")
    
    with open(index_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write("# Scientific Papers with Figure Captions - Index\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
        f.write(f"**Summary:**\n")
        f.write(f"- Papers processed: {total_papers}\n")
        f.write(f"- Total captions added: {total_captions}\n\n")
        f.write("This directory contains individual markdown files for each scientific paper with AI-generated figure captions.\n\n")
        f.write("## Papers\n\n")
        
        # Write paper list
        for i, (_, paper_info, captions_added) in enumerate(processed_files, 1):
            safe_paper_name = re.sub(r'[^\w\-_.]', '_', paper_info['paper_dir'])
            filename = f"{safe_paper_name}.md"
            f.write(f"{i}. [{paper_info['paper_dir']}]({filename}) - {captions_added} captions\n")
    
    print(f"Index file created: {index_file}")


def main():
    parser = argparse.ArgumentParser(description="Process JSON files with embedded markdown and add figure captions")
    parser.add_argument("--base-dir", default="extracted_pdfs_captioned", 
                       help="Base directory to search for result.json files")
    parser.add_argument("--output-dir", default="processed_papers_individual",
                       help="Output directory for individual markdown files")
    parser.add_argument("--skip-index", action="store_true",
                       help="Skip creating the index file")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all result.json files
    print("Finding result.json files...")
    result_files = find_result_json_files(args.base_dir)
    print(f"Found {len(result_files)} result.json files")
    
    if not result_files:
        print("No result.json files found!")
        return
    
    # Process each paper
    processed_papers = []
    successful_files = []
    
    print("\nProcessing papers...")
    for result_file in result_files:
        try:
            output_file = process_single_paper(result_file, args.output_dir)
            if output_file:
                # Get paper info for index
                paper_dir = os.path.dirname(result_file)
                json_data = load_json_file(result_file)
                if json_data:
                    paper_info = get_paper_info(json_data, paper_dir)
                    
                    # Count captions in the output file
                    with open(output_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        captions_added = content.count('**Caption:**')
                    
                    processed_papers.append((output_file, paper_info, captions_added))
                    successful_files.append(output_file)
        except Exception as e:
            paper_name = os.path.basename(os.path.dirname(result_file))
            print(f"  Error processing {paper_name}: {e}")
    
    print(f"\nSuccessfully processed {len(successful_files)} papers")
    print(f"Output directory: {args.output_dir}")
    
    if processed_papers and not args.skip_index:
        # Create index file
        create_index_file(processed_papers, args.output_dir)
    
    if successful_files:
        print(f"\nIndividual markdown files created:")
        for i, file_path in enumerate(successful_files[:10], 1):  # Show first 10
            filename = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            print(f"  {i}. {filename} ({file_size:,} bytes)")
        
        if len(successful_files) > 10:
            print(f"  ... and {len(successful_files) - 10} more files")
        
        print(f"\nProcessing complete!")
    else:
        print("No papers were successfully processed.")


if __name__ == "__main__":
    main() 