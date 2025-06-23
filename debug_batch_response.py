#!/usr/bin/env python3
"""
Debug script to test batch response parsing and reprocess failed papers.

This script helps diagnose and fix issues with "Failed to parse batch response" errors.
"""

import json
import os
import sys
from assistants_handler import AssistantsHandler
from config import PROCESSING_CONFIG
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test_paper_reprocessing(paper_path: str):
    """Test reprocessing a paper that previously failed."""
    
    print(f"Testing reprocessing of: {paper_path}")
    
    # Check if paper directory exists
    if not os.path.exists(paper_path):
        print(f"Error: Paper directory not found: {paper_path}")
        return
    
    # Check if images directory exists
    images_dir = os.path.join(paper_path, "images")
    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found: {images_dir}")
        return
    
    # Count images
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
    print(f"Found {len(image_files)} images in {images_dir}")
    
    if not image_files:
        print("No images to process!")
        return
    
    # Initialize handler
    handler = AssistantsHandler()
    
    try:
        print("\nStarting reprocessing...")
        results = handler.process_paper_with_assistants(paper_path, generate_queries=True)
        
        print(f"\nProcessing completed!")
        print(f"Results: {len(results)} figures processed")
        
        # Show results summary
        success_count = 0
        error_count = 0
        
        for figure_name, figure_data in results.items():
            if 'error' in figure_data:
                error_count += 1
                print(f"  ❌ {figure_name}: {figure_data['error']}")
            else:
                success_count += 1
                caption_preview = figure_data.get('caption', '')[:100] + "..." if len(figure_data.get('caption', '')) > 100 else figure_data.get('caption', '')
                query_count = len(figure_data.get('queries', []))
                print(f"  ✅ {figure_name}: {query_count} queries, caption: {caption_preview}")
        
        print(f"\nSummary: {success_count} successful, {error_count} errors")
        
        # Get usage summary
        usage = handler.get_usage_summary()
        print(f"\nUsage: ${usage['total_cost']:.4f} total cost, {usage['total_time']:.1f}s")
        
        # Save results if successful
        if success_count > 0:
            output_file = os.path.join(paper_path, "figure_analysis_reprocessed.json")
            
            paper_data = {
                "paper_id": os.path.basename(paper_path),
                "pdf_path": paper_path,
                "processing_timestamp": "reprocessed_debug",
                "figure_analysis": results,
                "usage_summary": usage
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(paper_data, f, indent=2, ensure_ascii=False)
            
            print(f"Saved reprocessed results to: {output_file}")
        
    except Exception as e:
        print(f"Error during reprocessing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        try:
            handler.cleanup_files()
        except:
            pass


def show_failed_papers():
    """Show all papers with failed batch responses."""
    
    failed_papers = []
    
    print("Searching for papers with failed batch responses...")
    
    for root, dirs, files in os.walk("extracted_pdfs"):
        if "figure_analysis.json" in files:
            analysis_file = os.path.join(root, "figure_analysis.json")
            
            try:
                with open(analysis_file, 'r') as f:
                    data = json.load(f)
                
                figure_analysis = data.get("figure_analysis", {})
                
                # Check for failed parsing
                failed_figures = [name for name, info in figure_analysis.items() 
                                if info.get("error") == "Failed to parse batch response"]
                
                if failed_figures:
                    failed_papers.append({
                        "paper_path": root,
                        "analysis_file": analysis_file,
                        "failed_count": len(failed_figures),
                        "total_figures": len(figure_analysis)
                    })
                    
            except Exception as e:
                print(f"Error reading {analysis_file}: {e}")
    
    if not failed_papers:
        print("No papers with failed batch responses found!")
        return
    
    print(f"\nFound {len(failed_papers)} papers with failed batch responses:")
    for i, paper in enumerate(failed_papers, 1):
        print(f"{i}. {paper['paper_path']}")
        print(f"   Failed: {paper['failed_count']}/{paper['total_figures']} figures")
    
    return failed_papers


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python debug_batch_response.py show                    # Show all failed papers")
        print("  python debug_batch_response.py reprocess <paper_path>  # Reprocess a specific paper")
        return
    
    command = sys.argv[1]
    
    if command == "show":
        show_failed_papers()
    
    elif command == "reprocess":
        if len(sys.argv) < 3:
            print("Error: Please provide paper path")
            print("Usage: python debug_batch_response.py reprocess <paper_path>")
            return
        
        paper_path = sys.argv[2]
        test_paper_reprocessing(paper_path)
    
    else:
        print(f"Unknown command: {command}")
        print("Use 'show' or 'reprocess'")


if __name__ == "__main__":
    main() 