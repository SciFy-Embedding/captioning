#!/usr/bin/env python3
"""
Figure Caption and Query Generator

This script processes scientific papers to generate captions and search queries 
for figures using OpenAI's vision models.

Usage:
    python main.py --help
    python main.py process --path ./extracted_pdfs
    python main.py stats --path ./extracted_pdfs
    python main.py process-single --path ./extracted_pdfs/arxiv/superconductors/arXiv:cond-mat_9909358
"""

import argparse
import json
import os
import sys
from pathlib import Path
from batch_processor import BatchProcessor
from config import OPENAI_CONFIG, PROCESSING_CONFIG

def check_environment():
    """Check if all required environment variables and dependencies are set."""
    issues = []
    
    # Check OpenAI API key
    if not OPENAI_CONFIG["api_key"]:
        issues.append("OPENAI_API_KEY environment variable not set")
    
    # Check required packages
    try:
        import openai
    except ImportError:
        issues.append("openai package not installed. Run: pip install openai")
    
    if issues:
        print("Environment issues found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nPlease fix these issues before running the script.")
        return False
    
    return True

def process_command(args):
    """Process papers in a directory."""
    try:
        processor = BatchProcessor(method=args.method)
        
        # Find all papers
        paper_paths = processor.find_papers(args.path)
        
        if not paper_paths:
            print(f"No papers found in {args.path}")
            return
        
        print(f"Found {len(paper_paths)} papers")
        
        # Get processing settings from config and args
        concurrent_papers = PROCESSING_CONFIG.get("concurrent_papers", True)
        max_paper_workers = PROCESSING_CONFIG.get("max_paper_workers", 3)
        generate_queries = not args.no_queries  # Invert the no_queries flag
        
        # Process papers with concurrent option
        results = processor.process_papers(
            paper_paths,
            force_reprocess=args.force,
            include_processed=args.include_processed,
            save_results=args.save_results is not None,
            results_path=args.save_results,
            concurrent_papers=concurrent_papers,
            max_paper_workers=max_paper_workers,
            generate_queries=generate_queries
        )
        
        # Print summary
        processor.print_batch_summary(results)
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()

def process_single_paper(args):
    """Process a single paper."""
    print(f"Processing single paper: {args.path}")
    print(f"Method: {args.method}")
    print(f"Force reprocess: {args.force}")
    print(f"Generate queries: {not args.no_queries}")
    print("-" * 50)
    
    try:
        processor = BatchProcessor(method=args.method)
        generate_queries = not args.no_queries  # Invert the no_queries flag
        success = processor.paper_processor.process_single_paper(
            paper_path=args.path,
            force_reprocess=args.force,
            generate_queries=generate_queries
        )
        
        if success:
            print("✓ Paper processed successfully")
            
            # Show summary
            summary = processor.paper_processor.get_processing_summary(args.path)
            print(f"\nSummary:")
            print(f"  Total figures: {summary.get('total_figures', 0)}")
            print(f"  Successful: {summary.get('successful_figures', 0)}")
            print(f"  Failed: {summary.get('failed_figures', 0)}")
            print(f"  Success rate: {summary.get('success_rate', 0):.1%}")
        else:
            print("✗ Failed to process paper")
        
        return success
        
    except Exception as e:
        print(f"Error processing paper: {e}")
        return False

def show_stats(args):
    """Show processing statistics."""
    print(f"Getting statistics for: {args.path}")
    print("-" * 50)
    
    try:
        processor = BatchProcessor()
        stats = processor.get_processing_stats(args.path)
        
        print(f"Papers found: {stats['total_papers_found']}")
        print(f"Processed papers: {stats['processed_papers']}")
        print(f"Unprocessed papers: {stats['unprocessed_papers']}")
        print(f"Papers with errors: {stats['papers_with_errors']}")
        print()
        print(f"Total figures: {stats['total_figures']}")
        print(f"Successful figures: {stats['total_successful_figures']}")
        print(f"Failed figures: {stats['total_failed_figures']}")
        
        if stats['total_figures'] > 0:
            figure_success_rate = (stats['total_successful_figures'] / stats['total_figures']) * 100
            print(f"Figure success rate: {figure_success_rate:.1f}%")
        
        if stats['total_papers_found'] > 0:
            paper_completion_rate = (stats['processed_papers'] / stats['total_papers_found']) * 100
            print(f"Paper completion rate: {paper_completion_rate:.1f}%")
        
        return stats
        
    except Exception as e:
        print(f"Error getting statistics: {e}")
        return None

def list_papers(args):
    """List all papers in the specified path."""
    print(f"Listing papers in: {args.path}")
    print("-" * 50)
    
    try:
        processor = BatchProcessor()
        paper_paths = processor.find_papers(args.path)
        
        if not paper_paths:
            print("No papers found.")
            return []
        
        for i, paper_path in enumerate(paper_paths, 1):
            # Get relative path for cleaner display
            rel_path = os.path.relpath(paper_path, args.path)
            
            # Check if processed
            try:
                paper_data = processor.paper_processor.load_paper_data(paper_path)
                processed = "✓" if "figure_analysis" in paper_data else "✗"
            except:
                processed = "?"
            
            print(f"{i:3d}. {processed} {rel_path}")
        
        print(f"\nTotal: {len(paper_paths)} papers")
        print("✓ = processed, ✗ = not processed, ? = error checking status")
        
        return paper_paths
        
    except Exception as e:
        print(f"Error listing papers: {e}")
        return []

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate captions and queries for figures in scientific papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast processing with smart paper extraction (default)
  python main.py process --path ./extracted_pdfs
  python main.py process-single --path ./extracted_pdfs/arxiv/superconductors/arXiv:cond-mat_9909358
  
  # Slower but complete paper access via Assistants API
  python main.py process --path ./extracted_pdfs --method assistants
  python main.py process-single --path ./extracted_pdfs/arxiv/superconductors/arXiv:cond-mat_9909358 --method assistants
  
  # Other options
  python main.py process --path ./extracted_pdfs --force --method chat
  python main.py stats --path ./extracted_pdfs
  python main.py list --path ./extracted_pdfs
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process all papers command
    process_parser = subparsers.add_parser('process', help='Process all papers in a directory')
    process_parser.add_argument('--path', required=True, help='Path to the extracted_pdfs directory')
    process_parser.add_argument('--method', choices=['chat', 'assistants'], default='chat', 
                              help='Processing method: "chat" (fast, smart extraction) or "assistants" (slow, complete paper access)')
    process_parser.add_argument('--force', action='store_true', help='Force reprocessing of already processed papers')
    process_parser.add_argument('--include-processed', action='store_true', help='Include already processed papers')
    process_parser.add_argument('--save-results', help='Save processing results to JSON file')
    process_parser.add_argument('--no-queries', action='store_true', help='Generate only captions, skip queries (faster and cheaper)')
    
    # Process single paper command
    single_parser = subparsers.add_parser('process-single', help='Process a single paper')
    single_parser.add_argument('--path', required=True, help='Path to the paper directory')
    single_parser.add_argument('--method', choices=['chat', 'assistants'], default='chat',
                             help='Processing method: "chat" (fast, smart extraction) or "assistants" (slow, complete paper access)')
    single_parser.add_argument('--force', action='store_true', help='Force reprocessing if already processed')
    single_parser.add_argument('--no-queries', action='store_true', help='Generate only captions, skip queries (faster and cheaper)')
    
    # Show statistics command
    stats_parser = subparsers.add_parser('stats', help='Show processing statistics')
    stats_parser.add_argument('--path', required=True, help='Path to the extracted_pdfs directory')
    
    # List papers command
    list_parser = subparsers.add_parser('list', help='List all papers in directory')
    list_parser.add_argument('--path', required=True, help='Path to the extracted_pdfs directory')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Check environment
    if not check_environment():
        return 1
    
    # Validate path
    if not os.path.exists(args.path):
        print(f"Error: Path does not exist: {args.path}")
        return 1
    
    # Execute command
    try:
        if args.command == 'process':
            process_command(args)
            return 0
        elif args.command == 'process-single':
            result = process_single_paper(args)
            return 0 if result else 1
        elif args.command == 'stats':
            result = show_stats(args)
            return 0 if result else 1
        elif args.command == 'list':
            result = list_papers(args)
            return 0 if result else 1
        else:
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 