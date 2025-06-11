#!/usr/bin/env python3
"""
Comparison script for Chat Completions vs Assistants API approaches.

Usage:
    python compare_approaches.py --paper-path ./extracted_pdfs/arxiv/superconductors/arXiv:cond-mat_9909358
"""

import argparse
import json
import time
from openai_handler import OpenAIHandler
from assistants_handler import AssistantsHandler

def compare_approaches(paper_path: str):
    """Compare both approaches on the same paper."""
    print("=" * 80)
    print("COMPARING APPROACHES: Chat Completions vs Assistants API")
    print("=" * 80)
    
    # Initialize both handlers
    chat_handler = OpenAIHandler()
    assistants_handler = AssistantsHandler()
    
    try:
        # Method 1: Chat Completions (with smart truncation)
        print("\n🔄 METHOD 1: Chat Completions API (Smart Section Extraction)")
        print("-" * 60)
        
        start_time = time.time()
        
        # Load and truncate paper content
        from paper_processor import PaperProcessor
        processor = PaperProcessor(chat_handler)
        paper_content = processor.load_paper_markdown(paper_path)
        truncated_content = chat_handler.truncate_paper_content(paper_content)
        
        print(f"Original paper length: {len(paper_content):,} characters")
        print(f"Truncated length: {len(truncated_content):,} characters")
        print(f"Truncation ratio: {len(truncated_content)/len(paper_content):.1%}")
        
        # Process first image
        images = processor.find_images(paper_path)
        if images:
            image_path = images[0]
            print(f"Processing image: {image_path}")
            
            caption1, queries1 = chat_handler.process_figure_with_retry(
                image_path, truncated_content
            )
            
            chat_time = time.time() - start_time
            print(f"⏱️  Time taken: {chat_time:.1f} seconds")
            print(f"💰 Estimated cost: ~$0.01-0.02 (approx)")
            
            print("\n📝 RESULTS:")
            print(f"Caption: {caption1}")
            print("Queries:")
            for i, query in enumerate(queries1, 1):
                print(f"  {i}. {query}")
        
        # Method 2: Assistants API (full paper)
        print("\n\n🔄 METHOD 2: Assistants API (Full Paper Context)")
        print("-" * 60)
        
        start_time = time.time()
        
        # Process with Assistants API
        if images:
            # Create assistant and upload paper
            assistants_handler.create_assistant()
            file_id = assistants_handler.upload_paper_content(paper_path)
            thread_id = assistants_handler.create_thread_with_paper(file_id)
            
            print(f"Full paper length: {len(paper_content):,} characters")
            print(f"Paper uploaded as searchable file")
            print(f"Processing same image: {image_path}")
            
            caption2, queries2 = assistants_handler.process_figure(thread_id, image_path)
            
            assistants_time = time.time() - start_time
            print(f"⏱️  Time taken: {assistants_time:.1f} seconds")
            print(f"💰 Estimated cost: ~$0.05-0.10 (approx)")
            
            print("\n📝 RESULTS:")
            print(f"Caption: {caption2}")
            print("Queries:")
            for i, query in enumerate(queries2, 1):
                print(f"  {i}. {query}")
        
        # Comparison
        print("\n\n📊 COMPARISON SUMMARY")
        print("=" * 80)
        
        comparison_data = {
            "chat_completions": {
                "paper_content_used": len(truncated_content),
                "processing_time": chat_time,
                "caption": caption1,
                "queries": queries1,
                "estimated_cost": "$0.01-0.02"
            },
            "assistants_api": {
                "paper_content_used": len(paper_content),
                "processing_time": assistants_time,
                "caption": caption2,
                "queries": queries2,
                "estimated_cost": "$0.05-0.10"
            }
        }
        
        print(f"{'Metric':<25} {'Chat Completions':<20} {'Assistants API':<20}")
        print("-" * 65)
        print(f"{'Paper Content Used':<25} {len(truncated_content):,} chars{'':<8} {len(paper_content):,} chars")
        print(f"{'Processing Time':<25} {chat_time:.1f}s{'':<15} {assistants_time:.1f}s")
        print(f"{'Context Completeness':<25} {'Partial':<20} {'Full':<20}")
        print(f"{'Setup Complexity':<25} {'Simple':<20} {'Complex':<20}")
        print(f"{'API Stability':<25} {'Stable':<20} {'Beta':<20}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        if chat_time < assistants_time:
            print("✅ Chat Completions is faster for this case")
        else:
            print("✅ Assistants API provided more comprehensive analysis")
        
        if len(caption2) > len(caption1):
            print("✅ Assistants API generated more detailed caption")
        
        print("\n🔍 QUALITY ANALYSIS:")
        print("Compare the captions and queries above to see which approach")
        print("provides better context-aware analysis for your use case.")
        
        # Save comparison results
        with open(f"comparison_results_{int(time.time())}.json", "w") as f:
            json.dump(comparison_data, f, indent=2)
        print(f"\n💾 Detailed results saved to comparison_results_{int(time.time())}.json")
        
    except Exception as e:
        print(f"❌ Error during comparison: {e}")
    
    finally:
        # Cleanup
        try:
            assistants_handler.cleanup_files()
            print("\n🧹 Cleaned up uploaded files")
        except:
            pass

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare Chat Completions vs Assistants API approaches"
    )
    parser.add_argument(
        "--paper-path", 
        required=True,
        help="Path to paper directory (e.g., ./extracted_pdfs/arxiv/superconductors/arXiv:cond-mat_9909358)"
    )
    
    args = parser.parse_args()
    
    compare_approaches(args.paper_path)

if __name__ == "__main__":
    main() 