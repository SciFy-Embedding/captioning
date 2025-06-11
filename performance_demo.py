#!/usr/bin/env python3
"""
Performance demonstration for Assistants API optimizations.
Shows the impact of different performance settings.
"""

import time
import os
import sys
from assistants_handler import AssistantsHandler
from config import PROCESSING_CONFIG

def test_performance_settings(paper_path: str):
    """Test different performance configurations."""
    
    if not os.path.exists(paper_path):
        print(f"❌ Paper path not found: {paper_path}")
        return
    
    print("🚀 Assistants API Performance Test")
    print("=" * 50)
    
    # Test configurations
    test_configs = [
        {
            "name": "🐌 Sequential (Original)",
            "parallel": False,
            "max_workers": 1,
            "description": "Process figures one by one (slowest)"
        },
        {
            "name": "⚡ Parallel - 2 Workers",
            "parallel": True,
            "max_workers": 2,
            "description": "Process 2 figures simultaneously"
        },
        {
            "name": "🚀 Parallel - 3 Workers",
            "parallel": True,
            "max_workers": 3,
            "description": "Process 3 figures simultaneously (fastest)"
        }
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n{config['name']}")
        print(f"Description: {config['description']}")
        print("-" * 30)
        
        # Create handler
        handler = AssistantsHandler()
        
        try:
            start_time = time.time()
            
            # Process with specific configuration
            figure_analysis = handler.process_paper_with_assistants(
                paper_path, 
                parallel=config["parallel"], 
                max_workers=config["max_workers"]
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Get usage summary
            usage = handler.get_usage_summary()
            
            print(f"⏱️  Processing Time: {processing_time:.1f} seconds")
            print(f"💰 Total Cost: ${usage.get('total_cost', 0):.3f}")
            print(f"📊 Figures Processed: {len(figure_analysis)}")
            
            if len(figure_analysis) > 0:
                avg_time = processing_time / len(figure_analysis)
                print(f"📈 Avg Time per Figure: {avg_time:.1f} seconds")
            
            results.append({
                "config": config["name"],
                "time": processing_time,
                "cost": usage.get('total_cost', 0),
                "figures": len(figure_analysis),
                "parallel": config["parallel"],
                "workers": config["max_workers"]
            })
            
            # Cleanup
            handler.cleanup_files()
            
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    # Performance summary
    if len(results) > 1:
        print("\n" + "=" * 50)
        print("📊 PERFORMANCE COMPARISON")
        print("=" * 50)
        
        baseline = results[0]  # Sequential processing
        
        for result in results[1:]:
            speedup = baseline["time"] / result["time"]
            cost_diff = result["cost"] - baseline["cost"]
            
            print(f"\n{result['config']} vs Sequential:")
            print(f"  ⚡ Speedup: {speedup:.1f}x faster")
            print(f"  💰 Cost Difference: ${cost_diff:+.3f}")
            print(f"  👥 Workers: {result['workers']}")

def main():
    """Main function to run performance test."""
    if len(sys.argv) != 2:
        print("Usage: python performance_demo.py <paper_path>")
        print("Example: python performance_demo.py ./extracted_pdfs/arxiv/sample_paper")
        return
    
    paper_path = sys.argv[1]
    
    print(f"Testing with paper: {paper_path}")
    print(f"Current config parallel: {PROCESSING_CONFIG.get('assistants_parallel_processing', True)}")
    print(f"Current config workers: {PROCESSING_CONFIG.get('assistants_max_workers', 3)}")
    
    test_performance_settings(paper_path)

if __name__ == "__main__":
    main() 