import os
import time
import logging
import concurrent.futures
import json
from typing import List, Dict, Any, Union, Optional
from pathlib import Path
from paper_processor import PaperProcessor
from openai_handler import OpenAIHandler
from assistants_handler import AssistantsHandler
from config import PROCESSING_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatchProcessor:
    """Processes multiple papers in batches to generate figure captions and queries."""
    
    def __init__(self, openai_api_key: Optional[str] = None, use_assistants: bool = False, method: str = 'chat'):
        """Initialize batch processor.
        
        Args:
            openai_api_key: OpenAI API key (optional, will use env var if not provided)
            use_assistants: Whether to use Assistants API (deprecated, use method instead)
            method: Processing method - 'chat' or 'assistants'
        """
        # Support both old and new initialization methods
        if method == 'assistants':
            self.use_assistants = True
        elif method == 'chat':
            self.use_assistants = False
        else:
            self.use_assistants = use_assistants  # Fallback to old parameter
        
        # Create handler based on method
        if self.use_assistants:
            logger.info("Using Assistants API (slower, complete paper access)")
            self.handler = AssistantsHandler(openai_api_key)
        else:
            logger.info("Using Chat Completions API (faster, limited paper access)")
            self.handler = OpenAIHandler(openai_api_key)
        
        self.paper_processor = PaperProcessor(self.handler, use_assistants=self.use_assistants)
        self.batch_size = PROCESSING_CONFIG["batch_size"]
    
    def find_papers(self, base_path: str) -> List[str]:
        """Find all paper directories containing result.json files."""
        paper_paths = []
        
        try:
            # Walk through all subdirectories
            for root, dirs, files in os.walk(base_path):
                if "result.json" in files and "images" in dirs:
                    images_dir = os.path.join(root, "images")
                    # Check if images directory has any image files
                    if os.path.exists(images_dir) and os.listdir(images_dir):
                        paper_paths.append(root)
            
            logger.info(f"Found {len(paper_paths)} papers to process")
            return sorted(paper_paths)
            
        except Exception as e:
            logger.error(f"Error finding papers in {base_path}: {e}")
            return []
    
    def filter_unprocessed_papers(self, paper_paths: List[str]) -> List[str]:
        """Filter out papers that have already been processed."""
        unprocessed = []
        
        for paper_path in paper_paths:
            try:
                # Use the optimized check method instead of loading full data
                if not self.paper_processor.check_if_processed(paper_path):
                    unprocessed.append(paper_path)
                else:
                    logger.info(f"Skipping already processed paper: {paper_path}")
            except Exception as e:
                logger.warning(f"Error checking processing status for {paper_path}: {e}")
                unprocessed.append(paper_path)  # Include in case of error
        
        logger.info(f"Found {len(unprocessed)} unprocessed papers")
        return unprocessed
    
    def process_papers(self, paper_paths: List[str], force_reprocess: bool = False, 
                      include_processed: bool = False, save_results: bool = False, 
                      results_path: str = None, concurrent_papers: bool = True, 
                      max_paper_workers: int = 3, generate_queries: bool = None) -> Dict[str, Any]:
        """Process multiple papers with optional concurrent processing.
        
        Args:
            paper_paths: List of paths to papers to process
            force_reprocess: Whether to reprocess already processed papers
            include_processed: Whether to include already processed papers in batch
            save_results: Whether to save batch results to file
            results_path: Path to save results file
            concurrent_papers: Whether to process papers concurrently
            max_paper_workers: Maximum number of papers to process simultaneously
            generate_queries: Whether to generate search queries (None uses config default)
        """
        # Use config default if not specified
        if generate_queries is None:
            generate_queries = PROCESSING_CONFIG.get("generate_queries", True)
            
        logger.info(f"Starting batch processing of {len(paper_paths)} papers")
        logger.info(f"Method: {'Assistants API' if self.use_assistants else 'Chat Completions'}")
        logger.info(f"Concurrent papers: {concurrent_papers} (max_workers: {max_paper_workers})")
        logger.info(f"Generate queries: {generate_queries}")
        
        start_time = time.time()
        
        # Filter papers based on processing status
        papers_to_process = []
        for paper_path in paper_paths:
            processor = PaperProcessor(self.handler, self.use_assistants)
            
            if not include_processed and processor.check_if_processed(paper_path) and not force_reprocess:
                logger.info(f"Skipping already processed paper: {paper_path}")
                continue
            
            papers_to_process.append(paper_path)
        
        logger.info(f"Processing {len(papers_to_process)} papers (skipped {len(paper_paths) - len(papers_to_process)})")
        
        if not papers_to_process:
            logger.info("No papers to process")
            return {
                "processed": 0, 
                "successful": 0, 
                "failed": 0, 
                "success_rate": 0.0,
                "duration": 0.0,
                "papers": []
            }
        
        results = []
        
        if concurrent_papers and len(papers_to_process) > 1:
            # Process papers concurrently
            results = self._process_papers_concurrently(
                papers_to_process, force_reprocess, max_paper_workers, generate_queries
            )
        else:
            # Process papers sequentially
            results = self._process_papers_sequentially(papers_to_process, force_reprocess, generate_queries)
        
        # Calculate summary statistics
        total_processed = len(results)
        successful = len([r for r in results if r.get("success", False)])
        failed = total_processed - successful
        
        end_time = time.time()
        duration = end_time - start_time
        
        batch_summary = {
            "processed": total_processed,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total_processed if total_processed > 0 else 0,
            "duration": duration,
            "papers": results
        }
        
        # Save results if requested
        if save_results and results_path:
            self._save_batch_results(batch_summary, results_path)
        
        logger.info(f"Batch processing completed in {duration:.1f}s")
        logger.info(f"Success rate: {successful}/{total_processed} ({batch_summary['success_rate']:.1%})")
        
        return batch_summary
    
    def _process_papers_concurrently(self, paper_paths: List[str], force_reprocess: bool, 
                                   max_workers: int, generate_queries: bool = True) -> List[Dict[str, Any]]:
        """Process papers concurrently using ThreadPoolExecutor."""
        logger.info(f"Processing {len(paper_paths)} papers concurrently (max_workers={max_workers})")
        
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all paper processing tasks
            future_to_paper = {
                executor.submit(self._process_single_paper_for_concurrent, paper_path, force_reprocess, generate_queries): 
                paper_path for paper_path in paper_paths
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_paper):
                paper_path = future_to_paper[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    status = "✅ SUCCESS" if result.get("success", False) else "❌ FAILED"
                    logger.info(f"{status}: {paper_path}")
                    
                except Exception as e:
                    logger.error(f"❌ FAILED: {paper_path} - {e}")
                    results.append({
                        "paper_path": paper_path,
                        "success": False,
                        "error": str(e),
                        "figures_processed": 0
                    })
        
        return results
    
    def _process_single_paper_for_concurrent(self, paper_path: str, force_reprocess: bool, generate_queries: bool = True) -> Dict[str, Any]:
        """Process a single paper in a concurrent context (each gets its own handler instance)."""
        try:
            # Create a fresh handler instance for this paper to avoid conflicts
            if self.use_assistants:
                handler = AssistantsHandler()
            else:
                handler = OpenAIHandler()
            
            # Create processor with the fresh handler
            processor = PaperProcessor(handler, self.use_assistants)
            
            # Process the paper
            success = processor.process_single_paper(paper_path, force_reprocess, generate_queries)
            
            # Get processing summary
            summary = processor.get_processing_summary(paper_path)
            
            return {
                "paper_path": paper_path,
                "success": success,
                "figures_processed": summary.get("successful_figures", 0),
                "total_figures": summary.get("total_figures", 0),
                "processing_time": summary.get("processing_timestamp", "")
            }
            
        except Exception as e:
            return {
                "paper_path": paper_path,
                "success": False,
                "error": str(e),
                "figures_processed": 0
            }
    
    def _process_papers_sequentially(self, paper_paths: List[str], force_reprocess: bool, generate_queries: bool = True) -> List[Dict[str, Any]]:
        """Process papers sequentially (original method)."""
        logger.info(f"Processing {len(paper_paths)} papers sequentially")
        
        results = []
        for i, paper_path in enumerate(paper_paths, 1):
            logger.info(f"Processing paper {i}/{len(paper_paths)}: {paper_path}")
            
            try:
                processor = PaperProcessor(self.handler, self.use_assistants)
                success = processor.process_single_paper(paper_path, force_reprocess, generate_queries)
                
                summary = processor.get_processing_summary(paper_path)
                
                result = {
                    "paper_path": paper_path,
                    "success": success,
                    "figures_processed": summary.get("successful_figures", 0),
                    "total_figures": summary.get("total_figures", 0),
                    "processing_time": summary.get("processing_timestamp", "")
                }
                
                results.append(result)
                
                status = "✅ SUCCESS" if success else "❌ FAILED"
                logger.info(f"{status}: {paper_path}")
                
            except Exception as e:
                logger.error(f"❌ FAILED: {paper_path} - {e}")
                results.append({
                    "paper_path": paper_path,
                    "success": False,
                    "error": str(e),
                    "figures_processed": 0
                })
        
        return results
    
    def process_all_papers(self, base_path: str, force_reprocess: bool = False, 
                          filter_processed: bool = True) -> Dict[str, Any]:
        """Process all papers in the base path."""
        logger.info(f"Starting batch processing of papers in: {base_path}")
        
        # Find all papers
        paper_paths = self.find_papers(base_path)
        
        if not paper_paths:
            logger.warning("No papers found to process")
            return {
                "total_papers": 0,
                "successful_papers": 0,
                "failed_papers": 0,
                "paper_results": [],
                "message": "No papers found"
            }
        
        # Filter out already processed papers if requested
        if filter_processed and not force_reprocess:
            paper_paths = self.filter_unprocessed_papers(paper_paths)
            
            if not paper_paths:
                logger.info("All papers have already been processed")
                return {
                    "total_papers": 0,
                    "successful_papers": 0,
                    "failed_papers": 0,
                    "paper_results": [],
                    "message": "All papers already processed"
                }
        
        # Process papers in batches
        all_results = {
            "total_papers": len(paper_paths),
            "successful_papers": 0,
            "failed_papers": 0,
            "paper_results": [],
            "batch_results": [],
            "start_time": time.time()
        }
        
        # Split into batches
        batches = [paper_paths[i:i + self.batch_size] 
                  for i in range(0, len(paper_paths), self.batch_size)]
        
        logger.info(f"Processing {len(paper_paths)} papers in {len(batches)} batches")
        
        for batch_num, batch_papers in enumerate(batches, 1):
            logger.info(f"Processing batch {batch_num}/{len(batches)} ({len(batch_papers)} papers)")
            
            batch_results = self.process_papers(batch_papers, force_reprocess)
            
            # Accumulate results
            all_results["successful_papers"] += batch_results["successful"]
            all_results["failed_papers"] += batch_results["failed"]
            all_results["paper_results"].extend(batch_results["papers"])
            all_results["batch_results"].append({
                "batch_number": batch_num,
                "papers_in_batch": len(batch_papers),
                "successful": batch_results["successful"],
                "failed": batch_results["failed"],
                "duration": batch_results["duration"]
            })
            
            logger.info(f"Batch {batch_num} completed: {batch_results['successful']}/{len(batch_papers)} successful")
        
        all_results["end_time"] = time.time()
        all_results["total_duration"] = all_results["end_time"] - all_results["start_time"]
        
        # Print summary
        self.print_summary(all_results)
        
        return all_results
    
    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print a summary of processing results."""
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        print(f"Total papers processed: {results['total_papers']}")
        print(f"Successful: {results['successful_papers']}")
        print(f"Failed: {results['failed_papers']}")
        if results['total_papers'] > 0:
            success_rate = (results['successful_papers'] / results['total_papers']) * 100
            print(f"Success rate: {success_rate:.1f}%")
        print(f"Total duration: {results.get('total_duration', 0):.1f} seconds")
        
        if "batch_results" in results:
            print(f"Number of batches: {len(results['batch_results'])}")
        
        # Show failed papers if any
        failed_papers = [r for r in results['paper_results'] if not r.get('success', False)]
        if failed_papers:
            print(f"\nFailed papers ({len(failed_papers)}):")
            for paper in failed_papers[:10]:  # Show first 10
                print(f"  - {paper['paper_path']}")
                if 'error' in paper:
                    print(f"    Error: {paper['error']}")
            if len(failed_papers) > 10:
                print(f"  ... and {len(failed_papers) - 10} more")
        
        print("="*60)
    
    def get_processing_stats(self, base_path: str) -> Dict[str, Any]:
        """Get statistics about processed papers without reprocessing."""
        paper_paths = self.find_papers(base_path)
        
        stats = {
            "total_papers_found": len(paper_paths),
            "processed_papers": 0,
            "unprocessed_papers": 0,
            "total_figures": 0,
            "total_successful_figures": 0,
            "total_failed_figures": 0,
            "papers_with_errors": 0
        }
        
        for paper_path in paper_paths:
            try:
                summary = self.paper_processor.get_processing_summary(paper_path)
                
                if summary.get("success", False):
                    stats["processed_papers"] += 1
                    stats["total_figures"] += summary.get("total_figures", 0)
                    stats["total_successful_figures"] += summary.get("successful_figures", 0)
                    stats["total_failed_figures"] += summary.get("failed_figures", 0)
                    
                    if summary.get("failed_figures", 0) > 0:
                        stats["papers_with_errors"] += 1
                else:
                    stats["unprocessed_papers"] += 1
                    
            except Exception as e:
                logger.warning(f"Error getting stats for {paper_path}: {e}")
                stats["unprocessed_papers"] += 1
        
        return stats 
    
    def print_batch_summary(self, results: Dict[str, Any]):
        """Print a summary of batch processing results."""
        print("\n" + "=" * 60)
        print("PROCESSING SUMMARY")
        print("=" * 60)
        
        processed = results.get('processed', 0)
        successful = results.get('successful', 0)
        failed = results.get('failed', 0)
        
        print(f"Total papers processed: {processed}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        
        # Calculate success rate if not present
        if 'success_rate' in results:
            success_rate = results['success_rate']
        else:
            success_rate = successful / processed if processed > 0 else 0
            
        print(f"Success rate: {success_rate:.1%}")
        
        duration = results.get('duration', 0)
        print(f"Total duration: {duration:.0f} seconds")
        print("=" * 60)
        
        # Show failed papers if any
        papers = results.get('papers', [])
        failed_papers = [r for r in papers if not r.get('success', False)]
        if failed_papers:
            print(f"\nFailed papers ({len(failed_papers)}):")
            for paper in failed_papers:
                error = paper.get('error', 'Unknown error')
                print(f"  ❌ {paper['paper_path']}: {error}")
        
        # Show successful papers summary
        successful_papers = [r for r in papers if r.get('success', False)]
        if successful_papers:
            total_figures = sum(p.get('figures_processed', 0) for p in successful_papers)
            print(f"\n✅ Successfully processed {len(successful_papers)} papers with {total_figures} figures total")
    
    def _save_batch_results(self, results: Dict[str, Any], results_path: str):
        """Save batch results to file."""
        try:
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Batch results saved to: {results_path}")
        except Exception as e:
            logger.error(f"Error saving batch results: {e}") 