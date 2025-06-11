import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
from openai_handler import OpenAIHandler
from assistants_handler import AssistantsHandler
from config import PROCESSING_CONFIG, OUTPUT_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaperProcessor:
    """Processes individual papers to generate figure captions and queries."""
    
    def __init__(self, handler: Union[OpenAIHandler, AssistantsHandler], use_assistants: bool = False):
        """Initialize with either OpenAI or Assistants handler.
        
        Args:
            handler: Either OpenAIHandler or AssistantsHandler instance
            use_assistants: Whether using Assistants API (for processing method tracking)
        """
        self.handler = handler
        self.use_assistants = use_assistants
        self.supported_formats = PROCESSING_CONFIG["supported_image_formats"]
    
    def load_paper_markdown(self, paper_path: str) -> str:
        """Load only the markdown content from paper JSON file."""
        json_path = os.path.join(paper_path, "result.json")
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"No result.json found in {paper_path}")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract markdown content from marker field
            marker = data.get("marker", {})
            if isinstance(marker, dict):
                markdown_content = marker.get("markdown", "")
            else:
                markdown_content = str(marker)
            
            if not markdown_content:
                logger.warning(f"No markdown content found in {json_path}")
                return ""
            
            return markdown_content
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {json_path}: {e}")
        except Exception as e:
            raise Exception(f"Error loading markdown from {json_path}: {e}")
    
    def get_figure_analysis_path(self, paper_path: str) -> str:
        """Get the path for the figure analysis JSON file."""
        return os.path.join(paper_path, "figure_analysis.json")
    
    def save_figure_analysis(self, paper_path: str, figure_analysis: Dict[str, Any]) -> None:
        """Save figure analysis data to separate JSON file."""
        analysis_path = self.get_figure_analysis_path(paper_path)
        
        # Load basic paper info from original result.json
        try:
            original_json_path = os.path.join(paper_path, "result.json")
            with open(original_json_path, 'r', encoding='utf-8') as f:
                original_data = json.load(f)
            
            # Create analysis data with basic paper info + figure analysis
            analysis_data = {
                "paper_id": original_data.get("id", ""),
                "pdf_path": original_data.get("pdf_path", ""),
                "processing_timestamp": None,  # Will be added by caller
                "figure_analysis": figure_analysis
            }
        except Exception as e:
            logger.warning(f"Could not load original paper data: {e}")
            # Fallback: just save the figure analysis
            analysis_data = {
                "paper_id": "",
                "pdf_path": "",
                "processing_timestamp": None,
                "figure_analysis": figure_analysis
            }
        
        # Add timestamp
        from datetime import datetime
        analysis_data["processing_timestamp"] = datetime.now().isoformat()
        
        try:
            with open(analysis_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, indent=OUTPUT_CONFIG["indent_json"], ensure_ascii=False)
            logger.info(f"Saved figure analysis to: {analysis_path}")
        except Exception as e:
            raise Exception(f"Error saving figure analysis to {analysis_path}: {e}")
    
    def load_figure_analysis(self, paper_path: str) -> Dict[str, Any]:
        """Load figure analysis data from separate JSON file."""
        analysis_path = self.get_figure_analysis_path(paper_path)
        
        if not os.path.exists(analysis_path):
            return {}
        
        try:
            with open(analysis_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading figure analysis from {analysis_path}: {e}")
            return {}
    
    def check_if_processed(self, paper_path: str) -> bool:
        """Check if paper has already been processed by looking for figure_analysis.json."""
        analysis_path = self.get_figure_analysis_path(paper_path)
        return os.path.exists(analysis_path)

    def find_images(self, paper_path: str) -> List[str]:
        """Find all supported image files in the images directory."""
        images_dir = os.path.join(paper_path, "images")
        
        if not os.path.exists(images_dir):
            logger.warning(f"Images directory not found: {images_dir}")
            return []
        
        image_files = []
        try:
            for file in os.listdir(images_dir):
                file_path = os.path.join(images_dir, file)
                if os.path.isfile(file_path):
                    _, ext = os.path.splitext(file.lower())
                    if ext in self.supported_formats:
                        image_files.append(file_path)
            
            image_files.sort()  # Sort for consistent processing order
            logger.info(f"Found {len(image_files)} image files in {images_dir}")
            return image_files
            
        except Exception as e:
            logger.error(f"Error scanning images directory {images_dir}: {e}")
            return []
    
    def process_single_paper(self, paper_path: str, force_reprocess: bool = False, generate_queries: bool = None) -> bool:
        """Process a single paper to generate figure captions and queries."""
        # Use config default if not specified
        if generate_queries is None:
            generate_queries = PROCESSING_CONFIG.get("generate_queries", True)
            
        try:
            logger.info(f"Processing paper: {paper_path}")
            
            # Check if already processed (looking for figure_analysis.json)
            if not force_reprocess and self.check_if_processed(paper_path):
                logger.info(f"Paper already processed: {paper_path}. Use force_reprocess=True to reprocess.")
                return True
            
            # Handle different processing methods
            if self.use_assistants:
                # Assistants API processes entire paper at once
                figure_analysis = self.handler.process_paper_with_assistants(paper_path, generate_queries)
                
                # Add method info to each figure
                for figure_key, figure_data in figure_analysis.items():
                    if "method" not in figure_data:
                        figure_data["method"] = "assistants_api"
                        figure_data["context_used"] = "complete_paper"
            else:
                # Chat Completions API processes figures individually
                # Load only the markdown content for processing
                paper_content = self.load_paper_markdown(paper_path)
                if not paper_content:
                    logger.warning(f"No paper content found for {paper_path}")
                    return False
                
                # Find images
                image_files = self.find_images(paper_path)
                if not image_files:
                    logger.warning(f"No images found for {paper_path}")
                    return False
                
                # Process each image
                figure_analysis = {}
                
                for image_path in image_files:
                    try:
                        logger.info(f"Processing image: {image_path}")
                        
                        # Generate caption and queries (with optional queries support)
                        caption, queries = self.handler.process_figure_with_retry(
                            image_path, paper_content, generate_queries=generate_queries
                        )
                        
                        # Store results
                        image_filename = os.path.basename(image_path)
                        figure_analysis[image_filename] = {
                            "caption": caption,
                            "queries": queries,
                            "image_path": image_path,
                            "method": "chat_completions",
                            "context_used": "smart_extraction"
                        }
                        
                        logger.info(f"Successfully processed {image_filename}")
                        
                    except Exception as e:
                        logger.error(f"Failed to process image {image_path}: {e}")
                        # Continue with other images rather than failing completely
                        image_filename = os.path.basename(image_path)
                        figure_analysis[image_filename] = {
                            "error": str(e),
                            "image_path": image_path,
                            "method": "chat_completions"
                        }
            
            # Check if any figures were processed
            if not figure_analysis:
                logger.warning(f"No figures were processed for {paper_path}")
                return False
            
            # Save figure analysis to separate file
            self.save_figure_analysis(paper_path, figure_analysis)
            
            # Get usage summary from handler
            if hasattr(self.handler, 'get_usage_summary'):
                usage_summary = self.handler.get_usage_summary()
                logger.info(f"Processing summary: {usage_summary}")
                
                # Add usage summary to the saved data
                analysis_path = self.get_figure_analysis_path(paper_path)
                try:
                    with open(analysis_path, 'r', encoding='utf-8') as f:
                        analysis_data = json.load(f)
                    
                    analysis_data["usage_summary"] = usage_summary
                    
                    with open(analysis_path, 'w', encoding='utf-8') as f:
                        json.dump(analysis_data, f, indent=OUTPUT_CONFIG["indent_json"], ensure_ascii=False)
                    
                    logger.info(f"Added usage summary to: {analysis_path}")
                except Exception as e:
                    logger.warning(f"Could not add usage summary: {e}")
            
            # Cleanup for Assistants API
            if self.use_assistants and hasattr(self.handler, 'cleanup_files'):
                self.handler.cleanup_files()
            
            logger.info(f"Completed processing paper: {paper_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing paper {paper_path}: {e}")
            return False
    
    def get_processing_summary(self, paper_path: str) -> Dict[str, Any]:
        """Get summary of processing results for a paper."""
        try:
            analysis_data = self.load_figure_analysis(paper_path)
            figure_analysis = analysis_data.get("figure_analysis", {})
            
            total_figures = len(figure_analysis)
            successful_figures = len([f for f in figure_analysis.values() if "caption" in f])
            failed_figures = len([f for f in figure_analysis.values() if "error" in f])
            
            return {
                "paper_path": paper_path,
                "total_figures": total_figures,
                "successful_figures": successful_figures,
                "failed_figures": failed_figures,
                "processed": total_figures > 0,
                "success_rate": successful_figures / total_figures if total_figures > 0 else 0,
                "processing_timestamp": analysis_data.get("processing_timestamp", "")
            }
            
        except Exception as e:
            return {
                "paper_path": paper_path,
                "error": str(e),
                "processed": False
            } 