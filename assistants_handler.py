import json
import os
import time
import concurrent.futures
from typing import Dict, List, Optional, Tuple, Any
from openai import OpenAI
import logging
from config import OPENAI_CONFIG, PROMPTS, PROCESSING_CONFIG

logger = logging.getLogger(__name__)

# Current OpenAI Assistants API pricing (as of 2024) - update as needed
ASSISTANTS_PRICING = {
    "gpt-4o": {
        "input_tokens": 0.0025 / 1000,   # $0.0025 per 1K input tokens  
        "output_tokens": 0.01 / 1000,    # $0.01 per 1K output tokens
        "image_tokens": 0.00765 / 1000   # $0.00765 per 1K image tokens
    },
    "file_storage": 0.20 / (1024**3) / 24,  # $0.20 per GB per day (hourly rate)
    "file_search": 0.10 / 1000           # $0.10 per 1K tokens searched
}

class AssistantsHandler:
    """Alternative implementation using OpenAI Assistants API for full paper processing.
    
    Key differences from Chat Completions approach:
    - Uploads entire paper as searchable file (no truncation needed)
    - AI can dynamically search through complete paper content
    - No paper content included in prompts (uses file search instead)
    """
    
    # Class-level assistant cache to reuse across instances
    _assistant_cache = {}
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Assistants API client."""
        self.api_key = api_key or OPENAI_CONFIG["api_key"]
        if not self.api_key:
            raise ValueError("OpenAI API key not found.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.assistant_id = None
        self.current_thread_id = None
        self.current_file_id = None
        self.uploaded_files = {}  # Track uploaded files for cleanup
        
        # Cost and timing tracking
        self.total_cost = 0.0
        self.total_time = 0.0
        self.request_count = 0
        self.file_storage_cost = 0.0
        self.file_search_cost = 0.0
        self.run_costs = []  # Track individual run costs
        self.file_sizes = {}  # Track file sizes for storage cost calculation
    
    def create_assistant(self) -> str:
        """Create or reuse the figure analysis assistant."""
        # Create a cache key based on the prompt content
        system_prompt = PROMPTS["caption_generation"]["system_prompt"]
        cache_key = hash(system_prompt)
        
        # Check if we already have a cached assistant
        if cache_key in self._assistant_cache:
            self.assistant_id = self._assistant_cache[cache_key]
            logger.info(f"Reusing cached assistant: {self.assistant_id}")
            return self.assistant_id
        
        if self.assistant_id:
            return self.assistant_id
        
        # Use the original system prompt from config as instructions
        # Enhanced with file search capability
        system_instructions = system_prompt + """

Additionally, you have access to the complete research paper through file search. 
Use this full context to provide more accurate and detailed analysis than would be possible with truncated content.

When processing figures:
1. Search the paper for relevant sections related to the figure
2. Generate detailed scientific captions using complete paper context
3. Create relevant search queries based on comprehensive understanding

No paper content will be provided in prompts - use file search to access what you need."""
        
        assistant = self.client.beta.assistants.create(
            name="Scientific Figure Analyzer",
            instructions=system_instructions,
            model="gpt-4o",
            tools=[{"type": "file_search"}]
        )
        
        self.assistant_id = assistant.id
        
        # Cache the assistant for reuse
        self._assistant_cache[cache_key] = self.assistant_id
        
        logger.info(f"Created and cached assistant: {self.assistant_id}")
        return self.assistant_id
    
    def upload_paper_content(self, paper_path: str) -> str:
        """Upload complete paper content as a searchable file."""
        upload_start = time.time()
        
        # Check if already uploaded
        if paper_path in self.uploaded_files:
            return self.uploaded_files[paper_path]
        
        try:
            # Load paper data
            json_path = os.path.join(paper_path, "result.json")
            with open(json_path, 'r', encoding='utf-8') as f:
                paper_data = json.load(f)
            
            # Extract complete markdown content (no truncation)
            marker = paper_data.get("marker", {})
            if isinstance(marker, dict):
                full_markdown_content = marker.get("markdown", "")
            else:
                full_markdown_content = str(marker)
            
            if not full_markdown_content:
                raise ValueError("No markdown content found")
            
            # Create temporary file with complete paper
            paper_id = paper_data.get("id", "unknown")
            temp_filename = f"temp_paper_{paper_id.replace(':', '_')}.txt"
            
            # Add paper metadata and complete content
            complete_content = f"""Paper ID: {paper_id}
PDF Path: {paper_data.get('pdf_path', 'Unknown')}

=== COMPLETE PAPER CONTENT ===
{full_markdown_content}
"""
            
            with open(temp_filename, 'w', encoding='utf-8') as f:
                f.write(complete_content)
            
            # Upload complete file to OpenAI
            with open(temp_filename, 'rb') as f:
                file = self.client.files.create(
                    file=f,
                    purpose='assistants'
                )
            
            # Clean up temp file
            os.remove(temp_filename)
            
            # Track file for cost calculation
            content_length = len(complete_content.encode('utf-8'))
            self.track_file_upload(file.id, content_length)
            
            # Cache the file ID
            self.uploaded_files[paper_path] = file.id
            upload_time = time.time() - upload_start
            logger.info(f"Uploaded complete paper content: {file.id} ({len(full_markdown_content):,} characters) in {upload_time:.2f}s")
            return file.id
            
        except Exception as e:
            logger.error(f"Error uploading paper content: {e}")
            raise
    
    def create_thread_with_paper(self, file_id: str) -> str:
        """Create a conversation thread and store file_id for message attachment."""
        try:
            # Create a simple thread without tool_resources (we'll attach files to messages instead)
            thread = self.client.beta.threads.create()
            
            self.current_thread_id = thread.id
            self.current_file_id = file_id  # Store for use in messages
            logger.info(f"Created thread: {thread.id}")
            return thread.id
            
        except Exception as e:
            logger.error(f"Error creating thread: {e}")
            raise
    
    def process_figure(self, thread_id: str, image_path: str, generate_queries: bool = None) -> Tuple[str, List[str]]:
        """Process a figure using the assistant with complete paper access."""
        # Use config default if not specified
        if generate_queries is None:
            generate_queries = PROCESSING_CONFIG.get("generate_queries", True)
            
        try:
            # Upload image as file for this specific figure
            with open(image_path, "rb") as image_file:
                image_upload = self.client.files.create(
                    file=image_file,
                    purpose='vision'
                )
            
            image_file_id = image_upload.id  # Store the image file ID separately
            logger.info(f"Uploaded image file: {image_file_id} for {image_path}")
            
            # Request caption using attached files
            caption_request = """Please analyze this figure and generate a detailed scientific caption.

You have access to the complete research paper. Please use the full paper context to understand:
- The experimental context and methodology
- How this figure relates to the research objectives
- Key findings and their significance
- Any relevant technical details or parameters

Generate a comprehensive caption using the complete paper context."""
            
            # Create message for caption generation with both paper and image file attachments
            message = self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=[
                    {
                        "type": "text",
                        "text": caption_request
                    },
                    {
                        "type": "image_file",
                        "image_file": {
                            "file_id": image_file_id
                        }
                    }
                ],
                attachments=[
                    {
                        "file_id": self.current_file_id,  # Use paper file ID for paper context
                        "tools": [{"type": "file_search"}]
                    }
                ]
            )
            
            # Run the assistant for caption
            run = self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=self.assistant_id
            )
            
            # Wait for caption completion
            caption = self._wait_for_completion(thread_id, run.id)
            
            queries = []
            if generate_queries:
                # Generate queries using the caption and complete paper access
                query_request = f"""Based on the figure and the caption you just generated, create 3-5 relevant search queries.

Generated caption: {caption}

Use the complete paper context to understand the broader research context and generate diverse queries that researchers might use to find:
- Papers with similar experimental approaches
- Related phenomena or findings  
- Comparable methodologies or techniques
- Applications in similar research areas

Use the complete paper context to make queries specific and valuable."""
                
                # Create message for query generation with file attachment (only paper, not image)
                query_message = self.client.beta.threads.messages.create(
                    thread_id=thread_id,
                    role="user",
                    content=[
                        {
                            "type": "text", 
                            "text": query_request
                        }
                    ],
                    attachments=[
                        {
                            "file_id": self.current_file_id,  # Use paper file ID for context
                            "tools": [{"type": "file_search"}]
                        }
                    ]
                )
                
                # Run assistant for queries
                query_run = self.client.beta.threads.runs.create(
                    thread_id=thread_id,
                    assistant_id=self.assistant_id
                )
                
                # Wait for queries completion
                queries_response = self._wait_for_completion(thread_id, query_run.id)
                queries = self._parse_queries(queries_response)
            
            # Clean up uploaded image file immediately after processing
            try:
                self.client.files.delete(image_file_id)
                logger.info(f"Deleted image file: {image_file_id}")
            except Exception as e:
                logger.warning(f"Could not delete image file {image_file_id}: {e}")
            
            if generate_queries:
                logger.info(f"Processed figure with complete paper context: {image_path}")
            else:
                logger.info(f"Processed figure (caption only) with complete paper context: {image_path}")
            return caption.strip(), queries
            
        except Exception as e:
            logger.error(f"Error processing figure {image_path}: {e}")
            # Clean up image file even if processing failed
            try:
                if 'image_file_id' in locals():
                    self.client.files.delete(image_file_id)
                    logger.info(f"Cleaned up image file after error: {image_file_id}")
            except:
                pass
            raise
    
    def _wait_for_completion(self, thread_id: str, run_id: str, max_wait: int = 60) -> str:
        """Wait for assistant run to complete with optimized polling to reduce API calls."""
        start_time = time.time()
        poll_count = 0
        
        while time.time() - start_time < max_wait:
            run = self.client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run_id
            )
            
            if run.status == "completed":
                # Calculate total run time
                run_time = time.time() - start_time
                
                # Track usage statistics
                self.update_usage_stats(run, run_time)
                
                # Get the response
                messages = self.client.beta.threads.messages.list(
                    thread_id=thread_id,
                    order="desc",
                    limit=1
                )
                
                if messages.data:
                    return messages.data[0].content[0].text.value
                else:
                    raise Exception("No response received")
            
            elif run.status in ["failed", "cancelled", "expired"]:
                # Get detailed error information
                error_details = f"Run failed with status: {run.status}"
                
                # Try to get more specific error information
                if hasattr(run, 'last_error') and run.last_error:
                    error_details += f"\nError code: {run.last_error.code}"
                    error_details += f"\nError message: {run.last_error.message}"
                
                # Also check if there are any failed steps
                if hasattr(run, 'required_action') and run.required_action:
                    error_details += f"\nRequired action: {run.required_action}"
                
                logger.error(f"Assistant run failed: {error_details}")
                raise Exception(error_details)
            
            # OPTIMIZED POLLING: Much longer intervals to reduce API calls
            poll_count += 1
            if poll_count == 1:
                sleep_time = 2.0   # First check after 2 seconds
            elif poll_count <= 3:
                sleep_time = 5.0   # Then check every 5 seconds for a few times
            elif poll_count <= 6:
                sleep_time = 10.0  # Then every 10 seconds
            else:
                sleep_time = 15.0  # Finally every 15 seconds for long-running tasks
            
            logger.debug(f"Polling attempt {poll_count}, waiting {sleep_time}s (run time: {time.time() - start_time:.1f}s)")
            time.sleep(sleep_time)
        
        raise Exception("Assistant run timed out")
    
    def _parse_queries(self, queries_text: str) -> List[str]:
        """Parse queries from assistant response (using same logic as openai_handler)."""
        lines = queries_text.strip().split('\n')
        queries = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove numbering and bullet points (same logic as openai_handler)
            if line.startswith(('1.', '2.', '3.', '4.', '5.', '-', '•', '*')):
                # Find the first space after the number/bullet and take the rest
                space_idx = line.find(' ')
                if space_idx != -1:
                    query = line[space_idx:].strip()
                    if query:
                        queries.append(query)
            elif line and len(queries) < 5:  # Fallback for lines without clear formatting
                queries.append(line)
        
        return queries[:5]  # Limit to 5 queries max
    
    def cleanup_files(self):
        """Clean up uploaded files."""
        for paper_path, file_id in self.uploaded_files.items():
            try:
                self.client.files.delete(file_id)
                logger.info(f"Deleted file: {file_id}")
            except Exception as e:
                logger.warning(f"Could not delete file {file_id}: {e}")
        
        self.uploaded_files.clear()
    
    def process_paper_with_assistants(self, paper_path: str, generate_queries: bool = None) -> Dict[str, any]:
        """Process entire paper using Assistants API with ONE call for ALL figures."""
        # Use config default if not specified
        if generate_queries is None:
            generate_queries = PROCESSING_CONFIG.get("generate_queries", True)
            
        try:
            # Ensure assistant exists
            if not self.assistant_id:
                self.create_assistant()
            
            # Upload complete paper content once
            paper_file_id = self.upload_paper_content(paper_path)
            
            # Create ONE thread for this entire paper
            thread = self.client.beta.threads.create()
            thread_id = thread.id
            logger.info(f"Created thread {thread_id} for paper: {os.path.basename(paper_path)}")
            
            # Find and upload ALL images at once
            images_dir = os.path.join(paper_path, "images")
            if not os.path.exists(images_dir):
                return {}
            
            image_files = []
            uploaded_image_files = []
            
            for image_file in os.listdir(images_dir):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                    image_path = os.path.join(images_dir, image_file)
                    image_files.append((image_file, image_path))
            
            if not image_files:
                logger.warning(f"No valid images found in {images_dir}")
                return {}
            
            # Upload ALL images
            logger.info(f"Uploading {len(image_files)} images for batch processing")
            image_attachments = []
            
            for image_file, image_path in image_files:
                with open(image_path, "rb") as img_file:
                    image_upload = self.client.files.create(
                        file=img_file,
                        purpose='vision'
                    )
                
                uploaded_image_files.append(image_upload.id)
                image_attachments.append({
                    "type": "image_file",
                    "image_file": {"file_id": image_upload.id}
                })
                
                # Track for cost calculation
                image_size = os.path.getsize(image_path)
                self.track_file_upload(image_upload.id, image_size)
                logger.info(f"Uploaded {image_file}: {image_upload.id}")
            
            # Process ALL figures in ONE API call
            if generate_queries:
                batch_request = f"""Please analyze ALL {len(image_files)} figures in this research paper and provide:

1. **Detailed Scientific Caption** for each figure
2. **3-5 Search Queries** for each figure

For each figure, use the complete paper context to understand:
- Experimental context and methodology  
- How each figure relates to research objectives
- Key findings and significance
- Technical details and parameters

Please format your response as JSON:
{{
  "figure_1_filename": {{
    "caption": "detailed caption here",
    "queries": ["query 1", "query 2", "query 3", "query 4", "query 5"]
  }},
  "figure_2_filename": {{
    "caption": "detailed caption here", 
    "queries": ["query 1", "query 2", "query 3", "query 4", "query 5"]
  }}
  // ... continue for all figures
}}

Figure filenames: {[img[0] for img in image_files]}"""
            else:
                batch_request = f"""Please analyze ALL {len(image_files)} figures in this research paper and provide:

1. **Detailed Scientific Caption** for each figure

For each figure, use the complete paper context to understand:
- Experimental context and methodology  
- How each figure relates to research objectives
- Key findings and significance
- Technical details and parameters

Please format your response as JSON:
{{
  "figure_1_filename": {{
    "caption": "detailed caption here"
  }},
  "figure_2_filename": {{
    "caption": "detailed caption here"
  }}
  // ... continue for all figures
}}

Figure filenames: {[img[0] for img in image_files]}"""
            
            # Create message with ALL images and paper attachment
            message_content = [{"type": "text", "text": batch_request}]
            message_content.extend(image_attachments)
            
            message = self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=message_content,
                attachments=[{
                    "file_id": paper_file_id,
                    "tools": [{"type": "file_search"}]
                }]
            )
            
            # Run the assistant ONCE for ALL figures
            logger.info(f"Processing ALL {len(image_files)} figures in single API call")
            run = self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=self.assistant_id
            )
            
            # Wait for completion
            start_time = time.time()
            response = self._wait_for_completion(thread_id, run.id, max_wait=180)  # Longer timeout for batch
            total_time = time.time() - start_time
            
            logger.info(f"Batch processing completed in {total_time:.2f}s")
            
            # Parse the JSON response
            figure_analysis = self._parse_batch_response(response, image_files)
            
            # Clean up all uploaded image files
            for image_file_id in uploaded_image_files:
                try:
                    self.client.files.delete(image_file_id)
                    logger.info(f"Deleted image file: {image_file_id}")
                except Exception as e:
                    logger.warning(f"Could not delete image file {image_file_id}: {e}")
            
            logger.info(f"Successfully processed {len(figure_analysis)} figures in batch")
            return figure_analysis
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            # Clean up on error
            if 'uploaded_image_files' in locals():
                for image_file_id in uploaded_image_files:
                    try:
                        self.client.files.delete(image_file_id)
                    except:
                        pass
            raise
    
    def calculate_run_cost(self, run_data) -> float:
        """Calculate cost for a single assistant run based on usage."""
        if not hasattr(run_data, 'usage') or not run_data.usage:
            return 0.0
        
        usage = run_data.usage
        pricing = ASSISTANTS_PRICING["gpt-4o"]
        
        input_cost = usage.prompt_tokens * pricing["input_tokens"] if hasattr(usage, 'prompt_tokens') else 0
        output_cost = usage.completion_tokens * pricing["output_tokens"] if hasattr(usage, 'completion_tokens') else 0
        
        total_cost = input_cost + output_cost
        return total_cost
    
    def calculate_file_storage_cost(self, file_size_bytes: int, hours_stored: float) -> float:
        """Calculate file storage cost."""
        gb_size = file_size_bytes / (1024**3)
        return gb_size * ASSISTANTS_PRICING["file_storage"] * hours_stored
    
    def track_file_upload(self, file_id: str, content_length: int):
        """Track file upload for cost calculation."""
        self.file_sizes[file_id] = {
            "size": content_length,
            "upload_time": time.time()
        }
        logger.info(f"Tracking file {file_id}: {content_length:,} bytes")
    
    def update_usage_stats(self, run_data, request_time: float):
        """Update usage statistics from assistant run."""
        cost = self.calculate_run_cost(run_data)
        
        self.run_costs.append({
            "cost": cost,
            "time": request_time,
            "timestamp": time.time()
        })
        
        self.total_cost += cost
        self.total_time += request_time
        self.request_count += 1
        
        logger.info(f"Run cost: ${cost:.4f} | Time: {request_time:.2f}s")
    
    def calculate_total_file_costs(self) -> float:
        """Calculate total file storage costs based on upload duration."""
        total_storage_cost = 0.0
        current_time = time.time()
        
        for file_id, file_info in self.file_sizes.items():
            hours_stored = (current_time - file_info["upload_time"]) / 3600
            storage_cost = self.calculate_file_storage_cost(file_info["size"], hours_stored)
            total_storage_cost += storage_cost
        
        return total_storage_cost
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get comprehensive usage summary including file costs."""
        file_storage_cost = self.calculate_total_file_costs()
        total_cost_with_storage = self.total_cost + file_storage_cost
        
        return {
            "total_cost": round(total_cost_with_storage, 4),
            "api_cost": round(self.total_cost, 4),
            "file_storage_cost": round(file_storage_cost, 4),
            "total_time": round(self.total_time, 2),
            "request_count": self.request_count,
            "average_cost_per_request": round(self.total_cost / max(1, self.request_count), 4),
            "average_time_per_request": round(self.total_time / max(1, self.request_count), 2),
            "files_uploaded": len(self.file_sizes),
            "total_file_size": sum(info["size"] for info in self.file_sizes.values())
        }
    
    def _parse_batch_response(self, response_text: str, image_files: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Parse the batch JSON response from assistant processing all figures."""
        try:
            import json
            import re
            
            # Try to extract JSON from the response
            # Sometimes the response includes extra text before/after the JSON
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                batch_data = json.loads(json_str)
            else:
                # Fallback: try to parse the entire response as JSON
                batch_data = json.loads(response_text)
            
            figure_analysis = {}
            
            # Process each figure from the batch response
            for image_file, image_path in image_files:
                # Try to find the figure data in various possible formats
                figure_data = None
                
                # Try exact filename match first
                if image_file in batch_data:
                    figure_data = batch_data[image_file]
                else:
                    # Try without extension
                    filename_no_ext = os.path.splitext(image_file)[0]
                    if filename_no_ext in batch_data:
                        figure_data = batch_data[filename_no_ext]
                    else:
                        # Try other common patterns
                        for key in batch_data.keys():
                            if image_file.lower() in key.lower() or filename_no_ext.lower() in key.lower():
                                figure_data = batch_data[key]
                                break
                
                if figure_data and isinstance(figure_data, dict):
                    # Extract caption and queries
                    caption = figure_data.get('caption', '')
                    queries = figure_data.get('queries', [])
                    
                    # Ensure queries is a list
                    if isinstance(queries, str):
                        queries = [queries]
                    elif not isinstance(queries, list):
                        queries = []
                    
                    figure_analysis[image_file] = {
                        "caption": caption,
                        "queries": queries[:5],  # Limit to 5 queries
                        "image_path": image_path,
                        "method": "assistants_api_batch",
                        "context_used": "complete_paper"
                    }
                else:
                    # Fallback if figure not found in response
                    logger.warning(f"Could not find data for {image_file} in batch response")
                    figure_analysis[image_file] = {
                        "error": "Not found in batch response",
                        "image_path": image_path,
                        "method": "assistants_api_batch"
                    }
            
            return figure_analysis
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse batch response as JSON: {e}")
            logger.debug(f"Response text: {response_text[:500]}...")
            
            # Fallback: create error entries for all figures
            figure_analysis = {}
            for image_file, image_path in image_files:
                figure_analysis[image_file] = {
                    "error": "Failed to parse batch response",
                    "image_path": image_path,
                    "method": "assistants_api_batch"
                }
            return figure_analysis
            
        except Exception as e:
            logger.error(f"Error parsing batch response: {e}")
            
            # Fallback: create error entries for all figures
            figure_analysis = {}
            for image_file, image_path in image_files:
                figure_analysis[image_file] = {
                    "error": str(e),
                    "image_path": image_path,
                    "method": "assistants_api_batch"
                }
            return figure_analysis 