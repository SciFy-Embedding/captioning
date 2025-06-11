import base64
import json
import time
from typing import Dict, List, Optional, Tuple, Any
import openai
from openai import OpenAI
import logging
from config import OPENAI_CONFIG, PROMPTS, PROCESSING_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Current OpenAI pricing (as of 2024) - update as needed
PRICING = {
    "gpt-4o": {
        "input_tokens": 0.0025 / 1000,   # $0.0025 per 1K input tokens
        "output_tokens": 0.01 / 1000,    # $0.01 per 1K output tokens
        "image_tokens": 0.00765 / 1000   # $0.00765 per 1K image tokens
    }
}

class OpenAIHandler:
    """Handles OpenAI API interactions for figure caption and query generation."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize OpenAI client with API key."""
        self.api_key = api_key or OPENAI_CONFIG["api_key"]
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = OPENAI_CONFIG["model"]
        self.max_tokens = OPENAI_CONFIG["max_tokens"]
        self.temperature = OPENAI_CONFIG["temperature"]
        self.total_cost = 0.0
        self.total_time = 0.0
        self.request_count = 0
        self.total_tokens = {"input": 0, "output": 0, "image": 0}
    
    def calculate_cost(self, input_tokens: int, output_tokens: int, image_tokens: int = 0) -> float:
        """Calculate cost based on token usage."""
        pricing = PRICING["gpt-4o"]
        cost = (
            input_tokens * pricing["input_tokens"] +
            output_tokens * pricing["output_tokens"] +
            image_tokens * pricing["image_tokens"]
        )
        return cost
    
    def update_usage_stats(self, response, request_time: float, image_tokens: int = 0):
        """Update usage statistics from API response."""
        if hasattr(response, 'usage') and response.usage:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            
            cost = self.calculate_cost(input_tokens, output_tokens, image_tokens)
            
            self.total_tokens["input"] += input_tokens
            self.total_tokens["output"] += output_tokens
            self.total_tokens["image"] += image_tokens
            self.total_cost += cost
            self.total_time += request_time
            self.request_count += 1
            
            logger.info(f"Request cost: ${cost:.4f} | Time: {request_time:.2f}s | Tokens: {input_tokens}+{output_tokens}")
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get summary of usage statistics."""
        return {
            "total_cost": round(self.total_cost, 4),
            "total_time": round(self.total_time, 2),
            "request_count": self.request_count,
            "total_tokens": self.total_tokens.copy(),
            "average_cost_per_request": round(self.total_cost / max(1, self.request_count), 4),
            "average_time_per_request": round(self.total_time / max(1, self.request_count), 2)
        }
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string for OpenAI API."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            raise
    
    def truncate_paper_content(self, content: str) -> str:
        """Extract relevant paper content to fit within token limits while preserving key information."""
        max_length = PROCESSING_CONFIG["max_paper_content_length"]
        if len(content) <= max_length:
            return content
        
        # Try to extract key sections instead of simple truncation
        sections_to_extract = [
            "abstract",
            "introduction", 
            "conclusion",
            "results",
            "discussion"
        ]
        
        extracted_content = self._extract_key_sections(content, sections_to_extract, max_length)
        
        if extracted_content and len(extracted_content) <= max_length:
            return extracted_content
        
        # Fallback to simple truncation if section extraction fails
        return self._simple_truncate(content, max_length)
    
    def _extract_key_sections(self, content: str, sections: List[str], max_length: int) -> str:
        """Extract key sections from paper content."""
        content_lower = content.lower()
        extracted_parts = []
        
        for section in sections:
            # Look for section headers (various formats)
            patterns = [
                f"\n# {section}",
                f"\n## {section}", 
                f"\n{section.upper()}",
                f"\n{section.title()}",
                f"\n**{section}**"
            ]
            
            for pattern in patterns:
                start_idx = content_lower.find(pattern.lower())
                if start_idx != -1:
                    # Find the next section or end of content
                    section_start = start_idx + len(pattern)
                    
                    # Look for next major section
                    next_section_idx = len(content)
                    for next_pattern in ["\n# ", "\n## ", "\n\n**"]:
                        next_idx = content_lower.find(next_pattern, section_start + 100)
                        if next_idx != -1 and next_idx < next_section_idx:
                            next_section_idx = next_idx
                    
                    section_content = content[section_start:next_section_idx].strip()
                    if section_content:
                        extracted_parts.append(f"\n## {section.title()}\n{section_content}")
                    break
        
        result = "".join(extracted_parts)
        
        # If extracted content is still too long, truncate proportionally
        if len(result) > max_length:
            result = self._simple_truncate(result, max_length)
        
        return result
    
    def _simple_truncate(self, content: str, max_length: int) -> str:
        """Simple truncation with sentence boundary detection."""
        if len(content) <= max_length:
            return content
        
        # Try to cut at a sentence boundary
        truncated = content[:max_length]
        last_period = truncated.rfind('.')
        if last_period > max_length * 0.8:  # If we find a period in the last 20%
            return truncated[:last_period + 1]
        else:
            return truncated + "..."
    
    def generate_caption(self, image_path: str, paper_content: str) -> str:
        """Generate caption for a figure using GPT-4o vision with timing and cost tracking."""
        # Encode image
        base64_image = self.encode_image(image_path)
        
        # Extract relevant sections and truncate if needed
        relevant_content = self.truncate_paper_content(paper_content)
        
        # Create messages with system prompt and user request
        messages = [
            {
                "role": "system",
                "content": PROMPTS["caption_generation"]["system_prompt"]
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": PROMPTS["caption_generation"]["user_prompt"].format(
                            paper_content=relevant_content
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        
        # Make API call with timing
        request_start = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        request_time = time.time() - request_start
        
        # Update usage statistics (estimate image tokens)
        image_tokens = len(base64_image) // 750  # Rough estimate: ~750 chars per token
        self.update_usage_stats(response, request_time, image_tokens)
        
        return response.choices[0].message.content
    
    def generate_queries(self, caption: str, paper_content: str) -> List[str]:
        """Generate search queries based on caption and paper content with timing and cost tracking."""
        # Extract relevant sections and truncate if needed  
        relevant_content = self.truncate_paper_content(paper_content)
        
        # Create messages for query generation
        messages = [
            {
                "role": "system", 
                "content": PROMPTS["query_generation"]["system_prompt"]
            },
            {
                "role": "user",
                "content": PROMPTS["query_generation"]["user_prompt"].format(
                    caption=caption,
                    paper_content=relevant_content
                )
            }
        ]
        
        # Make API call with timing
        request_start = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        request_time = time.time() - request_start
        
        # Update usage statistics
        self.update_usage_stats(response, request_time)
        
        # Parse queries from response
        queries_text = response.choices[0].message.content
        queries = self._parse_queries(queries_text)
        
        return queries
    
    def _parse_queries(self, queries_text: str) -> List[str]:
        """Parse queries from OpenAI response text."""
        lines = queries_text.strip().split('\n')
        queries = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove numbering and bullet points
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
    
    def process_figure_with_retry(self, image_path: str, paper_content: str, max_retries: int = 3, generate_queries: bool = None) -> Tuple[str, List[str]]:
        """Process a figure with retry logic, including timing and cost analysis."""
        # Use config default if not specified
        if generate_queries is None:
            generate_queries = PROCESSING_CONFIG.get("generate_queries", True)
            
        start_time = time.time()
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Processing figure: {image_path} (attempt {attempt + 1}/{max_retries})")
                
                # Generate caption with timing
                caption_start = time.time()
                caption = self.generate_caption(image_path, paper_content)
                caption_time = time.time() - caption_start
                
                # Conditionally generate queries
                queries = []
                queries_time = 0
                if generate_queries:
                    queries_start = time.time()
                    queries = self.generate_queries(caption, paper_content)
                    queries_time = time.time() - queries_start
                
                total_time = time.time() - start_time
                if generate_queries:
                    logger.info(f"Figure processed successfully in {total_time:.2f}s (caption: {caption_time:.2f}s, queries: {queries_time:.2f}s)")
                else:
                    logger.info(f"Figure processed successfully in {total_time:.2f}s (caption only: {caption_time:.2f}s)")
                
                return caption, queries
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to process figure after {max_retries} attempts: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff 