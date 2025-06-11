import os
from typing import Dict, Any

# OpenAI API Configuration
OPENAI_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY"),
    "model": "gpt-4o",  # Current vision model (replaces deprecated gpt-4-vision-preview)
    "max_tokens": 1000,
    "temperature": 0.3,
}

# Prompts Configuration
PROMPTS = {

    "user_prompt": """The figure is extracted from a scientific paper markdown in the json file. If the figure is a line graph, give decontextualized one sentence caption about the title, axies, curves, trend for each figure, and one sentence for its meaning. The caption should be literal, data‐driven, descriptive of the plot's number and slopes. If the figure is not a line graph, do your best to give a  detailed description and meaning of the graph decontextualizedly. Shoot for a 70 token description.""",
    "caption_generation": {
        "system_prompt": """You are an expert in analyzing scientific figures and generating accurate, detailed captions. 
Your task is to examine scientific figures in the context of their research paper and generate comprehensive captions that describe:
1. What the figure shows (data type, visualization method)
2. Key findings or trends visible in the figure
3. Any important details about experimental conditions or parameters
4. The scientific significance of what's presented

Keep captions concise but informative, typically 2-4 sentences.""",
        
        "user_prompt": """The figure is extracted from a scientific paper markdown in the json file. If the figure is a line graph, give decontextualized one sentence caption about the title, axies, curves, trend for each figure, and one sentence for its meaning. The caption should be literal, data‐driven, descriptive of the plot's number and slopes. If the figure is not a line graph, do your best to give a  detailed description and meaning of the graph decontextualizedly. Shoot for a 70 token description..

Paper context:
{paper_content}

Please provide a caption that accurately describes what this figure shows and its significance in the context of this research."""
    },
    
    "query_generation": {
        "system_prompt": """You are an expert in creating search queries for scientific literature. 
Your task is to generate relevant queries that researchers might use to find papers containing similar figures or discussing related concepts.

Generate 3-5 diverse queries that cover:
1. The specific phenomenon or measurement shown
2. The methodology or technique used
3. The broader scientific context
4. Related applications or implications

Make queries specific enough to be useful but broad enough to capture related work.""",
        
        "user_prompt": """Based on the research paper content and the figure image, generate 3-5 relevant search queries that researchers might use to find papers with similar content or figures.

Paper context:
{paper_content}

Figure caption: {caption}

Please provide search queries as a list, each focusing on different aspects of the figure's content."""
    }
}

# Processing Configuration
PROCESSING_CONFIG = {
    "batch_size": 10,
    "retry_attempts": 3,
    "retry_delay": 2,
    "max_paper_content_length": 8000,  # Characters, not tokens
    "supported_image_formats": ['.jpg', '.jpeg', '.png', '.gif', '.bmp'],
    
    # Feature toggles
    "generate_queries": True,           # Set to False to generate only captions (saves time/cost)
    
    # Concurrent Processing Settings
    "concurrent_papers": True,          # Process multiple papers simultaneously 
    "max_paper_workers": 3,            # Max papers processing concurrently
    "assistants_timeout": 90,          # Seconds to wait for assistant runs (increased)
    
    # Optimized Polling Settings (reduces API calls significantly)
    "assistants_polling_intervals": [2.0, 5.0, 10.0, 15.0],  # Progressive polling intervals
    "assistants_reduce_api_calls": True,  # Use optimized polling strategy
}

# Output Configuration
OUTPUT_CONFIG = {
    "indent_json": 2,  # Pretty print JSON output
} 