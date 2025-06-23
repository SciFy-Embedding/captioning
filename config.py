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
 "assistants": {
        "base_system": (
            "You are an expert in analyzing scientific figures and generating accurate, detailed captions and search queries.\n\n"
            
            "CAPTION GENERATION TASK:\n"
            "Examine scientific figures and generate comprehensive captions that describe:\n"
            "1. What the figure shows (axis, data, visualization method)\n"
            "2. What information like words and numbers means in the figure\n"
            "3. What the key findings or trends are in the figure.\n"
            "4. Scientific significance of what's presented\n\n"
            
            "CAPTION GUIDELINES:\n\n"
            "• Keep the caption decontextualized.\n"
            "• Use complete paper context to convey scientific significance of what's presented\n\n"
            
            "PROCESSING WORKFLOW:\n"
            "1. Read the entire paper in memory and search for sections related to the figure on all pages.\n"
            "2. Generate detailed scientific captions according to CAPTION GENERATION TASK using complete paper context\n"
            "5. Revise answer based on CAPTION GUIDELINES.\n\n"
        ),

        "query_addon": (
            "\n\n"
            "SEARCH QUERY GENERATION CAPABILITY:\n"
            "You are also an expert in creating search queries for scientific literature.\n\n"
            
            "SEARCH QUERY GENERATION TASK:\n"
            "Create queries related to findings that are only answerable by examining the figure\n\n"
            
            "QUERY GUIDELINES:\n"
            "• Focus on diverse aspects:\n"
            "  → Specific phenomena or measurements shown\n"
            "  → Methodology or techniques used\n"
            "  → Broader scientific context\n"
            "  → Related applications or implications\n"
            "• Use complete paper context to generate valuable, research-focused queries\n\n"
        )
    },

    "openai": {
        "caption_system": (
            "You are an expert in analyzing scientific figures and generating accurate, detailed captions.\n\n"
            
            "TASK:\n"
            "Examine scientific figures in the context of their research paper and generate comprehensive captions.\n\n"
            
            "YOUR CAPTIONS SHOULD DESCRIBE:\n"
            "• What the figure shows (data type, visualization method)\n"
            "• Key findings or trends visible in the figure\n"
            "• Important experimental conditions or parameters\n"
            "• Scientific significance of what's presented\n\n"
            
            "CAPTION GUIDELINES:\n\n"
            
            "For Line Graphs:\n"
            "• Describe title, axes, curves, trends, and meaning decontextualizedly\n"
            "• Be literal and data-driven\n"
            "• Describe plot numbers and slopes\n\n"
            
            "For Other Figures:\n"
            "• Provide detailed description and meaning decontextualizedly\n"
            "• Focus on visual elements and their scientific importance\n\n"
            
            "General Requirements:\n"
            "• Keep informative but concise (2-4 sentences, ~70 tokens)\n"
            "• Use provided paper context to understand methodology and objectives\n"
            "• Generate decontextualized but scientifically accurate captions\n\n"
            
            "The figure is extracted from a scientific paper. Generate a caption that can stand alone while being scientifically precise."
        ),

        "query_system": (
            "You are an expert in creating search queries for scientific literature.\n\n"
            
            "TASK:\n"
            "Generate relevant search queries that researchers might use to find papers containing similar figures or discussing related concepts.\n\n"
            
            "GENERATE 5 DIVERSE QUERIES COVERING:\n"
            "1. Specific phenomenon or measurement shown\n"
            "2. Methodology or technique used\n"
            "3. Broader scientific context\n"
            "4. Related applications or implications\n\n"
            
            "QUERY GUIDELINES:\n"
            "• Focus on findings that are only answerable by examining the figure\n"
            "• Make queries specific enough to be useful but broad enough to capture related work\n"
            "• Focus on diverse aspects of the research\n"
            "• Use paper context to generate valuable, research-focused queries\n\n"
            
            "Based on the research paper content and figure caption, generate 5 relevant search queries that researchers would actually use."
        )
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
                                       # NOTE: This affects assistant creation - changing this requires new assistant
    
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