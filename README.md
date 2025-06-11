# Figure Caption and Query Generator

**AI-powered figure analysis for scientific papers using OpenAI's vision models with optimized batch processing.**

## 🛠️ Installation

```bash
# Install dependencies
pip install openai

# Set API key
export OPENAI_API_KEY="your-key-here"
```

## 🎯 Usage

### Quick Start
```bash
# Fast processing (recommended)
python main.py process --path ./extracted_pdfs

# Complete paper access (slower but thorough)
python main.py process --path ./extracted_pdfs --method assistants

# Caption-only processing (faster and cheaper)
python main.py process --path ./extracted_pdfs --no-queries

# Single paper
python main.py process-single --path ./paper_directory
```

### Commands
```bash
python main.py process --path DIR        # Process all papers
python main.py process-single --path DIR # Process one paper  
python main.py stats --path DIR          # Show statistics
python main.py list --path DIR           # List papers
```

### Options
- `--method chat` - Fast smart extraction (default)
- `--method assistants` - Complete paper access via file search
- `--force` - Reprocess existing results
- `--save-results FILE` - Export batch results
- `--no-queries` - Generate only captions, skip queries (faster and cheaper)

## 📁 Input/Output

**Required Structure:**
```
paper_directory/
├── result.json          # Paper data with 'marker.markdown' field
└── images/              # Figure images (.jpg, .png, etc.)
```

**Generated Output:**
```
paper_directory/
├── result.json          # Original (untouched)
├── images/              # Original images
└── figure_analysis.json # 🆕 Generated analysis
```

**Analysis Format:**
```json
{
  "paper_id": "arxiv-id",
  "processing_timestamp": "2024-12-19T...",
  "figure_analysis": {
    "figure1.jpg": {
      "caption": "Detailed AI-generated caption...",
      "queries": ["search query 1", "search query 2", "..."],
      "image_path": "/full/path/to/image"
    }
  },
  "processing_summary": {
    "total_cost": 0.067,
    "total_time": 25.3,
    "request_count": 1
  }
}
```

## ⚙️ Processing Methods

### 🏃‍♂️ Chat Method (Default)
- **Speed:** Ultra-fast
- **API calls:** 1 per paper
- **Cost:** ~$0.03 per paper
- **Context:** Smart extraction (8K chars)
- **Best for:** Large-scale processing

### 🔍 Assistants Method  
- **Speed:** Moderate
- **API calls:** 1 per paper
- **Cost:** ~$0.07 per paper
- **Context:** Complete paper access
- **Best for:** Maximum accuracy

## 🎛️ Configuration

Edit `config.py` to customize:
- **Query generation toggle** - Set `generate_queries: False` for caption-only processing
- Concurrent processing settings
- API timeouts and retries
- Prompt templates
- Cost tracking

**Caption-only benefits:**
- ~50% faster processing
- ~30% cost reduction  
- Perfect for workflows that only need figure descriptions

## 🔧 Troubleshooting

**No papers found?** Check directory structure and `result.json` files  
**API errors?** Verify `OPENAI_API_KEY` is set correctly  
**High costs?** Use `--method chat` for faster, cheaper processing  
**Need support?** Check logs for detailed error information
