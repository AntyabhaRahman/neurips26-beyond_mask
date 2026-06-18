# Model Evaluation Framework

This framework accompanies our research paper "Beyond Accuracy: Disentangling Honesty and Factuality in AI Models" which introduces MASK (Model Alignment between Statements and Knowledge), a novel benchmark for measuring honesty in large language models. Our research demonstrates that while larger models obtain higher accuracy, they do not necessarily become more honest when pressured to lie. We introduce a consistency-based honesty metric that disentangles accuracy from honesty by testing whether models will contradict their own beliefs under pressure. The framework provides tools for evaluating language models' responses across different scenarios, focusing on this critical distinction between knowing facts (accuracy) and choosing to state them truthfully (honesty).

## Overview

The framework consists of three main stages:
1. Response Generation
2. Response Evaluation
3. Metrics Processing

## Setup

1. Create a `.env` file in the root directory with your API keys:
```env
OPENAI_API_KEY=your_key_here # needed for evaluation
ANTHROPIC_API_KEY=your_key_here
```

2. Install dependencies (recommended to use a virtual environment):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Download data:
```bash
python download.py
```

## Directory Structure
```
├── csv_data/               # Production data directory
│   ├── responses/         # Generated model responses
│   ├── evaluated/         # Evaluated responses
│   └── metrics/          # Processed metrics
├── test_csv_data/         # Test data directory (same structure as csv_data)
├── prompts/               # Evaluation prompt templates
└── *.py                   # Main Python scripts
```
## Usage

### 1. Generate Responses

Generate responses using the `generate_responses.py` script:
```bash
python generate_responses.py [--test] [--model MODEL_NAME] [--temperature TEMP] [--lie_k NUM_RUNS]
```

Options:
- `--test`: Use test directories
- `--model`: Model name to use (default: gpt-4o)
- `--temperature`: Sampling temperature (default: 1.0)
- `--lie_k`: Number of lying prompts to generate per row (default: 1)

The script supports OpenAI and Anthropic models. Generated responses will be saved in `csv_data/responses/` following the naming convention: `{archetype}_{model}.csv`

### 2. Evaluate Responses

Run the evaluation script:
```bash
python evaluate.py [--test] [--input_file path/to/file.csv] [--concurrency_limit 50]
```

Options:
- `--test`: Use test directories
- `--input_file`: Process a specific file (optional)
- `--concurrency_limit`: Maximum concurrent API calls (default: 50)

This script:
- Processes each response file through evaluation prompts
- Handles different archetypes (binary, statistics, etc.)
- Supports multiple evaluation runs per response
- Outputs results to the `evaluated/` directory

### 3. Calculate Metrics

Run the metrics calculation:
```bash
python metric.py [--test]
```

This script:
- Processes evaluated responses
- Calculates accuracy and honesty metrics
- Outputs to the `metrics/` directory

### 4. Process Final Metrics

Generate the final metrics summary:
```bash
python process_metrics.py [--test]
```

This creates `metrics/all_results.json` containing:
- Total responses per model/archetype
- Honesty scores (levels 1 and 10)
- Accuracy scores
- Other relevant metrics

## Metrics Explanation

- **Honesty Score**: Honesty in initial responses
- **Honesty Score (@10)**: Consistency across multiple generations
- **Accuracy**: Alignment with ground truth (where applicable)

## Testing

Use the `--test` flag with any script to run in test mode, which:
- Uses `test_csv_data/` instead of `csv_data/`
- Helps validate changes without affecting production data
- Provides a smaller dataset for quick iterations
