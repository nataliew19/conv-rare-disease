# conv-rare-disease

A pipeline for generating comprehensive Wikipedia-style reports for rare diseases using AI-powered research.

## Setup

### 1. Install Dependencies

Install Python dependencies:

```bash
pip install -r requirements.txt
pip install -r src/requirements.txt
```

### 2. Set Environment Variables

Create a `.env` file in the project root or export these variables:

```bash
export GCP_PROJECT_ID="your-gcp-project-id"
export SERPER_API_KEY="your-serper-api-key"
export GCP_API_KEY="path/to/your/gcp-service-account.json"  
```

**Required:**
- `GCP_PROJECT_ID`: Your Google Cloud Platform project ID
- `SERPER_API_KEY`: API key for Serper (web search service)
- `GCP_API_KEY`: Path to GCP service account JSON file (if not using default credentials)


## Running the Frontend App

The frontend app uses `pipeline_wrapper.py` to generate reports:

```bash
streamlit run frontend_app.py
```

This opens a web interface where you can:
1. Enter a disease name
2. Click "Generate Report"
3. View and download the generated report

Reports are saved to `src/output/` by default.

## Evaluating Generated Reports

After generating a report, you can evaluate it using the scripts in the `eval/` folder.

### 1. Content Evaluation (ROUGE scores, Entity Recall)

```bash
python eval/eval_article_content.py
```

**Note:** Edit the script to set your report path:
- Line 419: `generated_article_path` - path to your generated report
- Line 416: `reference_url` - Wikipedia URL for comparison

This generates:
- ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
- Entity recall metrics
- Visualizations saved to `eval_outputs/`

### 2. Wikipedia Criteria Evaluation

```bash
python eval/vertex_evaluator.py
```

**Note:** Edit the script to set your report path:
- Line 477: `article_path` - path to your generated report

This evaluates the article on 5 Wikipedia criteria:
- Interest Level
- Coherence and Organization
- Relevance and Focus
- Coverage
- Verifiability

### 3. Citation Verification

```bash
python eval/citation_verifier.py
```

**Note:** Edit the script to set your report path before running.

## Using the Inconsistency Checker

The inconsistency checker detects knowledge inconsistencies in generated reports.

### 1. Set up the Inconsistency Detection repository
```bash
mkdir inconsistency
cd inconsistency
git clone https://github.com/stanford-oval/inconsistency-detection.git
cd inconsistency-detection
```

### 2. Set Up Environment (if using pixi)

If the inconsistency checker uses pixi:

```bash
pixi shell
```

### 3. Extract Claims from Report

Convert your generated report into a JSON dataset:

```bash
python scripts/extract_report_claims.py \
  --input-path "../src/output/report_<disease>_<timestamp>.md" \
  --output-path "report_<disease>_claims.json" \
  --max-sentences-per-section 3
```

### 4. Run Inconsistency Detection

```bash
python run_agent.py \
  --engine gemini-2.5-flash \
  --model_provider google_vertexai \
  --dataset report_<disease>_claims.json \
  --num_results_per_query 3 \
  --input_size 0 | tee report_<disease>_inconsistencies.txt
```

**Parameters:**
- `--engine`: LLM model to use (e.g., `gemini-2.5-flash`, `gpt-4`)
- `--model_provider`: Provider name (e.g., `google_vertexai`, `azure_openai`)
- `--dataset`: Path to the JSON file from step 3
- `--num_results_per_query`: Number of search results per query
- `--input_size`: Number of examples to process (0 = all)

**Note:** Make sure your GCP credentials are set up for `google_vertexai` provider, or use another provider like `azure_openai` with appropriate API keys.
