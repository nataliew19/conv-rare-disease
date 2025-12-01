# Frontend Setup Guide

This guide will help you set up a simple frontend to run the rare disease report generation pipeline.

## First Steps

### 1. Install Dependencies

Make sure you have Streamlit installed:

```bash
pip install streamlit
```

Or add it to your requirements.txt:
```bash
echo "streamlit>=1.28.0" >> requirements.txt
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a `.env` file in the project root (if you don't have one) with the following variables:

```bash
GCP_PROJECT_ID=your-gcp-project-id
GCP_API_KEY=/path/to/service-account.json  # Optional if using gcloud auth
VERTEX_LOCATION=us-central1
VERTEX_MODEL=gemini-1.5-pro-002
SERPER_API_KEY=your-serper-api-key
```

**Note**: You can also use `gcloud auth application-default login` instead of a service account file.

### 3. Run the Frontend

From the project root directory, run:

```bash
streamlit run frontend_app.py
```

This will start a local web server (usually at `http://localhost:8501`).

### 4. Use the Frontend

1. Open your browser to the URL shown in the terminal (usually `http://localhost:8501`)
2. Enter a disease name (e.g., "Duchenne Muscular Dystrophy")
3. Click "Generate Report"
4. Wait for the pipeline to complete (this can take 5-10 minutes)
5. View and download the generated report

## What the Pipeline Does

The pipeline runs through these steps:

1. **Wikipedia Discovery**: Searches Wikipedia for relevant articles about the disease
2. **Guideline Generation**: Creates a research guideline based on Wikipedia findings
3. **RAG Evidence Gathering**: Uses Retrieval-Augmented Generation to gather evidence from the internet
4. **Research Plan**: Generates a step-by-step research plan
5. **Report Generation**: Creates a comprehensive Wikipedia-style report

## Output Files

All generated files are saved to `src/output/` by default (or the directory you specify):

- `guideline_<disease>_<timestamp>.json` - Guideline in JSON format
- `guideline_<disease>_<timestamp>.md` - Guideline in Markdown format
- `rag_response.json` - RAG evidence gathered
- `report_<disease>_<timestamp>.md` - Final generated report

## Troubleshooting

### Import Errors

If you get import errors, make sure you're running from the project root directory and that all dependencies are installed.

### GCP Authentication Errors

- Make sure `GCP_PROJECT_ID` is set
- Either set `GCP_API_KEY` to a service account JSON file path, or run `gcloud auth application-default login`

### Missing API Keys

- `SERPER_API_KEY` is required for internet search. Get one at https://serper.dev

### Long Processing Time

The pipeline can take 5-10 minutes because it:
- Makes multiple LLM API calls
- Searches Wikipedia iteratively
- Performs RAG retrieval and processing
- Generates comprehensive reports

Be patient! Progress updates will appear in the status area.

## Next Steps

Once you have the basic frontend working, you can:

- Customize the UI in `frontend_app.py`
- Modify pipeline parameters in `src/pipeline_wrapper.py`
- Add more features like report history, export formats, etc.

