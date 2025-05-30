# Data Analytics Multi-Agent System (ADK)

This project demonstrates a modular, multi-agent data analytics pipeline using Google ADK, Vertex AI (Gemini), OpenAI (via LiteLLM), and Python data science tools. It is based on the step-by-step coding demo from the referenced video.

## Project Structure

```
├── main_orchestrator.py         # Main pipeline orchestrator
├── agents/                      # All agent classes
│   ├── data_collector_agent.py
│   ├── data_preprocessor_agent.py
│   ├── google_llm_analyst_agent.py
│   ├── openai_llm_analyst_agent.py
│   └── visualization_agent.py
├── data/                        # Input data (sample CSV auto-generated)
├── results/                     # Output plots/images
├── requirements.txt             # Python dependencies
├── .env.example                 # Example environment variables
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## Setup Instructions

1. **Clone the repository:**
   ```sh
   git clone https://github.com/Ak-9647/data_analytics_mas.git
   cd data_analytics_mas
   ```

2. **Install dependencies:**
   ```sh
   python3 -m pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   - Copy `.env.example` to `.env` and add your OpenAI API key:
     ```sh
     cp .env.example .env
     # Edit .env and set OPENAI_API_KEY=your-openai-api-key
     ```
   - Authenticate with Google Cloud for Vertex AI:
     ```sh
     gcloud auth application-default login
     # Make sure Vertex AI API is enabled in your GCP project
     ```

4. **Run the pipeline:**
   ```sh
   python3 main_orchestrator.py
   ```

## What This Does
- **DataCollectorAgent:** Loads sample sales data from CSV.
- **DataPreprocessorAgent:** Cleans and preprocesses the data.
- **GeminiAnalystAgent:** Uses Google Vertex AI Gemini to analyze trends.
- **OpenAiAnalystAgent:** Uses OpenAI (via LiteLLM) to find anomalies.
- **VisualizationAgent:** Generates and saves plots, embedding LLM insights in chart titles.

## Output
- Console output shows progress and LLM findings.
- Plots are saved in the `results/` directory.

## Notes
- The sample data is auto-generated for demo purposes.
- Ensure your Google Cloud project has billing enabled and Vertex AI API is active.
- The `.env` file is excluded from git for security.

## Requirements
- Python 3.10+
- Google Cloud CLI (`gcloud`)
- OpenAI API key

## License
MIT 