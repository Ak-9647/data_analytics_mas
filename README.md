# Data Analytics Multi-Agent System (MAS)

A powerful multi-agent system built with Google ADK that performs end-to-end data analytics using AI agents.

## Architecture

The system consists of 5 specialized agents working in sequence:

1. **DataCollectorAgent** - Loads and ingests data from CSV files
2. **DataPreprocessorAgent** - Cleans and prepares data for analysis  
3. **GeminiAnalystAgent** - Performs trend analysis using Google's Gemini AI
4. **OpenAiAnalystAgent** - Conducts anomaly detection using OpenAI's GPT models
5. **VisualizationAgent** - Creates charts and visualizations with AI insights

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```bash
# Google AI Studio API Key (get from https://aistudio.google.com/)
GOOGLE_AI_API_KEY=your_google_ai_api_key_here

# OpenAI API Key (get from https://platform.openai.com/account/api-keys)
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Run the System
```bash
python main_orchestrator.py
```

## Features

- ✅ **Multi-Agent Architecture** - Specialized agents for different analytics tasks
- ✅ **AI-Powered Analysis** - Real insights from Gemini and OpenAI models  
- ✅ **Professional Visualizations** - Charts with embedded AI insights
- ✅ **Robust Error Handling** - Graceful fallbacks and comprehensive logging
- ✅ **Extensible Design** - Easy to add new agents or data sources

## Output

The system generates:
- **Analysis Reports** - Trend analysis and anomaly detection
- **Visualizations** - Professional charts in the `results/` directory
- **Session State** - Complete pipeline execution logs

## Sample Data

The system includes sample sales data for demonstration. In production, replace with your own data sources.

## API Requirements

- **Google AI Studio API** - Simple API key approach (no GCP setup required)
- **OpenAI API** - Standard OpenAI API access

No complex cloud authentication or Vertex AI setup needed! 