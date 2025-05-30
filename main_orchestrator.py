import asyncio
from dotenv import load_dotenv # Load environment variables from .env file
from google.adk.agents import SequentialAgent # For orchestration
from google.adk.events import Event
from google.adk.sessions import InMemorySessionService
from google.adk import Runner

# Import our custom agent classes
from agents.data_collector_agent import DataCollectorAgent
from agents.data_preprocessor_agent import DataPreprocessorAgent
from agents.google_llm_analyst_agent import GeminiAnalystAgent
from agents.openai_llm_analyst_agent import OpenAiAnalystAgent
from agents.visualization_agent import VisualizationAgent

# For self-contained demo, create sample data
import pandas as pd
import os
os.makedirs("data", exist_ok=True)
sample_df = pd.DataFrame({
    'Date': pd.to_datetime(['2023-01-01', '2023-01-15', '2023-02-01', '2023-02-12', '2023-03-05', '2023-03-20']),
    'Product_Name': ['AlphaSpark', 'BetaBolt', 'AlphaSpark', 'GammaGizmo', 'BetaBolt', 'AlphaSpark'],
    'Product_Category': ['Gadgets', 'Widgets', 'Gadgets', 'Gizmos', 'Widgets', 'Gadgets'],
    'Units_Sold': [100, 50, 120, 80, 45, 150],
    'Revenue': [1200.00, 750.00, 1440.00, 1250.00, 675.00, 1800.00]
})
sample_df.to_csv("data/sample_sales_data.csv", index=False)
# This ensures the tutorial is runnable out-of-the-box, a key for good learner experience.

# Load environment variables from .env file
load_dotenv()

async def main():
    # Initialize agents
    data_collector = DataCollectorAgent(name="DataCollector")
    preprocessor = DataPreprocessorAgent(name="Preprocessor")
    gemini_analyst = GeminiAnalystAgent(name="GeminiAnalyst", model_name="gemini-1.5-flash") 
    openai_analyst = OpenAiAnalystAgent(name="OpenAIAnalyst", model_name="gpt-4o-mini")
    visualizer = VisualizationAgent(name="Visualizer")

    # Define the pipeline using SequentialAgent
    pipeline = SequentialAgent(
        name="DataAnalyticsPipeline",
        sub_agents=[
            data_collector,
            preprocessor,
            gemini_analyst,
            openai_analyst,
            visualizer
        ]
    )

    # Set up the Runner and session service
    app_name = "ADKDataAnalyticsMAS"
    session_service = InMemorySessionService()
    runner = Runner(
        app_name=app_name,
        agent=pipeline,
        session_service=session_service
    )

    user_id = "tutorial_user"
    session_id = "demo-session-001"

    # Create the session before running the pipeline
    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id
    )

    print(f"Starting Data Analytics Multi-Agent Pipeline for user '{user_id}', session '{session_id}'...\n")
    
    # Try with None to see if pipeline can start without initial message
    print(f"📨 Starting pipeline without initial message")
    
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=None
    ):
        author = getattr(event, 'author', 'System')
        for part in getattr(event, 'parts', []):
            if getattr(part, 'text', None):
                print(f"[{author}]: {part.text}")
        if getattr(event, 'is_final_response', False):
            print("\nPipeline finished.")
            if getattr(event, 'parts', None):
                print("\n--- Final Analysis & Visualization Summary ---")
                for part in event.parts:
                    if getattr(part, 'text', None):
                        print(part.text)

    # Optionally, inspect the session state after the run
    final_session = await session_service.get_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    if final_session:
        print("\n📋 Final Session State:")
        for key, value in final_session.state.items():
            if isinstance(value, str) and len(value) > 200:
                print(f"  {key}: {value[:200]}... (truncated)")
            else:
                print(f"  {key}: {value}")
    else:
        print("Could not retrieve final session.")

if __name__ == "__main__":
    # Ensure API keys are set in your .env file or environment variables
    # GOOGLE_AI_API_KEY: Get from https://aistudio.google.com/
    # OPENAI_API_KEY: Get from https://platform.openai.com/account/api-keys
    asyncio.run(main())
