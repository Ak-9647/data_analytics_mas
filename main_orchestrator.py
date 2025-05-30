import asyncio
from google.adk.agents import SequentialAgent # For orchestration
from google.adk.agents.invocation_context import InvocationContext # For managing run context and state

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

async def main():
    # Initialize agents
    data_collector = DataCollectorAgent(name="DataCollector")
    preprocessor = DataPreprocessorAgent(name="Preprocessor")
    # Specify model names; ensure these are valid for your setup
    gemini_analyst = GeminiAnalystAgent(name="GeminiAnalyst", model_name="gemini-1.5-flash-001") 
    openai_analyst = OpenAiAnalystAgent(name="OpenAIAnalyst", model_name="openai/gpt-4o-mini") # LiteLLM format

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

    # Create an InvocationContext to manage state across the pipeline
    ctx = InvocationContext.create(user_id="tutorial_user", app_name="ADKDataAnalyticsMAS")
    
    print("Starting Data Analytics Multi-Agent Pipeline...\n")
    # Execute the pipeline
    async for event in pipeline.run_async(ctx):
        author = event.author if event.author else "System" # Agent name or system
        for part in event.parts: # Events can have multiple parts (text, function_call, etc.)
            if part.text:
                 print(f"[{author}]: {part.text}")
            # In more complex scenarios, you might inspect event.actions for state changes, etc.
    
    print("\nPipeline finished.")
    print("\n--- Final Analysis & Visualization Summary ---")
    if ctx.session.state.get("gemini_analysis"):
        print("\nGemini Analyst Findings:")
        print(ctx.session.state["gemini_analysis"])
    if ctx.session.state.get("openai_analysis"):
        print("\nOpenAI Analyst Findings:")
        print(ctx.session.state["openai_analysis"])
    if ctx.session.state.get("visualization_paths"):
        print("\nVisualizations Generated:")
        for path in ctx.session.state["visualization_paths"]:
            print(f"Plot saved at: {path}")

if __name__ == "__main__":
    # Ensure API keys and cloud configurations are set in your environment
    # e.g., os.environ["OPENAI_API_KEY"] = "your-key"
    # For Google Cloud, run `gcloud auth application-default login`
    asyncio.run(main())
