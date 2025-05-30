from google.adk.agents import BaseAgent, Event
from google.adk.agents.invocation_context import InvocationContext
import pandas as pd
from typing import AsyncGenerator

class DataPreprocessorAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        agent_name = self.name
        print(f"[{agent_name}]: Preprocessing data...")
        
        # Retrieve raw data from session state, put there by the DataCollectorAgent
        raw_data_json = ctx.session.state.get("raw_data_json")
        
        if not raw_data_json:
            error_msg = "Error: Raw data not found in state for preprocessing."
            print(f"[{agent_name}]: {error_msg}")
            yield self.create_event(parts=[{"text": error_msg}], is_final_response=True)
            return # Stop if critical data is missing

        try:
            df = pd.read_json(raw_data_json, orient="records")
            
            # Example preprocessing steps:
            # Handle missing values (simple fill with 0 for numeric columns for demo)
            for col in df.select_dtypes(include=['number']).columns:
                df[col].fillna(0, inplace=True)
            # Ensure 'Date' column is in datetime format
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            # Add more relevant preprocessing steps here based on actual data needs
            # (e.g., removing duplicates, feature engineering, text cleaning if applicable)

            # Store the processed data back into session state for the next agents
            ctx.session.state["processed_data_json"] = df.to_json(orient="records")
            yield self.create_event(parts=[{"text": "Data preprocessing complete. Processed data stored in state."}])
        except Exception as e:
            error_msg = f"Error during data preprocessing: {str(e)}"
            print(f"[{agent_name}]: {error_msg}")
            yield self.create_event(parts=[{"text": error_msg}], is_final_response=True) 