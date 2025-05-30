from google.adk.agents import BaseAgent, Event
from google.adk.agents.invocation_context import InvocationContext
from typing import AsyncGenerator
import os # For accessing environment variables
from litellm import acompletion # LiteLLM for asynchronous OpenAI calls

class OpenAiAnalystAgent(BaseAgent):
    def __init__(self, name: str, model_name: str = "openai/gpt-4o-mini"): # LiteLLM model string
        super().__init__(name=name)
        self.model_name = model_name
        # OPENAI_API_KEY should be set as an environment variable for LiteLLM
        if not os.getenv("OPENAI_API_KEY"):
            print(f"[{self.name}]: WARNING - OPENAI_API_KEY environment variable not set. OpenAI calls may fail.")

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        agent_name = self.name
        print(f"[{agent_name}]: Analyzing data with OpenAI ({self.model_name}) via LiteLLM...")
        
        processed_data_json = ctx.session.state.get("processed_data_json")
        if not processed_data_json:
            error_msg = "Error: Processed data not found in state for OpenAI analysis."
            print(f"[{agent_name}]: {error_msg}")
            yield self.create_event(parts=[{"text": error_msg}], is_final_response=True)
            return

        # Constructing the prompt for the OpenAI model
        prompt = f"""You are a meticulous data auditor.
        Based on the following sales data (in JSON format), identify potential anomalies or outliers.
        Consider unusual spikes or dips in units sold or revenue that deviate from general patterns.
        Explain any unusual patterns you detect in a brief, clear manner.

        Sales Data:
        {processed_data_json}

        Anomaly Report:
        """

        try:
            # Asynchronously call OpenAI API using LiteLLM
            response = await acompletion(
                model=self.model_name,
                messages=[{"content": prompt, "role": "user"}],
                # Additional parameters like temperature can be added here
            )
            # Extract the analysis text from LiteLLM's response structure
            analysis_text = response.choices[0].message.content
            
            # Store OpenAI's analysis in session state
            ctx.session.state["openai_analysis"] = analysis_text
            yield self.create_event(parts=[{"text": f"OpenAI analysis complete. Insights stored."}])
        except Exception as e:
            error_msg = f"Error during OpenAI analysis via LiteLLM: {str(e)}"
            print(f"[{agent_name}]: {error_msg}")
            yield self.create_event(parts=[{"text": error_msg}], is_final_response=True) 