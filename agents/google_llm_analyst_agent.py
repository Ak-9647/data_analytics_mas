from google.adk.agents import BaseAgent, Event
from google.adk.agents.invocation_context import InvocationContext
from typing import AsyncGenerator
# Ensure google-cloud-aiplatform is installed for Vertex AI SDK
from vertexai.generative_models import GenerativeModel, Part # Using the newer GenerativeModel API

class GeminiAnalystAgent(BaseAgent):
    def __init__(self, name: str, model_name: str = "gemini-1.5-flash-001"):
        super().__init__(name=name)
        self.model_name = model_name
        # Initialize the GenerativeModel client.
        # ADC (Application Default Credentials) should be set up via `gcloud auth application-default login`.
        # Project and location can be picked up from environment or set explicitly.
        try:
            self.model = GenerativeModel(self.model_name)
        except Exception as e:
            print(f"[{self.name}]: Error initializing Gemini model ({self.model_name}): {e}. Ensure Vertex AI is configured.")
            self.model = None

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        agent_name = self.name
        print(f"[{agent_name}]: Analyzing data with Google Gemini ({self.model_name})...")

        if not self.model:
            error_msg = "Gemini model not initialized. Check Vertex AI setup."
            print(f"[{agent_name}]: {error_msg}")
            yield self.create_event(parts=[{"text": error_msg}], is_final_response=True)
            return

        processed_data_json = ctx.session.state.get("processed_data_json")
        if not processed_data_json:
            error_msg = "Error: Processed data not found in state for Gemini analysis."
            print(f"[{agent_name}]: {error_msg}")
            yield self.create_event(parts=[{"text": error_msg}], is_final_response=True)
            return

        # Constructing the prompt for Gemini
        prompt = f"""You are an expert data analyst.
        Analyze the following sales data, provided in JSON format, to identify key trends.
        Focus specifically on:
        1. Monthly revenue changes: Describe any significant increases or decreases.
        2. Top-performing products: Identify products with high revenue or sales volume.
        Provide a concise, bullet-pointed summary of your findings.

        Sales Data:
        {processed_data_json}

        Your Analysis:
        """
        
        try:
            # Asynchronously generate content using the Vertex AI SDK
            gemini_response = await self.model.generate_content_async(prompt)
            
            # Extract the text from the response
            analysis_text = ""
            if hasattr(gemini_response, 'candidates') and gemini_response.candidates:
                for candidate in gemini_response.candidates:
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            if hasattr(part, 'text'):
                                analysis_text += part.text
            if not analysis_text: # Handle cases where response might be empty or in unexpected format
                analysis_text = "Gemini analysis did not return text."

            # Store Gemini's analysis in session state
            ctx.session.state["gemini_analysis"] = analysis_text
            yield self.create_event(parts=[{"text": f"Gemini analysis complete. Insights stored."}])
        except Exception as e:
            error_msg = f"Error during Gemini analysis: {str(e)}"
            print(f"[{agent_name}]: {error_msg}")
            yield self.create_event(parts=[{"text": error_msg}], is_final_response=True) 