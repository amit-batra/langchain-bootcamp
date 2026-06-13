from json import dumps
from typing import Any
import os, sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph

DEFAULT_MODEL_NAME: str = "google_genai:gemini-2.5-flash-lite"

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

def validate_environment_variables() -> tuple[str, str]:
    """Validates the API key and the model name from the environment file"""

    _ = load_dotenv()

    api_key: str | None = os.getenv("GOOGLE_API_KEY")
    if api_key is None:
        print("Unable to read the Google API key.")
        print("Please set the environment variable GOOGLE_API_KEY.")
        sys.exit(1)

    model_name: str | None = os.getenv("GOOGLE_MODEL_NAME")
    if model_name is None:
        print("Unable to read the environment variable GOOGLE_MODEL_NAME.")
        print(f"Defaulting to {DEFAULT_MODEL_NAME}.")
        model_name = DEFAULT_MODEL_NAME

    return api_key, model_name

def main() -> None:
    model_name: str
    _, model_name = validate_environment_variables()

    agent: CompiledStateGraph = create_agent(
        model=model_name,
        tools=[get_weather],
        system_prompt="You are a helpful assistant",
    )

    result: dict[str, Any] | Any = agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": "What's the weather in San Francisco?"
                }
            ]
        }
    )
    print(dumps(result["messages"][-1].content_blocks, indent=2))

if __name__ == "__main__":
    main()