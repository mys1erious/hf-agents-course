from huggingface_hub import login
from langchain.agents import load_tools
from smolagents import (
    ToolCallingAgent,
    DuckDuckGoSearchTool,
    HfApiModel,
    tool,
    CodeAgent,
    Tool,
    load_tool,
)

from config import settings


def run_simple_tool():
    agent = ToolCallingAgent(tools=[DuckDuckGoSearchTool()], model=HfApiModel())
    agent.run(
        "Search for the best music recommendations for a party at the Wayne's mansion."
    )


@tool
def catering_service_tool(query: str) -> str:
    """
    This tool returns the highest-rated catering service in Gotham City.

    Args:
        query: A search term for finding catering services.
    """
    # Example list of catering services and their ratings
    services = {
        "Gotham Catering Co.": 4.9,
        "Wayne Manor Catering": 4.8,
        "Gotham City Events": 4.7,
    }

    # Find the highest rated catering service (simulating search query filtering)
    best_service = max(services, key=services.get)

    return best_service


def run_catering_service():
    agent = CodeAgent(tools=[catering_service_tool], model=HfApiModel())
    # Run the agent to find the best catering service
    result = agent.run(
        "Can you give me the name of the highest-rated catering service in Gotham City?"
    )
    print(result)


class SuperheroPartyThemeTool(Tool):
    name = "superhero_party_theme_generator"
    description = """
    This tool suggests creative superhero-themed party ideas based on a category.
    It returns a unique party theme idea."""

    inputs = {
        "category": {
            "type": "string",
            "description": "The type of superhero party (e.g., 'classic heroes', 'villain masquerade', 'futuristic Gotham').",
        }
    }

    output_type = "string"

    def forward(self, category: str):
        themes = {
            "classic heroes": "Justice League Gala: Guests come dressed as their favorite DC heroes with themed cocktails like 'The Kryptonite Punch'.",
            "villain masquerade": "Gotham Rogues' Ball: A mysterious masquerade where guests dress as classic Batman villains.",
            "futuristic Gotham": "Neo-Gotham Night: A cyberpunk-style party inspired by Batman Beyond, with neon decorations and futuristic gadgets.",
        }

        return themes.get(
            category.lower(),
            "Themed party idea not found. Try 'classic heroes', 'villain masquerade', or 'futuristic Gotham'.",
        )


def run_party_theme():
    party_theme_tool = SuperheroPartyThemeTool()
    agent = CodeAgent(tools=[party_theme_tool], model=HfApiModel())
    # Run the agent to generate a party theme idea
    result = agent.run(
        "What would be a good superhero party idea for a 'villain masquerade' theme?"
    )
    print(result)


def run_load_tool():
    image_generation_tool = load_tool("m-ric/text-to-image", trust_remote_code=True)
    agent = CodeAgent(tools=[image_generation_tool], model=HfApiModel())
    img = agent.run(
        "Generate an image of a luxurious superhero-themed party at Wayne Manor with made-up superheros."
    )
    img.save("./superhero-themed.jpg")


def run_tool_from_space():
    image_generation_tool = Tool.from_space(
        "black-forest-labs/FLUX.1-schnell",
        name="image_generator",
        description="Generate an image from a prompt",
    )
    model = HfApiModel("Qwen/Qwen2.5-Coder-32B-Instruct")
    agent = CodeAgent(tools=[image_generation_tool], model=model)
    agent.run(
        "Improve this prompt, then generate an image of it.",
        additional_args={
            "user_prompt": "A grand superhero-themed party at Wayne Manor, with Alfred overseeing a luxurious gala"
        },
    )


def run_tool_from_langchain():
    model = HfApiModel("Qwen/Qwen2.5-Coder-32B-Instruct")
    search_tool = Tool.from_langchain(load_tools(["serpapi"])[0])
    agent = CodeAgent(tools=[search_tool], model=model)
    agent.run(
        "Search for luxury entertainment ideas for a superhero-themed event, such as live performances and interactive experiences."
    )


if __name__ == "__main__":
    login(token=settings.HF_TOKEN)

    # run_simple_tool()
    # run_catering_service()
    # run_party_theme()
    # run_load_tool()
    # run_tool_from_space()
    run_tool_from_langchain()
