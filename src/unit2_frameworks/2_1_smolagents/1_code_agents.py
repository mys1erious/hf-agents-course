import os

from huggingface_hub import login
from config import settings
from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    HfApiModel,
    tool,
    Tool,
    VisitWebpageTool,
)
from opentelemetry.sdk.trace import TracerProvider

from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor


def run_search_music():
    agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=HfApiModel())
    agent.run(
        "Search for the best music recommendations for a party at the Wayne's mansion."
    )


@tool
def suggest_menu(occasion: str) -> str:
    """
    Suggests a menu based on the occasion.
    Args:
        occasion: The type of occasion for the party.
    """
    if occasion == "casual":
        return "Pizza, snacks, and drinks."
    elif occasion == "formal":
        return "3-course dinner with wine and dessert."
    elif occasion == "superhero":
        return "Buffet with high-energy and healthy food."
    else:
        return "Custom menu for the butler."


def run_suggest_menu():
    agent = CodeAgent(tools=[suggest_menu], model=HfApiModel())
    agent.run("Prepare a formal menu for the party.")


def run_prep_time():
    agent = CodeAgent(
        tools=[], model=HfApiModel(), additional_authorized_imports=["datetime"]
    )
    agent.run(
        """
        Alfred needs to prepare for the party. Here are the tasks:
        1. Prepare the drinks - 30 minutes
        2. Decorate the mansion - 60 minutes
        3. Set up the menu - 45 minutes
        3. Prepare the music and playlist - 45 minutes

        If we start right now, at what time will the party be ready?
        """
    )


def run_full_flow():
    agent = CodeAgent(
        tools=[DuckDuckGoSearchTool(), suggest_menu],
        model=HfApiModel(),
        additional_authorized_imports=["datetime"],
    )
    # agent.run(
    #     "Search for the best music recommendations for a party at the Wayne's mansion."
    # )
    # agent.run("Prepare a formal menu for the party.")
    # agent.run(
    #     """
    #     Alfred needs to prepare for the party. Here are the tasks:
    #     1. Prepare the drinks - 30 minutes
    #     2. Decorate the mansion - 60 minutes
    #     3. Set up the menu - 45 minutes
    #     3. Prepare the music and playlist - 45 minutes
    #
    #     If we start right now, at what time will the party be ready?
    #     """
    # )

    publish_agent(agent)


def publish_agent(agent):
    agent.push_to_hub(f"{settings.HF_USERNAME}/AlfredAgent")

    alfred_agent = agent.from_hub(
        f"{settings.HF_USERNAME}/AlfredAgent", trust_remote_code=True
    )
    alfred_agent.run(
        "Give me the best playlist for a party at Wayne's mansion. The party idea is a 'villain masquerade' theme"
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


def run_hf_alfred_agent():
    agent = CodeAgent(
        tools=[
            DuckDuckGoSearchTool(),
            VisitWebpageTool(),
            suggest_menu,
            catering_service_tool,
            SuperheroPartyThemeTool(),
        ],
        model=HfApiModel(),
        max_steps=10,
        verbosity_level=2,
    )

    agent.run(
        "Give me best playlist for a party at the Wayne's mansion. The party idea is a 'villain masquerade' theme"
    )


def run_telemetry():
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = settings.OTEL_EXPORTER_OTLP_ENDPOINT
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = settings.OTEL_EXPORTER_OTLP_HEADERS

    trace_provider = TracerProvider()
    trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))

    SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)

    agent = CodeAgent(tools=[], model=HfApiModel())
    alfred_agent = agent.from_hub(
        f"{settings.HF_USERNAME}/AlfredAgent", trust_remote_code=True
    )
    alfred_agent.run(
        "Give me the best playlist for a party at Wayne's mansion. The party idea is a 'villain masquerade' theme"
    )


if __name__ == "__main__":
    login(token=settings.HF_TOKEN)

    # run_search_music()
    # run_suggest_menu()
    # run_prep_time()
    # run_full_flow()
    # run_hf_alfred_agent()
    run_telemetry()
