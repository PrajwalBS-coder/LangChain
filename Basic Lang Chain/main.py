import os

from fastapi import FastAPI
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from pydantic import BaseModel
import psycopg
from dotenv import load_dotenv


load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://192.168.1.3:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")


def get_database_connection() -> psycopg.Connection:
    """Create a PostgreSQL connection from the local environment variables."""
    required_settings = ("DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT", "DB_NAME")
    settings = {setting: os.getenv(setting) for setting in required_settings}
    missing_settings = [setting for setting, value in settings.items() if not value]

    if missing_settings:
        missing = ", ".join(missing_settings)
        raise RuntimeError(f"Missing database settings in .env: {missing}")

    return psycopg.connect(
        user=settings["DB_USER"],
        password=settings["DB_PASSWORD"],
        host=settings["DB_HOST"],
        port=settings["DB_PORT"],
        dbname=settings["DB_NAME"],
    )


def save_agent_weather_response(
    message: str,
    response: str,
    input_tokens: int,
    output_tokens: int,
) -> None:
    """Persist an agent weather response in PostgreSQL."""
    with get_database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_weather_responses (
                    id BIGSERIAL PRIMARY KEY,
                    message TEXT NOT NULL,
                    response TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor.execute(
                """
                INSERT INTO agent_weather_responses
                    (message, response, input_tokens, output_tokens)
                VALUES (%s, %s, %s, %s)
                """,
                (message, response, input_tokens, output_tokens),
            )


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)

agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are a helpful assistant. Use the weather tool when asked about weather.",
)


app = FastAPI(title="Weather API")


class WeatherRequest(BaseModel):
    city: str


class AgentWeatherRequest(BaseModel):
    message: str


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/weather")
def get_weather_api(city: str) -> dict[str, str]:
    return {"city": city, "weather": get_weather(city)}


@app.post("/weather")
def post_weather_api(payload: WeatherRequest) -> dict[str, str]:
    return {"city": payload.city, "weather": get_weather(payload.city)}


@app.post("/agent/weather")
def agent_weather_api(payload: AgentWeatherRequest) -> dict[str, int | str]:
    result = agent.invoke({"messages": [{"role": "user", "content": payload.message}]})

    last_message = result["messages"][-1]
    response = getattr(last_message, "content", "")
    usage = getattr(last_message, "usage_metadata", {}) or {}

    if isinstance(response, list):
        response_text = " ".join(
            block.text if hasattr(block, "text") else str(block) for block in response
        )
    else:
        response_text = str(response)

    response_data = {
        "message": payload.message,
        "response": response_text,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
    }
    save_agent_weather_response(**response_data)

    return response_data


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
