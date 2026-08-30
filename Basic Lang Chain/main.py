from fastapi import FastAPI
from langchain.agents import create_agent
from pydantic import BaseModel


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_agent(
    model="ollama:mistral",
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
def agent_weather_api(payload: AgentWeatherRequest) -> dict[str, str]:
    result = agent.invoke({"messages": [{"role": "user", "content": payload.message}]})

    last_message = result["messages"][-1]
    response = getattr(last_message, "content", "")

    if isinstance(response, list):
        return {
            "message": payload.message,
            "response": " ".join(
                block.text if hasattr(block, "text") else str(block) for block in response
            ),
        }

    return {"message": payload.message, "response": str(response)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
