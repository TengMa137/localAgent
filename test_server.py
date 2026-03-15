import asyncio
from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request_stream
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIChatModel(
    model_name="llama",
    provider=OpenAIProvider(
        base_url="http://host.docker.internal:8080/v1",
        api_key="no-api-key",
    ),
)

async def main():
    async with model_request_stream(
        model,
        [ModelRequest.user_text_prompt("Elaborate on the 10th point please.")],
        model_settings={
            "extra_body": {"id_slot": 1}, 
        },
    ) as stream:
        async for event in stream:
            # Handle streaming deltas
            if hasattr(event, "delta") and event.delta.part_delta_kind == "text":
                print(event.delta.content_delta, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())
