import asyncio
import json
import time
import httpx


BASE_URL = "http://host.docker.internal:8080"   # change if needed
MODEL = "llama"

def chat_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/v1/chat/completions"


async def one_call(prompt: str, slot: int) -> dict:
    url = chat_url(BASE_URL)
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 32,
        "stream": False,
        "id_slot": slot,  # top-level
    }

    async with httpx.AsyncClient(timeout=120) as client:
        t0 = time.time()
        r = await client.post(url, json=payload)
        dt = time.time() - t0

        out = {
            "slot": slot,
            "status": r.status_code,
            "elapsed_sec": round(dt, 3),
        }

        # Print server error body if not 2xx
        if r.status_code >= 400:
            out["error_text"] = r.text
            return out

        # Normal JSON response
        try:
            data = r.json()
        except Exception:
            out["error_text"] = f"Non-JSON response: {r.text[:500]}"
            return out

        out["response_id"] = data.get("id")
        out["finish_reason"] = (data.get("choices") or [{}])[0].get("finish_reason")
        out["content"] = (data.get("choices") or [{}])[0].get("message", {}).get("content")
        out["raw"] = data
        return out


async def try_slots_endpoint():
    # Some llama-server builds expose /slots or /v1/slots; many don't.
    candidates = [
        BASE_URL.rstrip("/") + "/slots",
        BASE_URL.rstrip("/") + "/v1/slots",
    ]
    async with httpx.AsyncClient(timeout=10) as client:
        for url in candidates:
            try:
                r = await client.get(url)
                if r.status_code < 400:
                    return {"url": url, "status": r.status_code, "json": r.json()}
            except Exception:
                pass
    return None


async def main():
    print("Chat URL:", chat_url(BASE_URL))

    slots_info = await try_slots_endpoint()
    if slots_info:
        print("\nSlots endpoint OK:", slots_info["url"])
        print(json.dumps(slots_info["json"], indent=2)[:2000])
    else:
        print("\nSlots endpoint not available (this is OK).")

    # Run two requests concurrently to *force* 2-slot behavior if supported.
    prompt0 = "test slot 0: reply with 'hi from slot 0'"
    prompt1 = "test slot 1: reply with 'hi from slot 1'"

    res0, res1 = await asyncio.gather(
        one_call(prompt0, 0),
        one_call(prompt1, 1),
    )

    print("\n--- result slot 0 ---")
    print(json.dumps(res0, indent=2)[:4000])

    print("\n--- result slot 1 ---")
    print(json.dumps(res1, indent=2)[:4000])

    # If both succeeded, also print just the assistant text:
    if res0.get("status") == 200 and res1.get("status") == 200:
        print("\nAssistant texts:")
        print("slot0:", res0.get("content"))
        print("slot1:", res1.get("content"))


if __name__ == "__main__":
    asyncio.run(main())
