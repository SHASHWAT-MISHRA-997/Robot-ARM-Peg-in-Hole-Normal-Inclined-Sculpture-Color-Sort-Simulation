import json
import sys

from groq import Groq


def main():
    payload = json.loads(sys.stdin.read() or "{}")
    client = Groq(api_key=payload["api_key"], timeout=float(payload.get("timeout", 20.0)))
    completion = client.chat.completions.create(
        model=payload["model"],
        messages=payload["messages"],
        temperature=float(payload.get("temperature", 0.45)),
        max_tokens=int(payload.get("max_tokens", 1500)),
        top_p=float(payload.get("top_p", 0.95)),
        frequency_penalty=float(payload.get("frequency_penalty", 0.55)),
        presence_penalty=float(payload.get("presence_penalty", 0.15)),
    )
    reply = (completion.choices[0].message.content or "").strip()
    print(json.dumps({"reply": reply or "No response text returned."}, ensure_ascii=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": str(exc)[:1000]}, ensure_ascii=True))
        sys.exit(1)
