import os
from dotenv import load_dotenv
from groq import Groq

# Load .env file
load_dotenv()

# Read API key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY not found. Set it in .env or environment variables.")

# Create Groq client
client = Groq(api_key=api_key)

MODEL_NAME = "llama-3.3-70b-versatile"  # you can change model here if needed


def chat_loop():
    print("💬 Simple Groq LLM Chatbot")
    print("Type 'exit' or 'quit' to stop.\n")

    # Conversation history (system + user + assistant)
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant. Answer clearly and concisely.",
        }
    ]

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Bot: Bye! 👋")
            break

        if not user_input:
            continue

        # Add user message
        messages.append({"role": "user", "content": user_input})

        try:
            # Call Groq Chat Completions API
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
            )
        except Exception as e:
            print(f"Error calling Groq API: {e}")
            # Remove last user message so it doesn't break future calls
            messages.pop()
            continue

        # Extract assistant reply
        reply = response.choices[0].message.content.strip()
        print(f"Bot: {reply}\n")

        # Add assistant reply back into history
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    chat_loop()
