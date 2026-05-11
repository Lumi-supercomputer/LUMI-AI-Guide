import httpx
import sys
import os
import argparse
from openai import OpenAI

def main():
    # 1. Automatically find the socket file and set up the connection
    socket_path = f"/tmp/vllm-{os.environ.get("SLURM_JOB_ID")}.sock"
    if not os.path.exists(socket_path):
        print(f"Error: Socket not found at {socket_path}")
        sys.exit(1)

    transport = httpx.HTTPTransport(uds=socket_path)

    client = OpenAI(
        base_url="http://localhost/v1",
        api_key="token-ignored",
        http_client=httpx.Client(transport=transport)
    )

    # 2. Parse the model name
    parser = argparse.ArgumentParser()
    parser.add_argument("MODEL", type=str, help="Name of the model as registered in vLLM")
    args = parser.parse_args()

    # 3. Initialize "messages" list which is the LLM's 'memory' of the conversation
    messages = [
        {"role": "system", "content": "You are a helpful and concise AI assistant."}
    ]

    print("--- Chat Started (Type 'quit' or 'exit' to stop) ---")

    # 4. Start the chat
    while True:
        # Get input from the user
        user_input = input("\nUser: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        # Add the user's message to the memory
        messages.append({"role": "user", "content": user_input})

        # Send a request to the LLM
        # You can change hyperparameters here
        response = client.chat.completions.create(
            model=args.MODEL,
            messages=messages, # Sends the full history
            #max_completion_tokens=5000, # Uncomment to set the max length of the LLM's output
            #temperature=0.6, # Uncomment to adjust temperature. Higher = more creative, lower = more focused
            stream=True # Sends tokens as soon as the model generates them
        )

        print("LLM Response: ", end="", flush=True)
        response_text = "" # stores the streamed text from the LLM

        # Handle the stream and show the text
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)
                response_text += content

        # Add the LLM's response to the memory
        messages.append({"role": "assistant", "content": response_text})
        print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInference cancelled by user.")
        sys.exit(0)
