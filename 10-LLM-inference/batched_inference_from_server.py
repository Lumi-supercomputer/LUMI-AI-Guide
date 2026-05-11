import json
import asyncio
import httpx
import argparse
import sys
import os
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# 1. Function to send requests asynchronously
async def get_response(model_name, client, user_prompt, sem):
    async with sem:
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful and concise AI assistant."},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=5000
            )
            return {
                "prompt": user_prompt,
                "response": response.choices[0].message.content.strip()
            }
        except Exception as e:
            # If one prompt fails, we don't want the whole batch to die
            print(f"Error processing prompt: {user_prompt[:30]}... -> {e}")
            return {
                "prompt": user_prompt,
                "ERROR": str(e)
            }

async def main():
    # 2. Parse the model name
    parser = argparse.ArgumentParser()
    parser.add_argument("MODEL", type=str, help="Name of the model as registered in vLLM")
    args = parser.parse_args()

    # 3. Read prompts (from https://huggingface.co/datasets/fka/prompts.chat except for the first 5)
    if not os.path.exists("prompts.txt"):
        print("Error: 'prompts.txt' not found.")
        sys.exit(1)
    with open("prompts.txt", "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

        prompts = prompts[:300] # truncate the number of prompts to 300

    # 4. Automatically find the socket file
    socket_path = f"/tmp/vllm-{os.environ.get('SLURM_JOB_ID')}.sock"

    if not os.path.exists(socket_path):
        print(f"Error: Socket not found at {socket_path}")
        sys.exit(1)

    transport = httpx.AsyncHTTPTransport(uds=socket_path)

    # 5. Connect to the socket and create a 'task' for every prompt
    sem=asyncio.Semaphore(256) # Set how many requests to send to the vLLM server at a time
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = AsyncOpenAI(
            base_url="http://localhost/v1", 
            api_key="token-ignored",
            http_client=http_client
        )

        print(f"--- Sending {len(prompts)} prompts ---")

        tasks = [get_response(args.MODEL, client, p, sem) for p in prompts]

        # asyncio.gather waits for all tasks to finish
        results = await tqdm.gather(*tasks, desc="Inference Progress")

    # 6. Save
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Done! Processed {len(results)} prompts.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInference cancelled by user.")
        sys.exit(0)
