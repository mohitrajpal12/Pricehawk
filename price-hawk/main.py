import os
import asyncio
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, function_tool

load_dotenv()

# --- STEP 1: Define the Free Search Tool ---
@function_tool
def web_search(query: str):
    """
    Searches the web for product prices and store links.
    """
    print(f"🔎 Searching for: {query}...")
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=5)]
        # Format the results into a simple string for the Agent to read
        formatted_results = "\n".join([f"- {r['title']}: {r['href']}" for r in results])
        return formatted_results

async def main():
    # --- STEP 2: Setup Gemini via OpenAI SDK ---
    # We use the Async client because the Agents SDK is built on top of it.
    client = AsyncOpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    gemini_model = OpenAIChatCompletionsModel(
        model="gemini-1.5-flash", # High rate limit for free tier
        openai_client=client
    )

    # --- STEP 3: Define the Scout Agent ---
    scout_agent = Agent(
        name="Price Scout",
        instructions=(
            "You are a shopping expert. Your goal is to find the most relevant "
            "store links for a specific product. Use the web_search tool to find "
            "at least 5 different store URLs (like Amazon, Flipkart, etc.)."
        ),
        model=gemini_model,
        tools=[web_search]
    )

    # --- STEP 4: Run the Agent ---
    print("🚀 Starting the Price Scout...")
    result = await Runner.run(scout_agent, "Find the best store links for boAt Rockerz 512 ANC in India.")
    
    print("\n--- FINAL SCOUT REPORT ---")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())