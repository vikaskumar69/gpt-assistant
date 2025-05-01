import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the key
openai_key = os.getenv("OPENAI_API_KEY")

# Print to check (only during dev!)
print(f"OpenAI Key Loaded: {openai_key}...")  # Don't print full key in real apps
