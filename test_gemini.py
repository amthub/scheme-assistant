import os

from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="pm kisan mein kitna paisa milta hai? Hinglish mein sirf ek line mein jawab do.",
)
print(response.text)