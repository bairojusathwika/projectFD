import google.generativeai as genai
import os
import time

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is not set")

genai.configure(api_key=GOOGLE_API_KEY)

def transaction_to_text(txn: dict) -> str:
    return (
        f"Transaction ID {txn['transaction_id']}. "
        f"Amount {txn['amount']} dollars. "
        f"From account {txn['from_account']} to {txn['to_account']}. "
        f"Device {txn['device_id']}, IP {txn['ip_address']}. "
        f"Location {txn['location']['city']}, {txn['location']['country']}. "
        f"Timestamp {txn['timestamp']}."
    )

def create_embedding(text: str, retries: int = 3) -> list:
    for attempt in range(retries):
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text
            )
            return result["embedding"]
        except Exception as e:
            print(f"Embedding error (attempt {attempt+1}): {e}")
            time.sleep(2)

    raise RuntimeError("Failed to generate embedding after retries")
