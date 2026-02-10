from kafka import KafkaConsumer
import json
import uuid
from qdrant_client import QdrantClient
from embedding_service import transaction_to_text, create_embedding

TOPIC = "fraud-transactions"
COLLECTION_NAME = "transaction_embeddings"

consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers="kafka:29092",
    group_id="qdrant-embedding-consumer",
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    value_deserializer=lambda x: json.loads(x.decode("utf-8"))
)

qdrant = QdrantClient(host="qdrant", port=6333)

print("Kafka → Qdrant consumer started and waiting for messages")

for msg in consumer:
    txn = msg.value

    try:
        text = transaction_to_text(txn)
        embedding = create_embedding(text)

        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                {
                    "id": txn["transaction_id"],  # deterministic ID
                    "vector": embedding,
                    "payload": txn
                }
            ]
        )

        print(f"Stored txn {txn['transaction_id']} in Qdrant")

    except Exception as e:
        print(f"Failed to process txn {txn.get('transaction_id')}: {e}")