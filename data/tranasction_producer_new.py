import requests
import json
import time
from kafka import KafkaProducer
from kafka.errors import KafkaError

# 🔹 Host machine generator (WSL/Docker-safe)
API_URL = "http://generator:8041/stream"

# 🔹 Kafka topic
KAFKA_TOPIC = "fraud-transactions"

# 🔹 Kafka service name (Docker DNS)
KAFKA_BOOTSTRAP = "192.168.8.180:9092"


def create_producer():
    while True:
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                linger_ms=10,
                batch_size=16384,
                retries=5,
                acks="all"
            )
            print("✅ Connected to Kafka")
            return producer
        except Exception as e:
            print("❌ Kafka connection failed, retrying...", e)
            time.sleep(3)


def stream_transactions():
    producer = create_producer()
    session = requests.Session()

    sent_count = 0
    last_log = time.time()

    while True:
        try:
            print(f"🔌 Connecting to generator: {API_URL}")
            with session.get(API_URL, stream=True, timeout=15) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line:
                        continue

                    try:
                        data = json.loads(line.decode("utf-8"))
                        producer.send(KAFKA_TOPIC, data)
                        sent_count += 1

                        if time.time() - last_log >= 1:
                            print(f"📤 Sent {sent_count} events/sec")
                            sent_count = 0
                            last_log = time.time()

                    except json.JSONDecodeError:
                        continue
                    except KafkaError as ke:
                        print("❌ Kafka error, recreating producer...", ke)
                        producer.close()
                        producer = create_producer()

        except requests.RequestException as e:
            print("⚠️ Generator disconnected, retrying...", e)
            time.sleep(2)

        except Exception as e:
            print("🔥 Unexpected error:", e)
            time.sleep(2)


if __name__ == "__main__":
    stream_transactions()