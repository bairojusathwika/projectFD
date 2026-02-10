from kafka import KafkaConsumer
import json
import pymysql
import time
from datetime import datetime, date
from collections import deque

# -----------------------------
# Wait for MariaDB (Docker-safe)
# -----------------------------
def wait_for_db(host, user, password, database, retries=30, delay=2):
    for i in range(retries):
        try:
            return pymysql.connect(
                host=host,

                user=user,
                password=password,
                database=database,
                autocommit=False
            )
        except Exception:
            print(f"⏳ Waiting for MariaDB... ({i+1}/{retries})")
            time.sleep(delay)
    raise RuntimeError("❌ MariaDB not reachable")

# -----------------------------
# Kafka Consumer
# -----------------------------
consumer = KafkaConsumer(
    "fraud-transactions",
    bootstrap_servers="192.168.8.180:9092",
    auto_offset_reset="earliest",
    enable_auto_commit=False,   # ✅ FIX 2
    group_id="txn-consumer-mariadb",
    value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    max_poll_records=100
)

print("🚀 Kafka consumer started")

# -----------------------------
# MariaDB Connection
# -----------------------------
db = wait_for_db(
    host="mariadb",

    user="user",
    password="userpass",
    database="fraud_db"
)

cursor = db.cursor()
print("✅ Connected to MariaDB")

# -----------------------------
# Metrics
# -----------------------------
BATCH_SIZE = 100
batch = []
recent_times = deque(maxlen=100)
message_count = 0
start_time = time.time()
last_report_time = start_time
last_report_count = 0

# -----------------------------
# Helper Functions
# -----------------------------
def ensure_user(user_id):
    cursor.execute("""
        INSERT IGNORE INTO users (user_id, created_at, status, risk_level)
        VALUES (%s, NOW(), 'ACTIVE', 'LOW')
    """, (user_id,))


def update_daily_stats(user_id, amount, tx_date):
    cursor.execute("""
        INSERT INTO daily_user_stats (
            user_id, stat_date, total_tx_count,
            total_tx_amount, avg_tx_amount,
            max_tx_amount, min_tx_amount, tx_per_minute
        )
        VALUES (%s, %s, 1, %s, %s, %s, %s, 0)
        ON DUPLICATE KEY UPDATE
            total_tx_count = total_tx_count + 1,
            total_tx_amount = total_tx_amount + VALUES(total_tx_amount),
            avg_tx_amount = total_tx_amount / total_tx_count,
            max_tx_amount = GREATEST(max_tx_amount, VALUES(max_tx_amount)),
            min_tx_amount = LEAST(min_tx_amount, VALUES(min_tx_amount))
    """, (user_id, tx_date, amount, amount, amount, amount))


def update_network_profile(user_id, ip, device):
    cursor.execute("""
        INSERT INTO user_network_profile (
            user_id, ip_address, device_id,
            first_seen, last_seen, usage_count
        )
        VALUES (%s, %s, %s, NOW(), NOW(), 1)
        ON DUPLICATE KEY UPDATE
            last_seen = NOW(),
            usage_count = usage_count + 1
    """, (user_id, ip, device))


def update_location(user_id, location):
    cursor.execute("""
        INSERT INTO user_home_location (
            user_id, country, city, confidence
        )
        VALUES (%s, %s, %s, 1)
        ON DUPLICATE KEY UPDATE
            confidence = LEAST(confidence + 0.01, 1.0)
    """, (
        user_id,
        location.get("country"),
        location.get("city")
    ))

# -----------------------------
# Consume Loop
# -----------------------------
print("📥 Waiting for messages...")

for message in consumer:
    batch_start = time.time()

    txns = message.value if isinstance(message.value, list) else [message.value]
    batch.extend(txns)

    if len(batch) >= BATCH_SIZE:
        try:
            for txn in batch:
                from_user = txn["from_account"]
                to_user = txn["to_account"]
                amount = float(txn["amount"])
                ip = txn.get("ip_address")
                device = txn.get("device_id")
                location = txn.get("location", {})
                tx_date = datetime.fromisoformat(txn["timestamp"]).date()

                # ---- FROM USER ----
                ensure_user(from_user)
                update_daily_stats(from_user, amount, tx_date)
                update_network_profile(from_user, ip, device)
                update_location(from_user, location)

                # ---- TO USER ----
                ensure_user(to_user)

            db.commit()
            consumer.commit()     # ✅ FIX 2
            message_count += len(batch)

        except Exception as e:
            print(f"❌ MariaDB error: {e}")
            db.rollback()

        recent_times.append(time.time() - batch_start)
        batch.clear()

        # ---- Metrics ----
        now = time.time()
        if now - last_report_time >= 10:
            elapsed = now - start_time
            window_msgs = message_count - last_report_count
            window_time = now - last_report_time
            avg_batch = sum(recent_times) / len(recent_times) if recent_times else 0  # ✅ FIX 3

            print("\n" + "=" * 60)
            print("📊 MARIA INGESTION METRICS")
            print("=" * 60)
            print(f"Total messages: {message_count:,}")
            print(f"Overall TPS:    {message_count / elapsed:.0f}")
            print(f"Window TPS:     {window_msgs / window_time:.0f}")
            print(f"Avg batch time: {avg_batch:.2f}s")
            print("=" * 60 + "\n")

            last_report_time = now
            last_report_count = message_count
