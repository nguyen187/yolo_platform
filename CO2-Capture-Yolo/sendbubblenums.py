import time
from kafka import KafkaProducer
import json


def send_temperature_data(producer, topic):
    while True:

        # Create JSON payload
        # payload = {"temperature": temperature}
        f = open('./realtime/data.json')

        # Serialize payload to JSON
        s = json.load(f)
        # data = {"liquidity":s['liquidity'],"bubble":s[bubble]}        
        # data = json.dumps(data)
        # print(data)

        # Send data to Kafka topic
        print(s)
        producer.send(topic, value=s.encode("utf-8"))
        producer.flush()


        time.sleep(5)  # Wait for 5 seconds
    

if __name__ == "__main__":
    bootstrap_servers = "localhost:9093"  # Kafka broker address
    topic = "states"  # Kafka topic to send data

    # Create Kafka producer
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        # Use default serialization
        value_serializer=lambda v: v,
        api_version=(2, 0, 2)
    )

    try:
        send_temperature_data(producer, topic)
    except KeyboardInterrupt:
        pass
    finally:
        producer.close()
