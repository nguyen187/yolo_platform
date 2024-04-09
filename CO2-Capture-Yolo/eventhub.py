# event Hub
import json
from azure.eventhub import EventHubProducerClient, EventData
import time 
event_hub_connection_string = ''
event_hub_name = 'yolorealtime'
producer = EventHubProducerClient.from_connection_string(conn_str=event_hub_connection_string, eventhub_name=event_hub_name)
# Opening JSON file
event_data_batch = producer.create_batch() # Create a batch. You will add events to the batch later.  

def send_data():
    try:
        while True:
            # returns JSON object as 
            # a dictionary
            f = open('./realtime/data.json')

            s = json.load(f)
            print(s)
            event_data_batch.add(EventData(s)) # Add event data to the batch.
            producer.send_batch(event_data_batch)
            time.sleep(0.5) #delate every 10s
            send_data.v = s
    except KeyboardInterrupt:
        print('Error!')
        producer.close()

    # Closing file
    f.close()

send_data()