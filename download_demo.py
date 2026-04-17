import requests
import os

dongle_id = "5beb9b58bd12b691"
route_name = "0000010a--a51155e496"
base_url = f"https://commadataci.blob.core.windows.net/openpilotci/{dongle_id}/{route_name}"

os.makedirs("samples", exist_ok=True)

valid_count = 0
for i in range(20):
    url = f"{base_url}/{i}/qlog.bz2"
    r = requests.head(url)
    if r.status_code == 200:
        print(f"Downloading segment {i}...")
        r = requests.get(url)
        with open(f"samples/qlog_{i}.bz2", "wb") as f:
            f.write(r.content)
        valid_count += 1
    else:
        print(f"Segment {i} not found (Status: {r.status_code})")

print(f"Total valid segments downloaded: {valid_count}")
