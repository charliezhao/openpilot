import requests
import os
from openpilot.tools.lib.comma_car_segments import get_comma_car_segments_database, get_url
from openpilot.tools.lib.route import SegmentRange

os.makedirs("samples", exist_ok=True)

db = get_comma_car_segments_database()
count = 0
for fp in db:
    for segment in db[fp]:
        if count >= 20: break
        sr = SegmentRange(segment)
        url = get_url(sr.route_name, sr.slice)
        print(f"Downloading segment {count} from {fp}: {url}")
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                with open(f"samples/qlog_{count}.bz2", "wb") as f:
                    f.write(r.content)
                count += 1
            else:
                print(f"Failed to download {url} (Status: {r.status_code})")
        except Exception as e:
            print(f"Error downloading {url}: {e}")
    if count >= 20: break

print(f"Total downloaded: {count}")
