import requests
import time


Start_time=time.time()

import sys,os

    
args = sys.argv[1:]
filename ="a"

if len(sys.argv) >1: 
    filename = args[0]
if os.path.exists(filename):
    image_process_start= time.time()


response = requests.get("http://127.0.0.1:5000/compute", params={"filename": filename})
print(response.json())


print(f"Elapsed time for call: {time.time()-Start_time:0.2f} seconds")