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
if len(sys.argv) >2: 
    save_plot = args[1].lower()
else:
    save_plot = "true" 

# print(f"Filename to process: {filename}")
# print(f"Save plot: {save_plot}")


response = requests.get("http://127.0.0.1:5000/compute", params={"filename": filename, "save_plot": save_plot})
result = response.json()
print(result)

# If a plot was saved, print the information
if "plot_saved" in result:
    print(f"Plot saved to: {result['plot_saved']}")
    print(f"Plot available at: http://127.0.0.1:5000{result['plot_url']}")


print(f"Elapsed time for call: {time.time()-Start_time:0.2f} seconds")