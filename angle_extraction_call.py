import requests
import time, argparse

from basler_capture import take_picture


Start_time=time.time()

import sys,os

    
args = sys.argv[1:]
filename ="a"

# make arguments for filename or capture with camera and save plot
arparser = argparse.ArgumentParser(description="Call angle extraction server")
arparser.add_argument("--filename", type=str, default=None, help="Path to the image file to analyze")
arparser.add_argument("--camera", action="store_true",default=False, help="Use camera to capture image instead of file")
arparser.add_argument("--save_plot", action="store_true", default=False, help="Whether to save the plot (true/false)")

args = arparser.parse_args()


if args.camera:
    response = requests.get("http://127.0.0.1:5000/take_picture")
    result = response.json()
    if "filename" in result:
        filename = result["filename"]
        print(f"Image captured and saved as {filename}")
else:
    if args.filename is None:
        print("Error: Must provide either --filename or --camera")
        sys.exit(1)
    filename = args.filename

save_plot = args.save_plot

response = requests.get("http://127.0.0.1:5000/compute", params={"filename": filename, "save_plot": save_plot})
result = response.json()
print(result)

# If a plot was saved, print the information
if "plot_saved" in result:
    print(f"Plot saved to: {result['plot_saved']}")
    print(f"Plot available at: http://127.0.0.1:5000{result['plot_url']}")


print(f"Elapsed time for call: {time.time()-Start_time:0.2f} seconds")