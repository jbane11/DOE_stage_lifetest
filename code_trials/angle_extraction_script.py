

def main(filename):
    
    angle = Analyze_Image_Simple(filename)
    
    
    print(f"File - {filename} has and angle of {angle} degrees")
    
if __name__ == "__main__":
    import time
    starttime = time.time()
    
    from angle_extraction_32bit    import Analyze_Image_Simple
    import sys,os

    importtime = time.time() -starttime
    args = sys.argv[1:]
    filename ="a"

    if len(sys.argv) >1: 
        filename = args[0]
    if os.path.exists(filename):
        iamge_process_start= time.time()
        main(filename)
    
    else:
        print("Please include filename for iamge")
    print(f"elasped time {(time.time()-starttime):0.2f}")
    print(f"Time to import main libraries {importtime:0.2}")

    print(f"Time to process image:: {(time.time()- iamge_process_start):0.2f}")
