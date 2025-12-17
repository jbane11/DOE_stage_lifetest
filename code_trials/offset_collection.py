import sys,time,os
sys.path.append('C:\\Users\\deotech\\Documents\\Bane\\DOE_stage_lifetest')
import angle_extraction as aex
import matplotlib.pyplot as plt
# Grab image from a Basler pua1600 camera using pypylon
try:
    from pypylon import pylon
except ImportError:
    print("pypylon library not found. Please install it with 'pip install pypylon'.")
    raise

running = True
count= 0
good_fit_count = 0

timestamp_file = time.strftime("%Y%m%d-%H%M%S")
data_filename = f"DOE_homing_data_{timestamp_file}.csv"

data_file = open(data_filename,"w")
data_file.write("Angle,Error,Success,Quailty\n")

camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
# Create an instant camera object with the camera device found first


# Print camera info
print("Using camera:", camera.GetDeviceInfo().GetModelName())


while running and count < 2000:


    # Start grabbing one image
    camera.StartGrabbingMax(1)
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_RGB8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    while camera.IsGrabbing():
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded():
            # Access the image data
            image = converter.Convert(grabResult)
            img = image.GetArray()
            from PIL import Image
            img_pil = Image.fromarray(img)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            file_name = f"images/basler_capture_{timestamp}.png"
            img_pil.save(file_name)
            # print(f"Image captured and saved as {file_name} ")
        else:
            print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
        grabResult.Release()
    
    #add timestamp to filename
    
    results = aex.Analyze_Image_lifetest(file_name,plot_level=1,verbose_level=0)
    
    if results[3] > 0.6 and results[2]:
        # save to file and incremit count
        data_file.write(f"{results[0]},{results[1]},{results[2]},{results[3]}\n")
        good_fit_count+=1
    if count%10==0:
        print(count, good_fit_count, results)
        
    if count%100!=0:
        ## delete the image  
        os.remove(file_name)
    if count%100==0:
        try:
            plt.savefig(f"ana_images/DOE_angle_home_{timestamp}.png")
        except:
            print(f"error savin image")
            
            
        
    
    if good_fit_count >=1000:
        break
    
    # plt.show()
    # time.sleep(0.5)
    plt.close('all')
    count +=1
    
data_file.close()
camera.Close()