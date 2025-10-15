import sys,time
sys.path.append('C:\\Users\\deotech\\Documents\\Bane\\DOE_stage_lifetest')
import angle_extraction as aex
import matplotlib.pyplot as plt
# Grab image from a Basler pua1600 camera using pypylon
try:
    from pypylon import pylon
except ImportError:
    print("pypylon library not found. Please install it with 'pip install pypylon'.")
    raise




for i in range(20):
    # Create an instant camera object with the camera device found first
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

    # Print camera info
    print("Using camera:", camera.GetDeviceInfo().GetModelName())

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
            print(f"Image captured and saved as {file_name} ")
        else:
            print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
        grabResult.Release()
    camera.Close()
    #add timestamp to filename
    
    
    print(aex.Analyze_Image_lifetest(file_name,plot_level=1,verbose_level=0))
    plt.show()
    # time.sleep(0.5)
    plt.close('all')
    