#Baser camera capture and connection code. 
from pypylon import pylon  


#connect to camera and take picture.

def take_picture(camera=None,filename="basler_capture.png"):
    
    # Initialize the camera if not provided
    if camera is None:
        # Create an instance of the camera
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

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
            img_pil.save(filename)
            print(f"Image captured and saved as {filename}")
        else:
            print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
        grabResult.Release()
    camera.Close()
    return filename


def close_camera(camera):
    if camera is not None and camera.IsOpen():
        camera.Close()
        
def connect_camera():
    # Create an instance of the camera
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    return camera
    
