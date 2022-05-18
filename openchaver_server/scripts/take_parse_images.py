from camera.models import  ScreenCapture
def run():
    """
    Take a screen capture and process it.
    """
    def take():
        ScreenCapture.snap(full_monitor=False)
    
    take()
    for i in ScreenCapture.objects.filter(is_proccessed=False):
        i.process_image(skin_threshold=2,parse_images=True)

    

