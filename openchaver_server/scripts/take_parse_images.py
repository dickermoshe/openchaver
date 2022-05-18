from camera.models import  ScreenCapture
def run():
    """
    Take a screen capture and process it.
    """
    def take():
        ScreenCapture.snap(full_monitor=False)
        ScreenCapture.snap(full_monitor=True)
    
    take()
    for i in ScreenCapture.objects.filter(is_proccessed=False):
        i.process_image(skin_threshold=2,parse_images=True)

    # take()
    # for i in ScreenCapture.objects.filter(is_proccessed=False):
    #     i.process_image(skin_threshold=0,parse_images=False)
    
    # take()
    # for i in ScreenCapture.objects.filter(is_proccessed=False):
    #     i.process_image(skin_threshold=0.1,parse_images=True)
    
    # take()
    # for i in ScreenCapture.objects.filter(is_proccessed=False):
    #     i.process_image(skin_threshold=0.1,parse_images=False)
    

