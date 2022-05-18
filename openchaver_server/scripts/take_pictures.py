from camera.models import ScreenCapture
import time
def run():
    """
    Take a screen capture and process it.
    """
    def take():
        ScreenCapture.snap(full_monitor=False)

    # Take a screen capture once every 5 seconds for 10 minutes
    
    for i in range(0,100):
        x = time.time()
        take()
        print('Time: '+str(time.time()-x))
        time.sleep(5)