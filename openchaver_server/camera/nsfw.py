from openchaver_server.settings import sct , interpreter ,desktop

from PIL import Image
import cv2 as cv
import numpy as np

from pywinauto.win32structures import RECT
from PIL.Image import Image as PILImage
from pywinauto.controls.uiawrapper import UIAWrapper

# Get the active window
def get_active_window():
    for window in desktop.windows():
        try:
            if window.is_active() and window.window_text() not in [
                'Program Manager',
                'Taskbar',
            ]:
                return window
        except:
            pass
    return None

# The first function return the coordinates of the picture we want to take
def get_coordinates_on_screen(source:str) -> list[dict[str,int]]:
    
    def _rectangle_to_coordinates(rectangle:RECT) -> dict[str,int]:
        top = rectangle.top
        left = rectangle.left
        width = rectangle.width()
        height = rectangle.height()
        return {'top':top,'left':left,'width':width,'height':height}

    if isinstance(source,UIAWrapper):
        coordinates = _rectangle_to_coordinates(source.rectangle())
        return [coordinates]

    elif source == 'monitor':
        if len(sct.monitors) == 1:
            return sct.monitors
        else:
            return sct.monitors[1:]
    
    return []
    
# Adjust the size of the coordinates to fit the screen
def fit_coordinates_to_monitor(coordinates:dict[str,int]) -> dict[str,int]:
    # Get the monitor
    monitor = sct.monitors[0]

    # Adjust the size of the monitor coordinates to remove the taskbar
    monitor['height'] = monitor['height'] - 40

    # Ensure that top and left are positive
    left = coordinates['left'] if coordinates['left'] > 0 else 0
    top = coordinates['top'] if coordinates['top'] > 0 else 0

    # Ensure top and left are within the monitor
    top = coordinates['top'] if coordinates['top'] < monitor['height'] else monitor['height']
    left = coordinates['left'] if coordinates['left'] < monitor['width'] else monitor['width']

    # Ensure that the width and height are within the monitor
    width = coordinates['width'] if coordinates['width'] < monitor['width'] else monitor['width']
    height = coordinates['height'] if coordinates['height'] < monitor['height'] else monitor['height']

    return {'top':top,'left':left,'width':width,'height':height}

# Take a screenshot of the coordinates
def take_picture_of_coordinates(coordinates:dict[str,int]) -> PILImage:
    # Take a screenshot of the coordinates
    screenshot = sct.grab(coordinates)
    if screenshot.size.width > 0 and screenshot.size.height > 0:
        return Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
    else:
        return None

# Get Title of the active window
def get_title_of_window(window) -> str:
    if window != None:
        return window.window_text()
    else:
        return ''

# Get the skin rating of an image
def get_skin_rating_of_image(img : PILImage) -> float:
    # Pillow to OpenCV and change to BGR
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)

    #converting from gbr to hsv color space
    img_HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    #skin color range for hsv color space 
    HSV_mask = cv.inRange(img_HSV, (0, 15, 0), (17,170,255)) 
    HSV_mask = cv.morphologyEx(HSV_mask, cv.MORPH_OPEN, np.ones((3,3), np.uint8))

    #converting from gbr to YCbCr color space
    img_YCrCb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)

    #skin color range for hsv color space 
    YCrCb_mask = cv.inRange(img_YCrCb, (0, 135, 85), (255,180,135)) 
    YCrCb_mask = cv.morphologyEx(YCrCb_mask, cv.MORPH_OPEN, np.ones((3,3), np.uint8))

    #merge skin detection (YCbCr and hsv)
    global_mask=cv.bitwise_and(YCrCb_mask,HSV_mask)
    global_mask=cv.medianBlur(global_mask,3)
    global_mask = cv.morphologyEx(global_mask, cv.MORPH_OPEN, np.ones((4,4), np.uint8))
    global_result=cv.bitwise_not(global_mask)

    # Get a percentage of skin pixels
    total_pixels = global_result.shape[0] * global_result.shape[1]
    skin_pixels = cv.countNonZero(cv.bitwise_not(global_result))
    skin_percentage = (skin_pixels / total_pixels) * 100
    return skin_percentage

# Get the nsfw rating of an image
def get_nsfw_rating_of_image(img : PILImage) -> float:
    IMAGE_DIM = 256
    img = np.array(img)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    img = np.asarray(img,dtype='float32')
    if len(img.shape) == 3:
        pass
    elif len(img.shape) == 2:
        img = img.reshape((img.shape[0], img.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: %s' % (img.shape,))
    img /= 255
    img = np.asarray([img])[0]
    img = cv.resize(img, (IMAGE_DIM, IMAGE_DIM))
    img = np.expand_dims(img, axis=0)
    input_data = np.array(img, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return  output_data[0][0]

# Parse screenshot into a list of images
def parse_screenshot_to_real_pictures(img:PILImage) -> list[PILImage]:

    # Convert the image to OpenCV format
    img = cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)

    shifted_up = np.roll(img, 1, axis=0)
    # Shift the image down one pixel
    shifted_down = np.roll(img, -1, axis=0)
    # Shift the image left one pixel
    shifted_left = np.roll(img, 1, axis=1)
    # Shift the image right one pixel
    shifted_right = np.roll(img, -1, axis=1)

    # Blur the images together
    t_img = cv.addWeighted(shifted_up, 1, shifted_down, 0, 0.0)
    t_img = cv.addWeighted(t_img, 1, shifted_left, 0, 0.0)
    mask = cv.addWeighted(t_img, 1, shifted_right, 0, 0.0)


    # Create a new image that is completely black besides any pixels that are the same between the two images
    diff = cv.absdiff(img, mask)

    # Threshhold any pixels that are not black
    _, thresh = cv.threshold(diff, 0, 255, cv.THRESH_BINARY)

    # Make the image grayscale
    gray = cv.cvtColor(thresh, cv.COLOR_BGR2GRAY)

    # Blur the image
    #blurred = cv.GaussianBlur(gray, (5, 5), 0)

    # Find the contours
    contours, _ = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Remove the contours that are too small
    contours = [c for c in contours if cv.contourArea(c) > 1000]


    bounding_boxes = []

    for cnt in contours:
        x,y,w,h = cv.boundingRect(cnt)
        # if box is too short or thin, skip it
        if w < 100 or h < 100:
            continue

        bounding_boxes.append((x,y,w,h))
    
    # Save the image
    images= []
    for box in bounding_boxes:
        x,y,w,h = box
        cropped_image = img[y:y+h, x:x+w]
        if cropped_image.size != 0:
            images.append(cropped_image)
    
    # Convert the images to PIL format
    images = [Image.fromarray( cv.cvtColor(img , cv.COLOR_RGB2BGR) ) for img in images]
    
    return images

