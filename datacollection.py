import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
# Initialize the hand detector
detector = HandDetector(maxHands=1)

# Capture video from the webcam
cap = cv2.VideoCapture(0)
imgsize = 300
offset = 20  # Extra space around the cropped image
 # save to folder when s in keyboard is pressed
folder = "imges/L"
counter=0
while True:
    success, img = cap.read()
    if not success:
        break

    # Detect hands in the frame
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a white image of fixed size
        imgwhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255

        # Crop the image with some offset
        imgcrop = img[max(0, y-offset):min(y+h+offset, img.shape[0]), max(0, x-offset):min(x+w+offset, img.shape[1])]

        # Resize the cropped image to fit within the white image
        aspectratio = h / w
        if aspectratio > 1:
            # Height is greater than width
            k = imgsize / h
            wcal = math.ceil(k * w)
            wgap = math.ceil((imgsize - wcal) / 2)
            imgresize = cv2.resize(imgcrop, (wcal, imgsize))
            imgresizeshape = imgresize.shape
            imgwhite[:, wgap:wcal+wgap] = imgresize  # Overlay the cropped image onto the white image
        else:
            # Width is greater than height or equal
            k = imgsize / w
            hcal = math.ceil(k * h)
            hgap = math.ceil((imgsize - hcal) / 2)
            imgresize = cv2.resize(imgcrop, (imgsize, hcal))
            imgresizeshape = imgresize.shape
            imgwhite[hgap:hcal+hgap, :] = imgresize  # Overlay the cropped image onto the white image

        # Display the cropped and white images
        cv2.imshow("ImageCrop", imgcrop)
        cv2.imshow("ImageWhite", imgwhite)

        
        

    cv2.imshow("image",img)
    key=cv2.waitKey(1)

    if key==ord("r"):
        counter+=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgwhite)
        print(counter)


