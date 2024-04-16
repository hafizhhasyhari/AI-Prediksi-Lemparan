import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np
import math

# Initialize the video
cap = cv2.VideoCapture('./files/Videos/vid (3).mp4')

# Create the color finder object
myColorFinder = ColorFinder(False)
hsvVals = {'hmin': 8, 'smin': 96, 'vmin': 115, 'hmax': 14, 'smax': 255, 'vmax': 255}

# Variables
posListX, posListY = [], []
xList = [item for item in range(0, 1300)]
prediction = False

while True:
    # Grab the image
    success, img = cap.read()
    # img = cv2.imread("./files/Ball.png")
    img = img[0:900, :]

    # Find the Color of Ball
    imgColor, mask = myColorFinder.update(img, hsvVals)

    # Find the location of the contours
    imgContours, contours = cvzone.findContours(img, mask, minArea=500)

    if contours:
        # Store the center coordinates of the first contour
        posListX.append(contours[0]['center'][0])
        posListY.append(contours[0]['center'][1])

    if posListX:
        # Polynomial Regression, y = Ax^2 + Bx + C 
        # Find the coefficients (A, B, C)
        A, B, C = np.polyfit(posListX, posListY, 2)

        for i, (posX, posY) in enumerate(zip(posListX, posListY)):
            pos = (posX, posY)

            # Draw circles at detected points
            cv2.circle(imgContours, pos, 10, (0, 255, 0), cv2.FILLED)
            if i == 0:
                cv2.line(imgContours, pos, pos, (0, 255, 0), 5)
            else:
                cv2.line(imgContours, pos, (posListX[i - 1], posListY[i - 1]), (0, 255, 0), 2)

        for x in xList:
            # Calculate predicted y values using the quadratic equation
            y = int(A * x**2 + B * x + C)
            cv2.circle(imgContours, (x, y), 2, (255, 0, 255), cv2.FILLED)

        if len(posListX) < 10:
            # Prediction: X values 330 to 430, Y 590
            # Solving Quadratic Equation
            a = A
            b = B
            c = C - 590
            x = int((-b - math.sqrt(b**2 - (4 * a * c))) / (2 * a))
            prediction = 330 < x < 430

        if prediction:
            cvzone.putTextRect(imgContours, "Basket", (50, 150),
                               scale=5, thickness=5, colorR=(0, 200, 0), offset=20)
        else:
            cvzone.putTextRect(imgContours, "No Basket", (50, 150), scale=5, thickness=5, colorR=(0, 0, 200), offset=20)

    # Display the processed image
    imgContours = cv2.resize(imgContours, (0, 0), None, 0.7, 0.7)
    cv2.imshow("Image", imgContours)

    # Wait for a key press and exit if a key is pressed
    if cv2.waitKey(100) == 27:  # 27 is the ASCII code for the 'Esc' key
        break
