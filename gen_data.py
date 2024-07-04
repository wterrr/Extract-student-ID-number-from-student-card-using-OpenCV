import cv2
import numpy as np

minContour = 100 
widthResize = 20 
heightResize = 30 

def preprocessImage(imageInput):
    grayImage = cv2.cvtColor(imageInput, cv2.COLOR_BGR2GRAY)
    blurredImage = cv2.GaussianBlur(grayImage, (5, 5), 0)
    thresholdImage = cv2.adaptiveThreshold(blurredImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 11, 2)
    return thresholdImage

def processContour(currentContour, thresholdImage, trainingImage, validCharacters):
    x, y, w, h = cv2.boundingRect(currentContour)
    
    cv2.rectangle(trainingImage, (x, y), (x+w, y+h), (0, 0, 255), 2)

    roiImage = thresholdImage[y:y+h, x:x+w]
    resizedRoiImage = cv2.resize(roiImage, (widthResize, heightResize))

    cv2.imshow("roiImage", roiImage)
    cv2.imshow("resizedRoiImage", resizedRoiImage)
    cv2.imshow("training_image.png", trainingImage)

    keyPress = cv2.waitKey(0)

    if keyPress == 27: 
        return None, None, True
    elif keyPress in validCharacters:
        print(keyPress)
        flattenedImage = resizedRoiImage.reshape((1, widthResize * heightResize))
        return flattenedImage, keyPress, False

    return None, None, False


def main():
    trainingImage = cv2.imread("training_chars.png")

    if trainingImage is None:
        print("error: image not read from file \n\n")
        input("Press Enter to continue...")
        return

    processedImage = preprocessImage(trainingImage)
    cv2.imshow("processedImage", processedImage)

    foundContours, _ = cv2.findContours(processedImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fltImages = np.empty((0, widthResize * heightResize))
    characterClassifications = []

    validCharacters = [ord(ch) for ch in '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ']

    for contour in foundContours:
        if cv2.contourArea(contour) > minContour:
            imageFlt, keyChar, exitLoop = processContour(contour, processedImage, trainingImage, validCharacters)
            if exitLoop:
                break
            if imageFlt is not None:
                characterClassifications.append(keyChar)
                fltImages = np.append(fltImages, imageFlt, 0)

    floatCls = np.array(characterClassifications, np.float32)
    label = floatCls.reshape((floatCls.size, 1))

    print("\n\ntraining complete !!\n")

    np.savetxt("lebel.txt", label)
    np.savetxt("train.txt", fltImages)

    cv2.destroyAllWindows()
    return

if __name__ == "__main__":
    main()
