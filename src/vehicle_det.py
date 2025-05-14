from PIL import Image
import cv2
import numpy as np

# Load and resize the image
image_path = r"Dynamic-Traffic-light-prediction-project/src/data/image.jpg"
image = Image.open(image_path).resize((450, 250))
image_arr = np.array(image)

# Convert to grayscale and preprocess
grey = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(grey, (5, 5), 0)
dilated = cv2.dilate(blur, np.ones((3, 3)))
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

# Load Haar cascade classifiers
car_cascade = cv2.CascadeClassifier(r"Dynamic-Traffic-light-prediction-project/src/data/cars.xml")
bus_cascade = cv2.CascadeClassifier(r"Dynamic-Traffic-light-prediction-project/src/data/bus.xml")          
bike_cascade = cv2.CascadeClassifier(r"Dynamic-Traffic-light-prediction-project/src/data/bikes.xml") 

# Detect vehicles
cars = car_cascade.detectMultiScale(closing, 1.1, 1)
buses = bus_cascade.detectMultiScale(closing, 1.1, 1)
bikes = bike_cascade.detectMultiScale(closing, 1.1, 1)

# Draw rectangles and count
for (x, y, w, h) in cars:
    cv2.rectangle(image_arr, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue for cars

for (x, y, w, h) in buses:
    cv2.rectangle(image_arr, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green for buses

for (x, y, w, h) in bikes:
    cv2.rectangle(image_arr, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red for bikes

# Display counts
print(f"{len(cars)} cars found")
print(f"{len(buses)} buses found")
print(f"{len(bikes)} bikes found")

# Show final annotated image
annotated_image = Image.fromarray(image_arr)
annotated_image.show()



















































# from PIL import Image
# import cv2
# import numpy as np
# import requests

# # Downloading and resizing the image from the URL
# image_url = r"D:\Mohammad Alfaiz\Project\ML Projects\Project testing\Dynamic-Traffic-light-prediction-project\data\raw\image.png"



# #response = requests.get(image_url, stream=True)
# image = Image.open(image_url)
# image = image.resize((450, 250))


# # Convert the image to a Numpy array
# image_arr = np.array(image)

# # Show the converted Numpy array as an image
# #cv2.imshow("Converted Image to Numpy Array", image_arr)
# #cv2.waitKey(0)
# #cv2.destroyAllWindows()

# # Convert the image to grayscale
# grey = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)

# # Show the grayscale image
# #cv2.imshow("Grayscale Image", grey)


# # Apply Gaussian blur to the grayscale image
# blur = cv2.GaussianBlur(grey, (5, 5), 0)

# # Show the blurred image
# #cv2.imshow("Blurred Image", blur)
# #cv2.waitKey(0)
# #cv2.destroyAllWindows()

# # Apply dilation to the blurred image
# dilated = cv2.dilate(blur, np.ones((3, 3)))

# # Show the dilated image
# #cv2.imshow("Dilated Image", dilated)
# #cv2.waitKey(0)
# #cv2.destroyAllWindows()

# # Apply morphological closing to the dilated image
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
# closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

# # Show the morphologically closed image
# #cv2.imshow("Morphologically Closed Image", closing)
# #cv2.waitKey(0)
# #cv2.destroyAllWindows()

# # Use CascadeClassifier for car detection
# car_cascade_src = r"D:\Mohammad Alfaiz\Project\ML Projects\Project testing\Dynamic-Traffic-light-prediction-project\data\raw\cars.xml"
# car_cascade = cv2.CascadeClassifier(car_cascade_src)
# cars = car_cascade.detectMultiScale(closing, 1.1, 1)

# # Draw rectangles around each detected car and count
# cnt_car = 0
# for (x, y, w, h) in cars:
#     cv2.rectangle(image_arr, (x, y), (x + w, y + h), (255, 0, 0), 2)
#     cnt_car += 1

# # Print the total number of detected cars and buses
# print(cnt_car, " cars found")



# # Use CascadeClassifier for bus detection
# bus_cascade_src = r"D:\Mohammad Alfaiz\Project\ML Projects\Project testing\Dynamic-Traffic-light-prediction-project\data\raw\bus.xml"
# bus_cascade = cv2.CascadeClassifier(bus_cascade_src)
# buses = bus_cascade.detectMultiScale(closing, 1.1, 1)

# # Draw rectangles around each detected bus and count
# cnt_bus = 0
# for (x, y, w, h) in buses:
#     cv2.rectangle(image_arr, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     cnt_bus += 1

# # Print the total number of detected buses
# print(cnt_bus, " bus found")



# # Use CascadeClassifier for bus detection
# two_wheeler_cascade_src = r"D:\Mohammad Alfaiz\Project\ML Projects\Project testing\Dynamic-Traffic-light-prediction-project\data\raw\two_wheeler.xml"
# two_wheeler_cascade = cv2.CascadeClassifier(two_wheeler_cascade_src)
# two_wheelers= two_wheeler_cascade.detectMultiScale(closing, 1.1, 1)

# # Draw rectangles around each detected car and count
# cnt_two_wheeler = 0
# for (x, y, w, h) in two_wheelers:
#     cv2.rectangle(image_arr, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     cnt_two_wheeler += 1

# # Print the total number of detected cars and buses
# print(cnt_two_wheeler, " two wheelers found")


# # Convert the annotated image to PIL Image format and display it
# annotated_image = Image.fromarray(image_arr)
# annotated_image.show()

# # Close the window when a key is pressed
# cv2.waitKey(0)
# cv2.destroyAllWindows()
