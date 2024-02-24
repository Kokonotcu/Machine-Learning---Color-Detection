import warnings
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import *
from PIL import Image

# Load the color names dataset from the CSV file
dataset = pd.read_csv('noisycolors.csv', header=None, names=['label', 'hex', 'r', 'g', 'b','hue1','hue2','hue3'])

# Extract features (RGB values) and labels
X = dataset[['r', 'g', 'b']]
y = dataset['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Suppress the UserWarning about feature names
warnings.filterwarnings("ignore", category=UserWarning)

################################################# Train the Support Vector Machine (SVM) classifier
################################################svm_classifier = SVC(kernel='linear')
################################################svm_classifier.fit(X_train, y_train)
################################################
################################################# Make predictions on the test set
################################################y_pred = svm_classifier.predict(X_test)
################################################y_pred_train = svm_classifier.predict(X_train)

################################################rf_classifier = RandomForestClassifier(n_estimators=100, random_state=12)
################################################rf_classifier.fit(X_train, y_train)
#################################################################################################
################################################y_pred = rf_classifier.predict(X_test)
################################################y_pred_train = rf_classifier.predict(X_train)

knn_classifier = KNeighborsClassifier(n_neighbors=4)  # You can adjust the number of neighbors (n_neighbors) as needed
knn_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn_classifier.predict(X_test)
y_pred_train = knn_classifier.predict(X_train)

# Use LabelEncoder to convert color names to numeric values
##label_encoder = LabelEncoder()
##y_pred_encoded = label_encoder.fit_transform(y_pred)
##y_test_encoded = label_encoder.fit_transform(y_test)
##y_train_encoded = label_encoder.fit_transform(y_train)

accuracy = accuracy_score(y_test, y_pred)
accuracy2 = accuracy_score(y_train,y_pred_train)

f1 = f1_score(y_test, y_pred, average='weighted')
f12 = f1_score(y_train, y_pred_train, average='weighted')

print(f"Accuracy on the test set: {accuracy*100:.2f}%")
print(f"Accuracy on the train set: {accuracy2*100:.2f}%")
print(f"f1 score on the test set: {f1:.3f}")
print(f"f1 score on the train set: {f12:.3f}")

# Function to predict color name for a given RGB value
def predict_color_name(rgb):
    rgb_array = [[rgb[0], rgb[1], rgb[2]]]
    #color_name = rf_classifier.predict(rgb_array)[0]
    color_name = knn_classifier.predict(rgb_array)[0]
    return color_name

# Function to get RGB values from an image at a given (x, y) coordinate
def get_rgb_from_image(image_path, x, y):
    img = Image.open(image_path)
    rgb_value = img.getpixel((x, y))
    return rgb_value


image_path = 'a.jpeg'  # Replace with your image path
image = io.imread(image_path)
image_org = image

height, width, _ = image.shape
flattened_image = image.reshape((height * width, 3))

# Display the original and smoothed images
plt.figure(figsize=(15, 5))


plt.subplot(1,2,1)
plt.imshow(image)
plt.axis('off')


plt.subplot(1,2,2)
plt.axis('off')

cursor_values = plt.ginput(n=1, timeout=0)
rgb_values = image_org[int(cursor_values[0][1])][int(cursor_values[0][0])]
predicted_color_name = predict_color_name(rgb_values)

plt.title(predicted_color_name)
ax = plt.imshow([[rgb_values]])

while plt.ginput(n=1, timeout=0):
    plt.cla()
    cursor_values = plt.ginput(n=1, timeout=0)
    rgb_values = image_org[int(cursor_values[0][1])][int(cursor_values[0][0])]
    predicted_color_name = predict_color_name(rgb_values)
    plt.subplot(1,2,2)
    plt.axis('off')
    plt.imshow([[rgb_values]])
    plt.title(predicted_color_name)

plt.show()


