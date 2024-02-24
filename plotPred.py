import warnings
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the color names dataset from the CSV file
dataset = pd.read_csv('wikipedia_color_names.csv', header=None, names=['label', 'hex', 'r', 'g', 'b','hue1','hue2','hue3'])

# Extract features (RGB values) and labels
X = dataset[['r', 'g', 'b']]
y = dataset['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.28, random_state=23)

# Suppress the UserWarning about feature names
warnings.filterwarnings("ignore", category=UserWarning)

###################################### Train the Support Vector Machine (SVM) classifier
###################################### svm_classifier = SVC(kernel='linear')
###################################### svm_classifier.fit(X_train, y_train)

####################################### Make predictions on the test set
####################################### y_pred = svm_classifier.predict(X_test)

####################################### rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust the number of estimators as needed
####################################### rf_classifier.fit(X_train, y_train)

####################################### Make predictions on the test set
####################################### y_pred = rf_classifier.predict(X_test)

knn_classifier = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors (n_neighbors) as needed
knn_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn_classifier.predict(X_test)


# Calculate and print the accuracy on the test set
accuracy = accuracy_score(y_test, y_pred)

# Assuming ytest and ypred are lists or arrays containing the true and predicted values

# Extract RGB values from the dataset
r_values = X_test['r']
g_values = X_test['g']
b_values = X_test['b']

predRval = []
predGval = []
predBval = []

for i in range(1,len(y_pred)):
    
    
    predR = dataset.loc[dataset['label'] == y_pred[i]]['r']
    predG = dataset.loc[dataset['label'] == y_pred[i]]['g']
    predB = dataset.loc[dataset['label'] == y_pred[i]]['b']

    predRval.append(predR.iloc[0])
    predGval.append(predG.iloc[0])
    predBval.append(predB.iloc[0])

    print(y_pred[i])
    print([float(predR.iloc[0]),float(predG.iloc[0]),float(predB.iloc[0])])
    print([float(r_values.iloc[i]),float(g_values.iloc[i]),float(b_values.iloc[i])])



# Create a 3D figure and a subplot with 3D projection
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Create a scatter plot using RGB values as both coordinates and colors
#ax.scatter(r_values, g_values, b_values, c=list(zip(r_values/255, g_values/255, b_values/255)), marker='o')
ax.scatter(r_values, g_values, b_values, c='green', marker='o')
ax.scatter(predRval, predGval, predBval, c='red', marker='o')


# Set labels for axes
ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')

# Set title
ax.set_title('3D Scatter Plot of RGB Colors')

# Display the plot
plt.show()
