
# Import necessary libraries
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import csv

# Function to add polynomial features to a dataset
def add_polynomial_features(data, degree=2):
    # Duplicate the original data
    data_poly = np.copy(data)
    num_features = data.shape[1]

    for feature in range(num_features):
        for d in range(2, degree + 1):
            data_poly = np.column_stack((data_poly, data[:, feature] ** d))
    
    return data_poly

# Open and read the CSV file "Admission_Predict_Ver1.1.csv" to load data into arrays
with open("Admission_Predict_Ver1.1.csv") as f:
    d = csv.reader(f, delimiter=",")

    # Initialize lists to store data columns
    sno, gre_score, toefle, university, sop, lor, cgpa, research, admit = [], [], [], [], [], [], [], [], []

    # Skip the header row
    for x in d:
        break

    # Read and store data from the CSV file
    for x in d:
        sno.append(float(x[0]))
        gre_score.append(float(x[1]))
        toefle.append(float(x[2]))
        university.append(float(x[3]))
        sop.append(float(x[4]))
        lor.append(float(x[5]))
        cgpa.append(float(x[6]))
        research.append(float(x[7]))
        admit.append(float(x[8]))

# Open the CSV file again to load data into a list and a NumPy array
with open("Admission_Predict_Ver1.1.csv") as f:
    d = csv.reader(f, delimiter=",")

    # Read the data into a list and remove the header
    data = list(d)
    data.pop(0)

    # Convert the list to a NumPy array
    data_array = np.array(data)

# Convert the "admit" list to a NumPy array
admit = np.array(admit)

# Extract relevant columns from the data array and convert them to floating-point values
data_array = np.array(data_array)
data_array = data_array[:, 1:-1]
data_array = data_array.astype(np.float64)
for x in range(len(data_array)):
    data_array[x][0] =  data_array[x][0] / 340
    data_array[x][1] = data_array[x][1] / 120
    data_array[x][2] = data_array[x][2] / 5
    data_array[x][3] = data_array[x][3] / 5
    data_array[x][4] = data_array[x][4] / 5
    data_array[x][5] = data_array[x][5] / 10
    data_array[x][6] = data_array[x][6]

# Add polynomial features to the data
degree = 5  # You can change this as needed
data_array_poly = add_polynomial_features(data_array, degree)
print (data_array_poly)
data_array_poly = np.column_stack((data_array_poly, np.ones(len(admit))))
# Perform linear regression on the data with polynomial features
x, res, _, _ = np.linalg.lstsq(data_array_poly, admit, rcond=None)

# Print the regression coefficients and residual sum of squares
print("Regression Coefficients (x):", x)
print("Residual Sum of Squares (res):", res)

# Calculate the predicted values by multiplying coefficients and data features
x_mul = np.dot(data_array_poly, x)

# Create a scatter plot of predicted values versus actual admission values
plt.scatter(x_mul, admit)
plt.plot(admit, admit, color='red')

# Calculate the differences between predicted and actual values
differences = x_mul - admit
mse=np.sum(differences**2,axis=0)/len(admit)
print ("mean squared error=",mse)
# Calculate the standard deviation of the differences
std_deviation = np.std(differences)
print("Standard Deviation of Differences:", std_deviation)

# Show the plot
plt.show()
plt.savefig("test1")
# Calculate correlation coefficients
correlation_coefficients = np.corrcoef(data_array, admit, rowvar=False)[:-1, -1]

# Print the correlation coefficients
for feature_name, corr_coeff in zip(["GRE Score", "TOEFL Score", "University Rating", "SOP", "LOR", "CGPA", "Research"], correlation_coefficients):
    print(f"Correlation coefficient for {feature_name}: {corr_coeff}")
def r_2(y_true, y_pred):
    """ Calculate R-squared (coefficient of determination) given the true and predicted values. """
    mean_y_true = sum(y_true) / len(y_true)
    numerator = sum((y_true_i - y_pred_i) ** 2 for y_true_i, y_pred_i in zip(y_true, y_pred))
    denominator = sum((y_true_i - mean_y_true) ** 2 for y_true_i in y_true)
    r_squared = 1 - (numerator / denominator)
    return r_squared

r_2(admit,x_mul)
