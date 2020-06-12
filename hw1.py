import pandas as pd

# Defining the functions
# Function for the fit the parameters into the given model
def polynomialModel(X,parameters):
    a0, a1, a2, a3 = parameters[0], parameters[1], parameters[2], parameters[3]
    m = X.shape[0]
    y = []
    for i in range(m):
        F1, F2, F3 = X[i][0], X[i][1], X[i][2]
        model = pd.read_csv("model.txt")
        y.append(eval(model.columns[0]))
    return y

# Function for calculating mean square error
def mse(Y, yhat):
    error = 0
    m = Y.shape[0]
    for i in range(m):
        error += (Y[i][0]-yhat[i])**2
    error = 1/m * error
    return error

# Gradient descent function
def gradient(parameters, X, Y):
    
    
    A = polynomialModel(X,parameters) 
    cost = mse(Y,A)  
    
    m = X.shape[0]
    da0, da1, da2, da3 = 0,0,0,0
    for i in range(m):
        
        dz = A[i] - Y[i]
        da0 += 2*pow(X[i][0],3) * dz
        da1 += 2*pow(X[i][1],2) * dz
        da2 += 2*X[i][2] * dz
        da3 += 2*dz
    
    da0 = 1/m * da0
    da1 = 1/m * da1
    da2 = 1/m * da2
    da3 = 1/m * da3
    
    grads = {"da0": da0, "da1": da1, "da2": da2, "da3": da3}
    
    return grads, cost


# Function which optimize the parameters according to the cost
def optimizer(parameters, X, Y, num_iterations, learning_rate, batch_size):

    costs = []
    a0,a1,a2,a3 = parameters[0], parameters[1], parameters[2], parameters[3]
    m = X.shape[0]
    for j in range(num_iterations):
        for i in range(0,m,batch_size):
            X_min = X[i:i+batch_size]
            Y_min = Y[i:i+batch_size]
            grads, cost = gradient(parameters, X_min, Y_min)
            da0, da1, da2, da3 = grads["da0"], grads["da1"], grads["da2"], grads["da3"]
            a0 = a0 - learning_rate * da0
            a1 = a1 - learning_rate * da1
            a2 = a2 - learning_rate * da2
            a3 = a3 - learning_rate * da3
            parameters = [a0,a1,a2,a3]
            costs.append(cost)


    parameters = {"a0": a0, "a1": a1, "a2": a2, "a3": a3}
    grads = {"da0": da0, "da1": da1, "da2": da2, "da3": da3}

    return parameters, grads, costs

# Initialize with random parameters
parameters = [10, 10, 10, 10]

# Load the dataset and split train and test set
dataset = pd.read_csv("data.csv")
X = dataset.iloc[:,:3].values
Y = dataset.iloc[:,3:4].values
split_factor = 0.80
split = int(split_factor * dataset.shape[0]) 
X_train = X[:split,:]
X_test = X[split:,:]
Y_train = Y[:split,:]
Y_test = Y[split:,:]

# Calculating new parameters with a 300 iteration, 0.01 learning rate and 8 batch size. Best results are achieved with this measures.
# Also, learning should not be so big, because it can pass the local optima, and not so small, because descending will be so slow.
# Gradient descent will stop after 300 iteration.
params, grads, costs = optimizer(parameters, X_train, Y_train, num_iterations= 300, learning_rate = 0.01, batch_size=8)
print("Old Parameters")
print("---------------------")
print ("a0 = " + str(parameters[0]))
print ("a1 = " + str(parameters[1]))
print ("a2 = " + str(parameters[2]))
print ("a3 = " + str(parameters[3]))

# Calculating new output values with initialized parameters
Y_prediction_train_before = polynomialModel(X_train, parameters)
Y_prediction_test_before = polynomialModel(X_test, parameters)

# Calculating mean-square error with initialized parameters
print("Mean Square Error in Train Set with Initialized Parameters: {}".format(
    mse(Y_train,Y_prediction_train_before)))
print("Mean Square Error in Test Set with Initialized Parameters: {}".format(
    mse(Y_test,Y_prediction_test_before)))

print("New Parameters")
print("---------------------")
print ("a0 = " + str(params["a0"]))
print ("a1 = " + str(params["a1"]))
print ("a2 = " + str(params["a2"]))
print ("a3 = " + str(params["a3"]))

# Inserting the calculated parameters into a new list
newParameters = []
newParameters.append(params["a0"][0])
newParameters.append(params["a1"][0])
newParameters.append(params["a2"][0])
newParameters.append(params["a3"][0])

# Calculating new output values with the new parameters
Y_prediction_train = polynomialModel(X_train, newParameters)
Y_prediction_test = polynomialModel(X_test, newParameters)

# Calculating mean-square error with initialized parameters
print("Mean Square Error in Train Set with New Parameters: {}".format(
    mse(Y_train,Y_prediction_train)))
print("Mean Square Error in Test Set with New Parameters: {}".format(
    mse(Y_test,Y_prediction_test)))

