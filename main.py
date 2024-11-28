"""
Implementación de una red neuronal de 3 parámetros 2 pesos y 1 sesgo , desarrollado para efectos didacticos, en el cual se revisan 
conceptos como propagación hacia adelante y función de costo que en este caso tiene una versión simple  
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import utils 


# GRADED FUNCTION: forward_propagation

def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m), where n_x is the dimension input (in our example is 2) and m is the number of training samples
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    Y_hat -- The output of size (1, m)
    """
    # Retrieve each parameter from the dictionary "parameters".
    W = parameters["W"]
    b = parameters["b"]
    
    # Implement Forward Propagation to calculate Z.
    ### START CODE HERE ### (~ 2 lines of code)
    Z = W @ X + b
    Y_hat = Z
    ### END CODE HERE ###
    

    return Y_hat

def compute_cost(Y_hat, Y):
    """
    Computes the cost function as a sum of squares
    
    Arguments:
    Y_hat -- The output of the neural network of shape (n_y, number of examples)
    Y -- "true" labels vector of shape (n_y, number of examples)
    
    Returns:
    cost -- sum of squares scaled by 1/(2*number of examples)
    
    """
    # Number of examples.
    m = Y.shape[1]

    # Compute the cost function.
    cost = np.sum((Y_hat - Y)**2)/(2*m)
    
    return cost

# GRADED FUNCTION: nn_model

def nn_model(X, Y, num_iterations=1000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (n_x, number of examples)
    Y -- labels of shape (1, number of examples)
    num_iterations -- number of iterations in the loop
    print_cost -- if True, print the cost every iteration
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to make predictions.
    """
    
    n_x = X.shape[0]
    
    # Initialize parameters
    parameters = utils.initialize_parameters(n_x) 
    
    # Loop
    for i in range(0, num_iterations):
         
        ### START CODE HERE ### (~ 2 lines of code)
        # Forward propagation. Inputs: "X, parameters, n_y". Outputs: "Y_hat".
        Y_hat = forward_propagation(X, parameters)
        
        # Cost function. Inputs: "Y_hat, Y". Outputs: "cost".
        cost = compute_cost(Y_hat, Y)
        ### END CODE HERE ###
        
        
        # Parameters update.
        parameters = utils.train_nn(parameters, Y_hat, X, Y, learning_rate = 0.001) 
        
        # Print the cost every iteration.
        if print_cost:
            if i%100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))

    return parameters


df = pd.read_csv("data/toy_dataset.csv")
print(df.head())

X = np.array(df[['x1','x2']]).T
Y = np.array(df['y']).reshape(1,-1)
parameters = nn_model(X,Y, num_iterations = 5000, print_cost= True)

# GRADED FUNCTION: predict

def predict(X, parameters):

    W = parameters['W']
    b = parameters['b']

    Z = np.dot(W, X) + b

    return Z

y_hat = predict(X,parameters)
df['y_hat'] = y_hat[0]

for i in range(10):
    print(f"(x1,x2) = ({df.loc[i,'x1']:.2f}, {df.loc[i,'x2']:.2f}): Actual value: {df.loc[i,'y']:.2f}. Predicted value: {df.loc[i,'y_hat']:.2f}")
