# Importing necessary Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pickle


# Layer class that will be used to initialize the weights and bias of the layer
class Layer:

    def __init__(self,prev_nodes,current_nodes):
        self.weights = np.random.randn(prev_nodes,current_nodes)
        self.bias = np.ones([current_nodes,1])



# This is the main class that will take the initialized layers and activations, and has some activation functions and their derivations that will be used for the forward and backward propagation
class Network:

    def __init__(self,layers, activations):
        """
            layer and the activations are saved here
        """
        self.layers = layers
        self.activations = activations


    def fit(self,X,Y,learning_rate,epochs,batch_size):
        """
            X : The data on which the model needs to be trained and the data must be in no_of_data_points x no_of_features
            Y : The target feature that needs to be predicted and it must be 1 x no_of_data_points
            learning_rate : learning_rate
            epochs: No. of epochs the model need to be run
            batch_size : No. of datapoints in each part of the dataset, eg if batch_size = 100 and total no_of_Data_points are 1000 then we have 10 parts of data containing 100 datapoint in each part
        """
        self.X = X
        self.Y = Y
        self.learning_rate = learning_rate
        total_data_points = X.shape[0]
        steps = total_data_points // batch_size
        current_epoch_loss = []
        for epoch in range(epochs):
            total_loss = 0
            for step in range(steps+1):
                start_index = step * batch_size
                end_index = (step + 1) * batch_size
                if end_index > total_data_points:
                    end_index = total_data_points
                if start_index == end_index:
                    print("Break")
                    break

                # current_X, current_y contains the smaller part of the data according to the batch_size
                current_X = self.X[start_index:end_index, :]
                current_y = self.Y[:,start_index:end_index]

                # Calling the forward method that will return the ouput of all the layer in a list.
                self.all_layer_ouput = self.forward(current_X)

                # Storing the last layer output as it is the final predicted output
                Y_pred = self.all_layer_ouput[-1]

                #Before performing the back performing we need the derivative of loss function and it is the most important part for a  neural network and it will be used in all the other layers
                self.first_derivative = self.mse_loss_derivative(current_y,Y_pred)
                
                # Calling the backward function that will perform the backward propagation and it akes the derivative of the loss as the input which will be used for every layer
                self.backward(self.first_derivative)

                #We get the loss on the current batch
                current_loss = self.mse_loss(current_y,Y_pred)

                #Adding up all the loss from each batch 
                total_loss += current_loss

            #Storing the error of the current epoch.
            current_epoch_loss.append(total_loss / steps)   

            #Priting and storing the loss for each 100th epoch
            if epoch % 100 == 0:
                print("Epoch {}, loss {}".format(epoch,current_epoch_loss[-1]))
                # self.learning_rate = self.learning_rate * 0.9
                print(self.learning_rate)


        #Plotting the loss of each 100th iteration
        self.plot_loss(current_epoch_loss)

        # returning the best layer after updating their weights and biases
        return self.layers, self.activations


    def plot_loss(self,loss_on_each_epoch):
        """
            This plots the error of the given input
        """
        sns.lineplot(loss_on_each_epoch)
        plt.show()


    def predict(self,X_test,Y_test):
        """
        This function takes testing data, applies forward propagation and return the predicted output
        """

        Y_pred = self.forward(X_test)[-1]
        loss = self.mse_loss(Y_test,Y_pred)
        print("Loss {}, Rmse {}".format(loss, (loss)**(0.5)))
        return Y_pred
        

    def mse_loss(self,Y,Y_pred):
        """
        This will return tha Mean squared error
        """
        return np.mean((Y-Y_pred)**2)

    def mse_loss_derivative(self,Y,Y_pred):
        """
        This will return derivative of the Mean square error
        """
        return (-2 * (Y - Y_pred)/Y.shape[1])
    


    def sigmoid(self,z):
        """
        Sigmoid function
        """
        return (1 / (1 + np.exp(-z)))


    def relu(self,z):
        """
        Relu function
        """
        return np.maximum(0,z)
    
    def tanh(self,z):
        """
        tanh function
        """
        return np.tanh(z)

    def deriv_sigmoid(self,z):
        """
        Derivative of sigmoid function
        """
        return (self.sigmoid(z))*(1 - self.sigmoid(z))
    
    def deriv_relu(self,z):
        """
        Derivative of relu function
        """

        z[z<=0] = 0
        z[z>0] = 1
        return z
    
    def deriv_tanh(self,z):
        """
        Derivative of tanh function
        """
        return 1 - (np.tanh(z)**2)

    def forward(self,X):
        """
        This function performs the forward propagation, by performing the dot product of current_layer_weights and the incoming output from the previous layer and adding bias of the current layer and passing this layer output to the next layer and so on. 

        X : The dataset on which the forward propagation needs to be done
        """

        # At starting for the first layer the input from the previous layer is the actual data itself, in the all_layer_output the first output is the dataset itself.
        all_layers_output = [X.T]

        #The below code performs forward propagation , applies the passed activation function and store the output and pass this output to the next layer.
        for index in range(len(self.layers)):
            
            current_layer = self.layers[index]
            current_activation = self.activations[index]

            pre_layer_output = all_layers_output[-1]

            current_layer_output = np.dot(current_layer.weights.T,pre_layer_output) + current_layer.bias

            if current_activation == "sigmoid":
                current_layer_output = self.sigmoid(current_layer_output)
            elif current_activation == "relu":
                current_layer_output = self.relu(current_layer_output)
            elif current_activation == "tanh":
                current_layer_output = self.tanh(current_layer_output)
            elif current_activation == "linear" or current_activation == "None":
                pass
            else:
                print("{} activation is not implemented".format(current_activation))


            all_layers_output.append(current_layer_output)
        return all_layers_output
    


    def backward(self,current_derivative):
        """
        This is the backward propagation function that calculates the gradient of the current layer updates the curent layers weights and biases passes this layer gardient to the previous layer.
        The beginining of the gradient needs the derivative of the loss function thus we need paas the derivative of the loss function to this backward method        
        """
        prev_derivative = [current_derivative]

        for layer_index in range(len(self.layers)-1,-1,-1):

            current_layer = self.layers[layer_index]
            current_layer_activation = self.activations[layer_index]

            current_derivative = prev_derivative[-1]
            prev_layer_output = self.all_layer_ouput[layer_index]
            current_layer_output = self.all_layer_ouput[layer_index+1]

            
            if current_layer_activation == "sigmoid":
                deriv_wrt_activation = self.deriv_sigmoid(current_layer_output)
            elif current_layer_activation == "relu":
                deriv_wrt_activation = self.deriv_relu(current_layer_output)
            elif current_layer_activation == "tanh":
                deriv_wrt_activation = self.deriv_tanh(current_layer_output)
            elif current_layer_activation == "None" or current_layer_activation == "linear":
                deriv_wrt_activation = 1
            else:
                print("This activation function is not implemented yet")

            current_derivative = deriv_wrt_activation*current_derivative

            dw = (1/self.X.shape[0]) * np.dot(prev_layer_output,current_derivative.T)
            db = (1/self.X.shape[0]) * np.sum((current_derivative), axis = 1, keepdims = True)


            current_layer.weights -= self.learning_rate * dw
            current_layer.bias -= self.learning_rate * db

            updated_derivative = np.dot(current_layer.weights,current_derivative)
            
            prev_derivative.append(updated_derivative)

df = pd.read_csv("simple_polynomial_train_data.csv")

X_train = np.array(df.drop(["y", "Unnamed: 0"], axis = 1))  #5x2
Y = np.array(df["y"]).reshape(1,X_train.shape[0])           #1x5

# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)

# Best combination for simple -10 to 10 for polynomial regression
l1 = Layer(3,30)
a1 = "relu"
l2 = Layer(30,20)
a2 = "relu"
l3 = Layer(20,10)
a3 = "relu"
l4 = Layer(10,1)
a4 = "linear"
learning_rate = 0.005
epochs = 2000


# l1 = Layer(3,30)
# a1 = "relu"
# l2 = Layer(30,20)
# a2 = "relu"
# l3 = Layer(20,10)
# a3 = "relu"
# l4 = Layer(10,1)
# a4 = "linear"
# learning_rate = 0.0000001
# epochs = 8000


layers = [l1,l2,l3,l4]
activations = [a1,a2,a3,a4]

# print(X_train.shape)
nn = Network(layers, activations)
layers, activations = nn.fit(X_train,Y,learning_rate,epochs, 999)


# pickle.dump(layers, open("layers.pkl","wb"))
# pickle.dump(activations, open("activations.pkl","wb"))

layers = pickle.load(open("layers.pkl","rb"))
activations = pickle.load(open("activations.pkl","rb"))


# df = pd.read_csv("Linear_test_data.csv")
# X_test = np.array(df.drop(["y","Unnamed: 0"], axis = 1))
# y_test = np.array(df["y"]).reshape(1,X_test.shape[0])  

r = 10

X_test = np.array([[i,i,i] for i in range(-(r),r)])
Y_test = np.array([[(i[0]**3) + i[1]**2 + (i[2]+6)] for i in X_test])

# X_test = np.array([[i,i] for i in range(-10,10)])
# y_test = np.array([[(i[0]**2) + i[1]] for i in X_test])

# X_test = np.array([[i,i] for i in range(-10,10)])
# y_test = np.array([[(i[0]) + i[1]] for i in X_test])

# X_test = scaler.transform(X_test)


nn = Network(layers,activations)
Y_pred = nn.predict(X_test,Y_test)


plt.plot(range(-r,r),Y_test,linewidth = '10')
plt.plot(range(-r,r),Y_pred.T)
plt.show()


