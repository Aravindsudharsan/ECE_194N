import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#loading the value from text file and converting to numpy array
data = np.loadtxt("housing_prices.txt",delimiter = ",")
print(data)
# splitting the 3 columns into seperate arrays
first_value,second_value,third_value=data.T
#printing the obtained values
print(first_value)
print(second_value)
print(third_value)

#getting the square feet , no of bedrooms and price data sets as X , X1 and Y and printing them
X1_value = first_value
X2_value = second_value
Y = third_value
print("Square feet set " , X1_value)
print("Bedroom set " , X2_value)
print("Price set " , Y)

#printing the shape of the arrays
print("shape of 1st set is",np.shape(X1_value))
print("shape of 2nd set is",np.shape(X2_value))
print("shape of 3rd set is",np.shape(Y))


# getting all but the last 10 samples and printing them aling with the shape
X_sqfeet = X1_value[0:37]
print("First 38 samples of sq feel",X_sqfeet)
print("Shape of final data set is",np.shape(X_sqfeet))
X_bedrooms = X2_value[0:37]
print("First 38 samples of sq feel",X_bedrooms)
print("Shape of final data set is",np.shape(X_bedrooms))
Y_price = Y[0:37]
print("First 38 samples of price",Y_price)
print("Shape of final label is",np.shape(Y_price))

# visualizing the data sets
plt.plot(X_sqfeet,Y_price,'ro',label = 'Square feet')
plt.legend()
plt.show()

plt.plot(X_bedrooms,Y_price,'ro',label = 'bedrooms')
plt.legend()
plt.show

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X_sqfeet, X_bedrooms, Y_price)
ax.set_xlabel('Sqaure feet')
ax.set_ylabel('Bedrooms')
ax.set_zlabel('Price')
plt.show()

#set model parameters
learning_rate = 0.01
iterations = 1000

weight = np.ones([1,3])
print("Shape of weight is",np.shape(weight))

# combining the 2 X data sets 
X = np.column_stack((X_bedrooms,X_sqfeet))
ones = np.ones([X.shape[0],1])
X = np.concatenate((ones,X),axis = 1)
print("New dataset is",X)
print("Shape is",np.shape(X))
Y1 = Y_price
print("Price is",Y1)
print("Shape is",np.shape(Y1))

# cost function
def cost_function(X,y,theta):
    add_result = np.power(((X @ theta.T) - y),2)
    return np.sum(add_result)/(2 * len(X))

#gradient descent
def gradient_function(X,y,theta,iters,alpha):
    cost = np.zeros(iters)
    for i in range(iters):
        theta = theta - (alpha/len(X)) * np.sum(X * (X @ theta.T - y), axis=0)
        cost[i] = cost_function(X, y, theta)
    
    return theta,cost

# working with normal data set
a = cost_function(X,Y1,weight)
print("Cost value of normal data is",a)

#running gd and cost function for normal data - overflown result
Y1 = np.reshape(Y1,(37,1))
print(np.shape(Y1))
g1,cost1 = gradient_function(X,Y1,weight,iterations,learning_rate)
print("Gradient is",g1)

predicted_output = cost_function(X,Y1,g1)
print("Minimized cost for normal data is",predicted_output)


#normalizing the data sets
def normalize_function(array):
	return (array - array.mean())/array.std()

# getting the normalized values
X1_normalize_data = normalize_function(X_sqfeet)
X2_normalize_data = normalize_function(X_bedrooms)
Y_normalize_data = normalize_function(Y_price)

Y_new = Y_price.mean()
Y_new1 = Y_price.std()

#printing the normalized values
print("Normalized X data" , X1_normalize_data)
print("Normalized X data" , X2_normalize_data)
print("Normalized Y data" , Y_normalize_data)

#visualizing the normalized values
plt.plot(X1_normalize_data,Y_normalize_data,'ro',label = 'Square feet')
plt.legend()
plt.show()

plt.plot(X2_normalize_data,Y_normalize_data,'ro',label = 'bedrooms')
plt.legend()
plt.show

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X1_normalize_data, X2_normalize_data, Y_normalize_data)
ax.set_xlabel('sqaure feet')
ax.set_ylabel('Bedrooms')
ax.set_zlabel('Price')
plt.show()

# working with normalized data
X1 = np.column_stack((X1_normalize_data,X2_normalize_data))
ones = np.ones([X1.shape[0],1])
X1 = np.concatenate((ones,X1),axis = 1)
print("New dataset is",X1)
print("Shape is",np.shape(X1))
Y2 = Y_normalize_data
print("Price is",Y2)
print("Shape is",np.shape(Y2))


b = cost_function(X1,Y2,weight)
print("Cost value for normalized data is",b)


#running the gd and cost function for normalized data
Y2 = np.reshape(Y2,(37,1))
print(np.shape(Y2))
g,cost = gradient_function(X1,Y2,weight,iterations,learning_rate)
print("Gradient is",g)

predicted_cost = cost_function(X1,Y2,g)
print("Minimized cost for normalized data is",predicted_cost)

predicted_new = (predicted_cost * Y_new1) + Y_new
print("Precited cost in normal form is", predicted_new)


#plot the cost
fig, ax = plt.subplots()  
ax.plot(np.arange(iterations), cost, 'r')  
ax.set_xlabel('Iterations')  
ax.set_ylabel('Cost')  
ax.set_title('Error vs. Training Epoch')  
plt.show()