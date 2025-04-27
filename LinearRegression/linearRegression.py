# f(X) = W^TX+b
# 1. datasets, labels are needed in regression
# 2. calculate MSE
# 3. update W by GD methods and a defined learning rate
import numpy as np
from utils import prepare_for_training

class LinearRegression:
    def __init__(self, data, labels, polynomial_degree, sinusoid_degree, normalize_data):
        """
        1. preprocess data.
        2. number of feature vectors are defined
        3. for $f(X) = XW$, W is defined as weight.
        """
        # preprocess is done.
        processed_data = prepare_for_training(data, polynomial_degree, sinusoid_degree, normalize_data = normalize_data)
        #construction function
        self.data = processed_data
        self.labels = labels
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data
        # column nums of data, indicating number of feature vectors
        self.num_features = self.data.shape[1]      
        #[[0],[0],...[0]] indicates a num_features * 1 column zero vector
        self.weight = np.zeros((self.num_features, 1))
    

    def train(self, eta, num_iterations = 500):
        """
        update W by GD methods and a defined learning rate eta.
        num_iterations: defined to stop updating.
        
        """
        self.GD(eta, num_iterations)
        
    def GD(self, eta, num_iterations = 500):
        cost_history = []
        for _ in range(num_iterations):
            self.__BGD_step(eta)
            cost_history.append(self.loss_function(self.data, self.labels))
        return cost_history
     
    def __BGD_step(self, eta):
        """
        weight is updated once for optimization.
        Args:
            eta (number): learning rate
        """
        transposed_data = self.data.T
        # number of samples
        rows = self.data.shape[0]
        predictions = LinearRegression.regression(self.data, self.weight)
        delta = predictions - self.labels
        
        # MSE is calculated.
        nabla_MSE = 2/rows * np.dot(transposed_data, delta)
        self.weight = self.weight - eta * nabla_MSE
        
    @staticmethod
    def regression(data, weight):
        # f(X) = XW
        return np.dot(data, weight)
    
    def loss_function(self, data, labels):
        """MSE is calculated.

        Returns:
            _type_: number
        """
        rows = self.data.shape[0]
        # for testing or either. Weight is not updated
        delta = LinearRegression.regression(data, self.weight) - labels
        loss = (1/rows) * np.dot(delta.T, delta)
        print("loss value: ", loss)
        return loss
    
    def predict(self, test_data):
        processed_data = prepare_for_training(test_data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data)
        # for testing
        LinearRegression.regression(processed_data, self.weight)
        
    def get_loss(self, data, labels):
        data_processed = prepare_for_training(data, 
                                              self.polynomial_degree, 
                                              self.sinusoid_degree, 
                                              self.normalize_data)
        
        self.loss_function(data_processed, labels)