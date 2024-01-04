import csv
import numpy as np

# class GradientDescent:
#     def __init__(self, X, Y):
#         self.X = np.array(X)
#         self.Y = np.array(Y)
#         self.theta = [[0] * len(Y), [0] * len(Y)]

#     def predict(self):  
#         Y = []
#         i = 0
#         for index, x in enumerate(self.X):
#             estimatePrice = self.theta[0][index] + (self.theta[1][index] * x)
#             Y.append(estimatePrice)
#         print(f"predict: {Y}")
#         return Y

#     def training(self):
#         m = len(self.Y)
#         for i in range(7):
#             Ypredict = np.array(self.predict())
#             # print(f'Y = {self.Y}')
#             # print(f'Ypredict = {Ypredict}')
#             self.theta[0] = 0.01 * (1 / m) * np.subtract(Ypredict, self.Y)
            
#             self.theta[1] = 0.01 * (1 / m) * np.matmul(np.subtract(Ypredict, self.Y), self.X)
#             # print(f'theta[0] = {self.theta[0]}, theta[1]= {self.theta[1]}')
#             print(f'f{np.sum(np.matmul(np.subtract(Ypredict, self.Y), self.X))}')

#         test = self.theta[0] + (self.theta[1] * 166800)
#         # print(f'result 166800km ,5800 price to guess-> {test}, m = {m}')

#     # def addThetaValues(self, theta0, theta1):
#     #     self.theta0Values.append(theta0)
#     #     self.theta1Values.append(theta1)
#     #     with open(self.parameters_path, mode='w') as param_file:
#     #         param_file.write("theta0,theta1\n")
#     #         for i in range(len(self.theta0Values)):
#     #             param_file.write(str(self.theta0Values[i]))
#     #             param_file.write(",")
#     #             param_file.write(str(self.theta1Values[i]))
#     #             param_file.write("\n")

class GradientDescent:
    def __init__(self, X, Y):
        self.X = np.array(X)
        self.X = np.vstack((np.ones((self.X.size, )), X)).T
        self.Y = np.array(Y).reshape(len(Y), 1)
        self.theta = np.zeros((2, 1))
        print(f'Y = {self.Y}, \n X = {self.X}')

    
    def training(self, learning_rate=0.00001, n_iterations=1000):
        m = len(self.Y)
        for iteration in range(n_iterations):
            Y_predict = np.dot(self.X, self.theta)
            # Mise à jour simultanée des paramètres theta_0 et theta_1
            cost = (1 / (2*m)) * np.sum(np.square(Y_predict - self.Y ))
            # print(Y_predict - self.Y)
            d_theta = (1/m) * np.dot(self.X.T, Y_predict - self.Y )
            if iteration % 100 == 0:
                print(f'dtheta = {d_theta} selftheta = {self.theta} newvalue = {self.theta - (learning_rate * d_theta)} predictY = {Y_predict}')
            self.theta = self.theta - (learning_rate * d_theta)
            # print(self.theta, "\ncost", cost)
        print("Paramètres finaux après descente de gradient :")
        print(self.theta)

        return self.theta


def main():
    X = []
    Y = []
    theta0Values = []
    theta1Values = []
    learningRate = 0.10
    data_path = 'data.csv'
    parameters_path = 'parameters.csv'
    
    # open data file
    try:
        with open(data_path, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                X.append(float(row["km"]))
                Y.append(float(row["price"]))
    except FileNotFoundError:
        print(f'File {data_path} not found.')
        return
    except Exception as e:
        print(f"An error occurred: {e} is not in csv")
        return
    # open parameters file
    try:
        with open(parameters_path, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                theta0Values.append(float(row["theta0"]))
                theta1Values.append(float(row["theta1"]))
    except FileNotFoundError:
        print(f'File {parameters_path} not found, starting theta values with 0.')
    except Exception as e:
        print(f'An error occurred: {e}, quitting the program')
        return
    print(f'X = {X}, Y = {Y}')
    gradientAlgo = GradientDescent(X, Y)
    theta = gradientAlgo.training()
    test = theta[0] + (theta[1] * 166800)
    print(f'result 166800km ,5800 is the price to guess-> {test}')   

if __name__ == "__main__":
    main()