import csv
import numpy as np
import matplotlib.pyplot as plt

class GradientDescent:
    def __init__(self, X, Y):
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.theta = np.zeros((2, 1))
        print(f'X = {self.X} Y = {self.Y}')
    def training(self):
        m = len(self.Y)
        for i in range(50):
            error = 0
            for j in range(m - 1):
                estimatePrice = self.theta[0] + (self.theta[1] * self.X[j])
                price = self.Y[j]
                # print(self.X[j], self.Y[j])
                error += estimatePrice - price
            # print(error)
            temptheta0 = 0.00000000000975 * ((1 / m) * error)

            error = 0
            for j in range(m - 1):
                estimatePrice = self.theta[0] + (self.theta[1] * self.X[j])
                price = self.Y[j]
                error += (estimatePrice - price) * self.X[j]
                
            temptheta1 = 0.00000000000975 * ((1 / m) * error)
            # print(f'f{np.sum(np.matmul(np.subtract(Ypredict, self.Y), self.X))}')
            self.theta[0] -= temptheta0
            self.theta[1] -= temptheta1
        print(f'theta[0] = {self.theta[0]}, theta[1]= {self.theta[1]}')
        # for i in range(len(self.X)):
        #     self.checking_value(self.X[i],self.Y[i])

    def checking_value(self, km, price):
        test = self.theta[0] + (self.theta[1] * km)
        print(f'result {km}km ,guess price -> {test}, actual price {price}, theta0= {self.theta[0]} and theta1 = {self.theta[1]}')

    def plot_regression_line(self):
        # Créer des points pour la ligne de régression
        x_line = np.linspace(min(self.X), max(self.X), 100)
        y_line = self.theta[0] + (self.theta[1] * x_line)

        # Tracer le scatter plot
        plt.scatter(self.X, self.Y, color='blue', label='Data Points')

        # Tracer la ligne de régression
        plt.plot(x_line, y_line, color='red', label='Linear Regression Line')

        plt.title('Scatter Plot with Linear Regression Line')
        plt.xlabel('Kilometers (km)')
        plt.ylabel('Price')
        plt.legend()
        plt.show()


    # def addThetaValues(self, theta0, theta1):
    #     self.theta0Values.append(theta0)
    #     self.theta1Values.append(theta1)
    #     with open(self.parameters_path, mode='w') as param_file:
    #         param_file.write("theta0,theta1\n")
    #         for i in range(len(self.theta0Values)):
    #             param_file.write(str(self.theta0Values[i]))
    #             param_file.write(",")
    #             param_file.write(str(self.theta1Values[i]))
    #             param_file.write("\n")

# class GradientDescent:
#     def __init__(self, X, Y):
#         self.X = np.array(X)
#         self.X = np.vstack((np.ones((self.X.size, )), X)).T
#         self.Y = np.array(Y).reshape(len(Y), 1)
#         self.theta = np.zeros((2, 1))
#         print(f'Y = {self.Y}, \n X = {self.X}')

    
#     def training(self, learning_rate=0.00001, n_iterations=1000):
#         m = len(self.Y)
#         for iteration in range(n_iterations):
#             Y_predict = np.dot(self.X, self.theta)
#             # Mise à jour simultanée des paramètres theta_0 et theta_1
#             cost = (1 / (2*m)) * np.sum(np.square(Y_predict - self.Y ))
#             # print(Y_predict - self.Y)
#             d_theta = (1/m) * np.dot(self.X.T, Y_predict - self.Y )
#             if iteration % 100 == 0:
#                 print(f'dtheta = {d_theta} selftheta = {self.theta} newvalue = {self.theta - (learning_rate * d_theta)} predictY = {Y_predict}')
#             self.theta = self.theta - (learning_rate * d_theta)
#             # print(self.theta, "\ncost", cost)
#         print("Paramètres finaux après descente de gradient :")
#         print(self.theta)

#         return self.theta


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
                X.append(int(row["km"]))
                Y.append(int(row["price"]))
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
    gradientAlgo.plot_regression_line()
    # test = theta[0] + (theta[1] * 166800)
    # print(f'result 166800km ,5800 is the price to guess-> {test}')   

if __name__ == "__main__":
    main()