import csv
import numpy as np

class GradientDescent:
    def __init__(self, x, y, theta0Values, theta1Values, parameters_path):
        # self.x = np.column_stack((x, y))
        self.x = x
        self.y = y
        self.theta0Values = theta0Values
        self.theta1Values = theta1Values
        self.parameters_path = parameters_path
        self.theta = []
        if theta0Values and theta1Values:
            self.theta = [theta0Values[-1], theta1Values[-1]]
        else:
            self.theta = [0, 0]

    def training_algo(self):
        theta = [0, 0]
        # print(f'x = {self.x} theta = {self.theta}')
        x_array = np.array(self.x)
        y_array = np.array(self.y)
        theta0 = 0
        theta1 = 0
        m = len(self.x)
        for i in range(10000):
            j = 0
            estimatePrice = 0
            while j < (m - 1):
                estiPrice = theta0 + (theta1 * self.x[i)
                tempestimatePrice = estiPrice - self.y[j]
                estimatePrice += tempestimatePrice
                j = j + 1
            print(estimatePrice)
            theta0 = 0.10 *( 1 / m ) * estimatePrice

            estimatePrice = 0
            j = 0
            while j < (m - 1):
                estiPrice = theta0 + (theta1 * self.x[j])
                tempestimatePrice = (estiPrice - self.y[j]) * self.x[j]
                estimatePrice += tempestimatePrice
                j = j + 1
            print(estimatePrice)
            theta1 = 0.10 *(1 / m )* estimatePrice
            print(theta1)
            # while i < (m - 1):
            #     tempestimatePrice = self.x[i] - self.y[i]
            #     estimatePrice += tempestimatePrice 
            #     i = i + 1



            # theta_array = np.array(theta)
            # # print(f'xarr = {x_array } theta = {theta_array} y_array = {y_array}')
            # predictionY = np.matmul(x_array, theta_array)
            # print(f'predictionY = {predictionY}')
            # sub =  np.subtract(y_array, predictionY)
            # cost = (1 / (2 * m)) * np.square(self.y - predictionY)

            # print(f'Cost: {cost}, sub = {sub}')
            # xTranspose = x_array.T
            # # print(xTranspose)
            # dtheta = (1 / m) * np.matmul(xTranspose, sub)
            # print(f'dtheta = {dtheta}')
            # theta = theta - 0.000001 * dtheta
        
        print(f'theta = {theta0}, theta1 = {theta1}')
        test = theta0 + (theta1 * 139800)
        print(f'estimateprice 139800 km = {test} -> real price 3800 ')

        # self.addThetaValues(self.theta0, self.theta[1])

        # self.addThetaValues(5, 1)

    def addThetaValues(self, theta0, theta1):
        self.theta0Values.append(theta0)
        self.theta1Values.append(theta1)
        with open(self.parameters_path, mode='w') as param_file:
            param_file.write("theta0,theta1\n")
            for i in range(len(self.theta0Values)):
                param_file.write(str(self.theta0Values[i]))
                param_file.write(",")
                param_file.write(str(self.theta1Values[i]))
                param_file.write("\n")

def main():
    x = []
    y = []
    theta0Values = []
    theta1Values = []
    data_path = 'data.csv'
    parameters_path = 'parameters.csv'
    
    # open data file
    try:
        with open(data_path, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                x.append(float(row["km"]))
                y.append(float(row["price"]))
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
    print(theta1Values)
    gradientAlgo = GradientDescent(x, y, theta0Values, theta1Values, parameters_path)
    gradientAlgo.training_algo()

if __name__ == "__main__":
    main()