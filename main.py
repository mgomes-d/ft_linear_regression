import csv
import numpy as np

class GradientDescent:
    def __init__(self, x, y, theta0Values, theta1Values, parameters_path):
        self.x = np.column_stack((x, y))
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
        m = len(self.y)
        theta0 = self.theta[0]
        # print(f'x = {self.x} theta = {self.theta}')
        for i in range(10):
            x_array = np.array(self.x)
            theta_array = np.array(self.theta)
            y_array = np.array(self.y)
            # print(f'xarr = {x_array } theta = {theta_array} y_array = {y_array}')
            predictionY = np.matmul(x_array, theta_array)
            # print(predictionY)
            sub = np.subtract(y_array, predictionY)
            cost = 0.5 * np.mean(np.square(self.y - predictionY))
            # print(f'Cost: {cost}')
            xTranspose = x_array.T
            dtheta = (1 / m) * np.matmul(x_array.T, sub)
            # print(f'dtheta = {xTranspose}')
            self.theta = self.theta - (0.10 * dtheta)
        
        print(f'theta = {self.theta}')
        # self.addThetaValues(self.theta[0], self.theta[1])

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
                x.append(int(row["km"]))
                y.append(int(row["price"]))
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
                theta0Values.append(int(row["theta0"]))
                theta1Values.append(int(row["theta1"]))
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