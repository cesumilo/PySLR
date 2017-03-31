import csv
import math

#Uncomment de following line to use pyplot :
#import matplotlib.pyplot as plt

def compute_error(w0, w1, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i][0]
        y = points[i][1]
        totalError += (y - (w0 + w1 * x)) ** 2
    return totalError

def step_gradient(w0, w1, points, learningRate):
    w0_gradient = 0
    w1_gradient = 0

    for i in range(0, len(points)):
        x = points[i][0]
        y = points[i][1]
        w0_gradient += -2 * (y - (w0 + (w1 * x)))
        w1_gradient += -2 * x * (y - (w0 + (w1 * x)))
    new_w0 = w0 - (learningRate * w0_gradient)
    new_w1 = w1 - (learningRate * w1_gradient)
    return [new_w0, new_w1]

def gradient_descent(points, starting_w0, starting_w1, learning_rate, num_iterations):
    w0 = starting_w0
    w1 = starting_w1
    for i in range(num_iterations):
        w0, w1 = step_gradient(w0, w1, points, learning_rate)
    return [w0, w1]

def normalize_data(x, y):
    x_mean = sum(x) / float(len(x))
    y_mean = sum(y) / float(len(y))

    x_v = []
    for value in x:
        x_v.append((value - x_mean) ** 2)
    y_v = []
    for value in y:
        y_v.append((value - y_mean) ** 2)

    x_deriv = math.sqrt(sum(x_v) / float(len(x_v)))
    y_deriv = math.sqrt(sum(y_v) / float(len(y_v)))

    x_new = []
    y_new = []
    for value in x:
        x_new.append((value - x_mean) / float(x_deriv))
    for value in y:
        y_new.append((value - y_mean) / float(y_deriv))

    return (x_new, y_new, x_mean, y_mean, x_deriv, y_deriv)

def get_sale_price(y, mean, deriv):
    return y * deriv + mean

def run():
    points = []
    x = []
    y = []

    with open('train.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        i = 0
        for row in reader:
            x.append(float(row['LotArea']))
            y.append(float(row['SalePrice']))
            if i == 100:
                break
            i += 1

    """ Uncomment this to plot dataset (100 points).
    for i in range(len(x)):
        plt.plot(x[i], y[i], 'r.')
    plt.savefig('visualize_data.png')
    """

    x, y, x_mean, y_mean, x_deriv, y_deriv = normalize_data(x, y)

    """ Uncomment this to plot normalized dataset (100 points).
    plt.clf()
    for i in range(len(x)):
        plt.plot(x[i], y[i], 'r.')
    plt.savefig('visualize_normalized_data.png')
    """

    for i in range(len(x)):
        points.append([ x[i], y[i] ])

    learning_rate = 0.0001
    initial_w0 = 0  # initial y-intercept guess
    initial_w1 = 0  # initial slope guess
    num_iterations = 100000

    print "Starting gradient descent at w0 = {0}, w1 = {1}, error = {2}".format(initial_w0, initial_w1, compute_error(initial_w0, initial_w1, points))
    print "Running..."

    w = gradient_descent(points, initial_w0, initial_w1, learning_rate, num_iterations)

    print "After {0} iterations w0 = {1}, w1 = {2}, error = {3}".format(num_iterations, w[0], w[1], compute_error(w[0], w[1], points))

    print "Exemple of predicted price vs real sale price : predicted = {0}, real = {1}".format(get_sale_price(w[0] + w[1] * x[0], y_mean, y_deriv), get_sale_price(y[0], y_mean, y_deriv))

    """ Uncomment this to plot the fitted line on the normalized data.
    line_x = []
    line_y = []
    plt.clf()
    for i in range(len(x)):
        plt.plot(x[i], y[i], 'r.')
        line_x.append(x[i])
        line_y.append(w[0] + w[1] * x[i])
        plt.plot(x[i], (w[0] + w[1] * x[i]))
    plt.plot(line_x, line_y, 'b-')
    plt.savefig('fitted_line.png')
    """

if __name__ == '__main__':
    run()