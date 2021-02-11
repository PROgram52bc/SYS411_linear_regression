import sys
import numpy as np
import matplotlib.pyplot as plt


def h(theta_0, theta_1, x):
    """the hypothesis function y = theta_0 + theta_1 * x

    :theta_0: the constant
    :theta_1: the slope
    :x: the input value
    :returns: the y value

    """
    return theta_0 + theta_1 * x

def get_cost(theta_0, theta_1, h, data):
    """the cost function

    :theta_0: the specified theta_0 (constant)
    :theta_1: the specified theta_1 (slope)
    :h: the hypothesis function, taking theta_0, theta_1, and x, returning y
    :data: an iterable of tuples, each containing (x_i, y_i)
    :returns: the cost

    """
    total_cost = 0
    m = 0
    for x, y in data:
        diff = h(theta_0, theta_1, x) - y
        total_cost += diff * diff
        m += 1
    total_cost /= 2 * m
    return total_cost
    

def gd(theta_0, theta_1, h, data, alpha=1):
    """gradient descent function

    :theta_0: the initial theta_0 (constant)
    :theta_1: the initial theta_1 (slope)
    :h: the hypothesis function, taking theta_0, theta_1, and x, returning y
    :data: an iterable of tuples, each containing (x_i, y_i)
    :alpha: the step of the descent, default to 1
    :returns: the updated theta_0 and theta_1 as a tuple

    """
    step_theta_0 = 0
    step_theta_1 = 0
    m = 0
    for x, y in data:
        step_theta_0 += h(theta_0, theta_1, x) - y
        step_theta_1 += (h(theta_0, theta_1, x) - y) * x
        m += 1

    theta_0 -= alpha * step_theta_0 / m
    theta_1 -= alpha * step_theta_1 / m

    return theta_0, theta_1

def main():
    # initialize data
    x_data = [1,2,3]
    y_data = [9,5,3]
    if len(sys.argv) == 3:
        print("Reading training data from files...")
        with open(sys.argv[1]) as x_data_file:
            x_data = [ float(x) for x in x_data_file ]
        with open(sys.argv[2]) as y_data_file:
            y_data = [ float(y) for y in y_data_file ]
    else:
        print("Using default values")
    print("x_data: {}".format(x_data))
    print("y_data: {}".format(y_data))
    data = list(zip(x_data, y_data))

    # initialize plot
    fig, (plt_linreg, plt_cost) = plt.subplots(2)
    fig.suptitle("Training Visualization")
    plt_linreg.scatter(np.array(x_data), np.array(y_data))

    # set parameters
    theta_0, theta_1 = 0, 1
    alpha = 0.003
    max_repetition = 10000
    threshold = 0.0001

    # perform gradient descent
    for it in range(max_repetition):
        cost = get_cost(theta_0, theta_1, h, data)
        theta_0, theta_1 = gd(theta_0, theta_1, h, data, alpha)
        if cost < threshold:
            break
        if it % 100 == 0:
            print("it: {}, cost: {}".format(it, cost))
            # plot the line
            line_x = np.arange(min(x_data),max(x_data),0.1)
            line_y = h(theta_0, theta_1, line_x)
            line = plt_linreg.plot(line_x, line_y)
            # TODO: 
            # try to not remove the line at the last iteration
            # <2021-02-11, David Deng> #
            plt_cost.plot(it, cost, "-ok")
            plt.pause(0.05)
            line.pop(0).remove()

    # print result
    plt.show()
    print("cost: {}".format(get_cost(theta_0, theta_1, h, data)))
    print("theta_0: {}; theta_1: {}".format(theta_0, theta_1))

main()
