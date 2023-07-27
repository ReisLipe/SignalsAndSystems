import matplotlib.pyplot as plt
from numpy import sin, cos, pi, linspace
from scipy.integrate import quad as integral
from celluloid import Camera


# Function for Fourier Series Analysis
def desired_function(t):
    return t ** 2


def a_zero(function, half_period, limits):
    """
    Calculates the value of A0 for a desired function.

    :param function: The desired function
    :param half_period: The half period of the function
    :param limits: The upper and lower limits of the analyzed function
    :return: The value of A0
    """
    return (1 / (2 * half_period)) * integral(function, *limits)[0]


def a_k(k, half_period, limits):
    """
    Calculates the value of Ak for a desired function.

    :param k: The current harmonic value
    :param half_period: The half period of the function
    :param limits: The upper and lower limits of the analyzed function
    :return: The value of Ak
    """
    return (1 / half_period) * integral(lambda x: desired_function(x) * cos(x * k * (pi / half_period)), *limits)[0]


def b_k(k, half_period, limits):
    """
    Calculates the value of Bk for a desired function.

    :param k: The current harmonic value
    :param half_period: The half period of the function
    :param limits: The upper and lower limits of the analyzed function
    :return: The value of Bk
    """
    return (1 / half_period) * integral(lambda x: desired_function(x) * sin(x * k * (pi / half_period)), *limits)[0]


def main(total_harmonics=30, num_points=1000):
    fig = plt.figure()
    camera = Camera(fig)

    limits = (-pi, pi)  # Lower limit = -pi, upper limit = pi
    half_period = (limits[1] - limits[0]) / 2  # Half period is half of a fundamental period
    x = linspace(*limits, num_points)  # Values of x ranging from the lower limit to the upper limit with n points in between

    f = a_zero(desired_function, half_period, limits)

    # Summation of harmonics from 1 to the total number of harmonics
    for k in range(1, total_harmonics + 1):
        trigonometric_argument = k * x * (pi / half_period)
        cosoidal_part = a_k(k, half_period, limits) * cos(trigonometric_argument)
        sinusoidal_part = b_k(k, half_period, limits) * sin(trigonometric_argument)
        f += cosoidal_part + sinusoidal_part

        # Plotting the graph and creating the animation
        plt.plot(x, desired_function(x), color='k')
        plt.plot(x, f, label=f'k = {k}')
        camera.snap()

    plt.style.use('bmh')
    plt.legend()
    plt.show()

    animation = camera.animate()
    plt.close()
    animation.save('animation.gif')


if __name__ == '__main__':
    main()
