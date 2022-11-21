import matplotlib.pyplot as plt


def plot_2ts_yinvert(xvalues, y1values, y2values, color1="red", color2="blue", xlabel="dates", y1label=None,
                     y2label=None):
    """
    Plot y1values and y2values, but inverts y2 axis.
    y1values and y2values must have same lenght, equal to len(xvalues)
    :param xvalues:
    :param y1values:
    :param y2values:
    :param color1:
    :param color2:
    :param xlabel:
    :param y1label:
    :param y2label:
    :return:
    """

    fig, ax1 = plt.subplots()

    ax1.set_xlabel(xlabel)
    if y1label is not None:
        ax1.set_ylabel(y1label, color=color1)
    ax1.plot(xvalues, y1values, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    # Adding Twin Axes
    ax2 = ax1.twinx()

    if y2label is not None:
        ax2.set_ylabel(y2label, color=color2)
    ax2.plot(xvalues, y2values, color=color2)
    ax2.invert_yaxis()
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.show()

    return None
