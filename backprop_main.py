import backprop_data
import backprop_network
import matplotlib.pyplot as plt


def myplot(X, Ys, xlabel, ylabel, rates):
    """
    Plot all (x,y) for all Y in Ys in the same plot
    :param X: x values
    :param Ys: list of y values
    :param xlabel: x axis label
    :param ylabel: y axis label
    :param rates: SGD's learning rates
    """
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for i,Y in enumerate(Ys):
        plt.plot(X, Y, label=rates[i])
    plt.legend()
    plt.savefig(f"{ylabel}.pdf")
    plt.show()


def b():
    training_data, test_data = backprop_data.load(train_size=10000, test_size=5000)
    rates = [0.001, 0.01, 0.1, 1, 10, 100]
    train_acc_list, train_loss_list, test_acc_list = [], [], []

    for rate in rates:
        net = backprop_network.Network([784, 40, 10])
        train_acc, train_loss, test_acc = net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=rate, test_data=test_data)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        test_acc_list.append(test_acc)

    epochs = range(30)
    myplot(epochs, train_acc_list, "epochs", "Train Accuracy", rates)
    myplot(epochs, train_loss_list, "epochs", "Train CE Loss", rates)
    myplot(epochs, test_acc_list, "epochs", "Test Accuracy", rates)


def c():
    training_data, test_data = backprop_data.load(train_size=50000, test_size=10000)
    net = backprop_network.Network([784, 40, 10])
    net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)
    print(net.one_label_accuracy(test_data))


def main():
    b()
    # c()


if __name__ == "__main__":
    main()
