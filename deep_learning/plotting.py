from matplotlib import pyplot as plt

# To plot both training and testing cost/loss ratio
def plot_loss(train_loss, valid_loss):
    fig, ax = plt.subplots()
    fig_size = [12, 9]
    plt.rcParams["figure.figsize"] = fig_size
    plt.plot(train_loss, label='Training Cost')
    plt.plot(valid_loss, label='Validation Cost')
    plt.title('Cost over time during training')
    legend = ax.legend(loc='upper right')
    plt.show()


# To plot both training and validation accuracy
def plot_acc(train_acc, valid_acc):
    fig, ax = plt.subplots()
    fig_size = [12, 9]
    plt.rcParams["figure.figsize"] = fig_size
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(valid_acc, label='Validation Accuracy')
    plt.title('Accuracy over time during training')
    legend = ax.legend(loc='upper left')
    plt.show()



training_acc = [0.42, 0.52, 0.64, 0.66, 0.68, 0.69, 0.61, 0.64, 0.72, 0.69,
                0.70, 0.72, 0.65, 0.73, 0.75, 0.78, 0.80, 0.77, 0.78, 0.77,
                0.79, 0.81, 0.83, 0.78, 0.76, 0.77, 0.81, 0.86, 0.85, 0.82,
                0.81, 0.82, 0.84, 0.79, 0.83, 0.86, 0.84, 0.86, 0.85, 0.88,
                0.79, 0.83, 0.87, 0.84, 0.85, 0.86, 0.89, 0.93, 0.87, 0.93]

validation_acc = [0.40, 0.20, 0.60, 0.53, 0.53, 0.60, 0.67, 0.69, 0.68, 0.57,
                  0.57, 0.65, 0.67, 0.70, 0.72, 0.68, 0.67, 0.72, 0.74, 0.72,
                  0.79, 0.75, 0.76, 0.72, 0.77, 0.75, 0.78, 0.75, 0.72, 0.82,
                  0.83, 0.84, 0.78, 0.82, 0.76, 0.80, 0.81, 0.82, 0.82, 0.85,
                  0.81, 0.85, 0.87, 0.81, 0.84, 0.79, 0.85, 0.88, 0.85, 0.89
                  ]
plot_acc(training_acc, validation_acc)