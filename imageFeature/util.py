import matplotlib.pyplot as plt
import numpy as np

def dataloader(imgRoot, labelTxtFile):
    pass

def randomShuffle():
    pass

def obtainMiniBatch():
    pass

def getAccuracy():
    pass

def plots(loss,acc,filename):
    dim = np.arange(1, len(loss), int(len(loss)/5))
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.set_title("Loss vs Trainig Iterations")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    ax1.set_xlim(1, len(loss))
    ax1.set_xticks(dim)
    ax1.plot(loss)

    dim = np.arange(1, len(acc), int(len(acc)/5))
    ax2.set_title("Accuracy vs Trainig Iterations")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_xlim(1, len(acc))
    ax2.set_xticks(dim)
    ax2.plot(acc)

    fig.savefig(filename)
    plt.show(block=False)