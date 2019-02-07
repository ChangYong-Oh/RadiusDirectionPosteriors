import socket


def data_dir():
    if socket.gethostname() == 'peters-MBP':
        return '/Users/changyong/data'
    elif socket.gethostname() == 'DTA160000':
        return '/home/coh1/Data'
    elif 'node' == socket.gethostname()[:4]:
        return '/var/scratch/coh/Data'
    else: #Any machine in MPI Tuebingen
        return '/is/ei/com/data'


def exp_dir():
    if socket.gethostname() == 'peters-MBP':
        return '/Users/changyong/Experiments/BayesianNeuralNetwork'
    elif socket.gethostname() == 'DTA160000':
        return '/home/coh1/Experiments/BayesianNeuralNetwork'
    elif 'node' == socket.gethostname()[:4]:
        return '/var/scratch/coh/Experiments/BayesianNeuralNetwork'
    elif socket.gethostname() == 'hekla':
        return '/is/ei/com/Experiments/BayesianNeuralNetwork'
    else:  # Any other machine in MPI Tuebingen
        return '/home/com/Experiments/BayesianNeuralNetwork'


EXP_DIR = exp_dir()
DATA_DIR = data_dir()