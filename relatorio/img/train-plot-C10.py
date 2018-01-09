import matplotlib.pyplot as plt
import pandas as pd
import sys
import re
import os

color_dic = {
    'CIFAR-10': {'baseline': 'green', 'stn': 'blue'},
    'CIFAR-10-DISTORTED': {'baseline': 'orange', 'stn': 'red'},
}

fig, [train_ax, val_ax] = plt.subplots(1, 2, sharey=True)

train_ax.set_title('Training Accuracy')
val_ax.set_title('Validation Accuracy')

for filename in os.listdir(os.path.join(os.path.dirname(__file__), 'C10-training')):
    filepath = os.path.join(os.path.dirname(__file__), 'C10-training', filename)
    match = re.match(r'run_(CIFAR-10|CIFAR-10-DISTORTED)_(baseline|stn)_.*-tag-(acc|val_acc).csv', filename)
    if match:
        dataset, modelname, scalar = match.groups()
        print('dataset: {}, model: {}, scalar: {}'.format(dataset, modelname, scalar), file=sys.stderr) #!#
        df = pd.read_csv(filepath)
        if scalar == 'acc':
            ax = train_ax
            ls = '-'
        else:
            ax = val_ax
            ls = '--'
        ax.plot(df['Step'], df['Value'], color=color_dic[dataset][modelname],
            ls=ls, label='{} accuracy on {}'.format(modelname, dataset))
    else:
        print('Error!!')
        exit()

for ax in train_ax, val_ax:
    ax.set_xticks(range(0,20,2))
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlim([0,19])
    ax.set_ylim([0,1.1])
plt.legend()

save_path = os.path.join(os.path.dirname(__file__), 'C10-training-accs.png')
save = False
if save:
    plt.savefig(save_path)
else:
    plt.show()
