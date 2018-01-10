import matplotlib.pyplot as plt
import os

# Baseline data
baseline_losses = [
    1.4331,
    0.5697,
    0.3748,
    0.2632,
    0.1878,
    0.1331,
    0.1098,
    0.0838,
    0.0514,
    0.0203,
]

baseline_accs = [
    0.5275,
    0.8130,
    0.8799,
    0.9181,
    0.9412,
    0.9614,
    0.9683,
    0.9758,
    0.9880,
    0.9978,
]

baseline_val_losses = [
    0.7188,
    0.4880,
    0.4289,
    0.3654,
    0.3601,
    0.3531,
    0.3781,
    0.3521,
    0.3600,
    0.3851,
]

baseline_val_accs = [
    0.7650,
    0.8420,
    0.8730,
    0.8860,
    0.8890,
    0.8900,
    0.8970,
    0.9040,
    0.9020,
    0.9050,
]

# STN data.
stn_losses = [
    1.2142,
    0.3302,
    0.2122,
    0.1463,
    0.1274,
    0.0921,
    0.0745,
    0.0552,
    0.0365,
    0.0444,
]

stn_accs = [
    0.6059,
    0.8959,
    0.9345,
    0.9540,
    0.9591,
    0.9696,
    0.9766,
    0.9831,
    0.9902,
    0.9865,
]

stn_val_losses = [
    0.4529,
    0.2731,
    0.2262,
    0.1649,
    0.1985,
    0.1400,
    0.1351,
    0.1284,
    0.1289,
    0.1349,
]

stn_val_accs = [
    0.8680,
    0.9180,
    0.9330,
    0.9480,
    0.9450,
    0.9590,
    0.9580,
    0.9640,
    0.9680,
    0.9610,
]


fig, [train_ax, val_ax] = plt.subplots(1, 2, sharey=True)

train_ax.set_title('Training Accuracy')
train_ax.plot(range(10), baseline_accs, color='blue', label='Baseline accuracy')
train_ax.plot(range(10), stn_accs,      color='red',  label='STN accuracy')
val_ax.set_title('Validation Accuracy')
val_ax.plot(range(10), baseline_val_accs, color='blue', ls='--', label='Baseline accuracy')
val_ax.plot(range(10), stn_val_accs,      color='red',  ls='--', label='STN accuracy')

for ax in train_ax, val_ax:
    ax.set_xticks(range(0,10,1))
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlim([0,9])
    ax.set_ylim([0,1.1])
plt.legend()

save_path = os.path.join(os.path.dirname(__file__), 'CM-training-accs.png')
save = False
if save:
    plt.savefig(save_path)
else:
    plt.show()
