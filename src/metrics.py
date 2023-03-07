
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def calculate_eer(y_true, y_score):
    '''
    Returns the equal error rate for a binary classifier output.
    '''
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    # plt.plot(fpr, tpr)
    # plt.savefig("hey.png")
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer


def calculate_far_frr(cm):
    far, frr = 0, 0
    if cm["FP"] + cm["TN"] > 0:
        far = cm["FP"] / (cm["FP"] + cm["TN"])
    if cm["FN"] + cm["TP"] > 0:
        frr = cm["FN"] / (cm["FN"] + cm["TP"])
    
    return far, frr
