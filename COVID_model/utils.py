import os
from sklearn.metrics import roc_auc_score

def mkdir(path):
    """build a new folder 
    """
    folder = os.path.exists(path)
    if not folder:                   
        os.makedirs(path)            
        print('---  new folder  ---')
        print('---  OK  ---')



def compute_AUCs(gt_np, pred_np):
    """Computes Area Under the Curve (AUC) from prediction scores
    """
    AUROCs = []   
    AUROCs.append(roc_auc_score(gt_np[:, 0], pred_np[:, 0]))
    
    return AUROCs

