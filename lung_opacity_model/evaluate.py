import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from MIMIC_dataset import MIMIC_dataset
import pandas as pd
from torchvision.models import densenet121
from utils import compute_AUCs
import scipy 

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

N_CLASSES = 2
CLASS_NAMES = ['abnormal', 'normal']
DATA_DIR = 'c:/data/covid/Total_png/NonCovid/'

#########################################################################
def test(df, save_folder, model_index, BATCH_SIZE = 40):
    
    ckpt_path = save_folder + 'DenseNet_train_' + str(model_index) + '.pth.tar'
    cudnn.benchmark = True

   
    dev_count = torch.cuda.device_count()
    print('Using {} GPUs'.format(dev_count))
    
    model = densenet121(weights=None,drop_rate = 0.2)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(num_ftrs, N_CLASSES), nn.Softmax(dim=1))
    model = nn.DataParallel(model.cuda())

    if os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['state_dict'])
        print('loaded checkpoint for model_opacity')
    else:
        print('no checkpoint found for model_opacity')
    normalizer = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    transformList_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalizer[0], normalizer[1])])
    
    test_dataset = MIMIC_dataset(df=df, base_folder=DATA_DIR, transform=transformList_test)
    
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=4, pin_memory=True, drop_last=False, persistent_workers=True)
    print("Test data size: ",len(test_loader.dataset))
    
    
    prediction_score, label = epochVal(model, test_loader)
    auc_result = compute_AUCs(label, prediction_score)

    print('AUC = {:.3f}'.format(auc_result))
    

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    thres = 0.2
    for ij in range(len(prediction_score)):
        if label[ij] == 1: # covid
            if prediction_score[ij] > thres:
                TP = TP + 1
            else:
                FN = FN +1 
                
        else:  # non-covid 
            if prediction_score[ij] < thres:
                TN = TN +1
            else:
                FP = FP +1 
    print('sensitivity', TP/(TP+FN))
    print('spercificty', TN/(TN+FP))
    


def epochVal (model, dataLoader):
    model.eval()
    gt = torch.zeros(1, N_CLASSES)
    gt = gt.cuda()
    pred = torch.zeros(1, N_CLASSES)
    pred = pred.cuda()
    
        
    with torch.no_grad():
        for i, (inp, target) in enumerate (dataLoader):
            input_var = Variable(inp).cuda()
            vartarget = Variable(target).cuda()    
        
            varOutput = model(input_var)
            pred = torch.cat((pred, varOutput.data), 0)
            target = target.cuda()
            gt = torch.cat((gt, target), 0)    
        del input_var, vartarget, varOutput, target
        torch.cuda.empty_cache()
            
    gt_np = gt.cpu().detach().numpy()
    pred_np = pred.cpu().detach().numpy()
    del gt, pred
    torch.cuda.empty_cache()
    gt_np = gt_np[1: gt_np.shape[0],0]
    pred_np = pred_np[1: pred_np.shape[0],0]
        
    return pred_np, gt_np

if __name__ == '__main__':

    pretrain = True
    init_index = 0
    num_models = 1 
    folder = './Data/Model_MIMIC_Opacity/'
    csv_path = folder + 'Val_0.csv'
    df = pd.read_csv(csv_path)

    for model_index in range(num_models):
        model_index = init_index + model_index
        test(df, folder, model_index, BATCH_SIZE = 128)
        
    


