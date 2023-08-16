import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from CV19DataSet import CV19DataSet
from utils import compute_AUCs
import pandas as pd
import argparse
from torchvision.models import efficientnet_v2_m

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
N_CLASSES = 2
CLASS_NAMES = ['Covid', 'Non-covid']
def main(root_dir, df, num_models, run_name, BATCH_SIZE = 128, model_folder = '', input_size = 224):
    
    DATA_DIR = root_dir
    ckpt_folder = './{}/'.format(model_folder) + run_name + '/'
    normalizer = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    transformSequence_test = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(normalizer[0], normalizer[1])])
    
    test_dataset = CV19DataSet(df=df, base_folder=DATA_DIR, transform=transformSequence_test,img_size=input_size)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=6, pin_memory=True, drop_last=False,persistent_workers=True)
    pred_np_total = np.zeros((len(df),num_models))
    for model_index in range(num_models):
        model_name = ckpt_folder + 'EfficientNet_train_' + str(model_index) + '.pth.tar'
        print('model path:', model_name)  

        cudnn.benchmark = True
        # initialize and load the model
        model = efficientnet_v2_m(weights=None,drop_rate = 0.2)
        model.classifier[1] = nn.Sequential(nn.Linear(1280, N_CLASSES), nn.Softmax(dim=1))
        model = nn.DataParallel(model.cuda() ,device_ids=[0])

        if os.path.isfile(model_name):
            checkpoint = torch.load(model_name)
            state_dict = checkpoint['state_dict']
            model.load_state_dict(state_dict)
        else:
            print("=> no checkpoint found")

        # Testing mode 
        gt_np, pred_np = epochVal(model, test_loader)
        alpha = 0.95
        auc_result = compute_AUCs(gt_np, pred_np)
      
        
        print('AUC = {:.3f}'.format(auc_result))
        pred_np_total[:,model_index] = pred_np
        
    pred_np_ensemble = np.sqrt(np.mean(pred_np_total**2, axis=1))
    auc_result = compute_AUCs(gt_np, pred_np_ensemble)
 
    print('Ensemble AUC = {:.3f}'.format(auc_result))
    return pred_np_ensemble, gt_np, auc_result, ci
    #-------------------------------------------------------------------------------- 
def epochVal (model, dataLoader):
    model.eval()
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()
    with torch.no_grad():
        for i, (inp, target) in enumerate(dataLoader):
            target = target.cuda()
            gt = torch.cat((gt, target), 0)
            
            output = model(inp.cuda())
            pred = torch.cat((pred, output.data), 0)
                
    torch.cuda.empty_cache()       
    gt_np = gt.cpu().detach().numpy()
    pred_np = pred.cpu().detach().numpy()
    del gt, pred
    return gt_np[:,0], pred_np[:,0] 
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type = str)
    parser.add_argument('test_name', type = str)
    parser.add_argument('num_of_ensemble_models', type = int, help='10')
    parser.add_argument('delta_min', type = float)
    parser.add_argument('delta_max', type = float)
    parser.add_argument('model_folder', type = str)
    parser.add_argument('input_size', type = int)
    
    
    args = parser.parse_args()
    run_name = args.run_name
    test_name = args.test_name
    num_models = args.num_of_ensemble_models
    root_dir = 'E:/data/Total_png/'
    delta_min = args.delta_min
    delta_max = args.delta_max
    model_folder = args.model_folder
    input_size = args.input_size
    
    csv_path = '../CSV_files/' + test_name + '.csv'
    
    df = pd.read_csv(csv_path)
    df = df[(df.delta<=delta_max) & (df.delta>=delta_min)]
    pred_ensemble, gt, auc, ci = main(root_dir, df, num_models, run_name, BATCH_SIZE=256, model_folder = model_folder, input_size = input_size)
    df["covid_score_HF"] = pred_ensemble
    csv_path_result = './Result_csv/' + test_name + '_result_{}.csv'.format(model_folder)
    df.to_csv(csv_path_result,index=False)
