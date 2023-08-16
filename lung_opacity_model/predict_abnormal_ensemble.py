import time
import pandas as pd
import argparse
import os
import torch.nn as nn
from torchvision.models import densenet121
import torch
import torchvision.transforms as transforms
from MIMIC_dataset import MIMIC_dataset
from torch.utils.data import DataLoader
import numpy as np
import glob
N_CLASSES = 2

def get_opacity_score(df, base_folder, ckpt_path, batch_size = 300, name = 'abnormality_score'):
    
    """
    get opacity/abnormality score for input

    Parameters
    ----------
    df : dataframe of test set
    ckpt_path : model weight filename

    Returns
    -------
    dataframe with scores
    """
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
    
    test_dataset = MIMIC_dataset(df=df, base_folder=base_folder, transform=transformList_test)
    
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=6, pin_memory=True, drop_last=False, persistent_workers=True)
    
     
    pred = torch.FloatTensor()
    pred = pred.cuda()
    model.eval()
    time_start = time.time()
    counter = 0
    num_batch = len(test_loader)
    with torch.no_grad():
        for _, (inp, target) in enumerate(test_loader):
            input_var = inp.cuda()
            varOutput = model(input_var)
            pred = torch.cat((pred, varOutput.data), 0)
            counter += 1
            time_end = time.time()
            print('Progress:{}%, Remaining time:{}min'.format(round(counter/num_batch*100,2), round((time_end-time_start)/60/counter*(num_batch-counter),2)), end="\r")
    # save to dataframe
    torch.cuda.empty_cache()       
    pred = pred.cpu().detach().numpy()
    return pred[:,0]




if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('input_csv', type = str)
    parser.add_argument('ckpt_folder', type = str)
    parser.add_argument('output_csv', type = str)
    parser.add_argument('name', type = str)

    args = parser.parse_args()
    input_csv = args.input_csv
    ckpt_path = args.ckpt_folder
    output_csv = args.output_csv
    name = args.name
    data_folder = 'E:/data/Total_png/'
    df = pd.read_csv(input_csv)
    print("Test data size: ",len(df))
   
    model_list = glob.glob("{}/*.pth.tar".format(ckpt_path), recursive=False)
    num_models = len(model_list)
    print('{} models'.format(num_models))
    pred_np_total = np.zeros((len(df),num_models))
    start = time.time()
    for i in range(num_models):
        ckpt_name = model_list[i]
        print(ckpt_name)
        score = get_opacity_score(df, data_folder, ckpt_name, 1024, name)
        pred_np_total[:,i] = score
    
    pred_np_ensemble = np.mean(pred_np_total, axis=1)
    df[name] = pred_np_ensemble
    df.to_csv(output_csv,index=False)
    end = time.time()
    print('Elapsed time for opacity prediction: {:.1f} seconds.'.format(end-start))
