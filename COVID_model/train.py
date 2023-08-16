import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from CV19DataSet import CV19DataSet
from utils import mkdir, removekey, compute_AUCs
import pandas as pd
from train_val_split import df_sample_patient, df_train_val_split_ratio, df_pos_neg_balance
import argparse
from torchvision.models import efficientnet_v2_m


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DATA_DIR = '/mnt/SSD/covid_png/'
CKPT_PATH = './NIH_weights/model_efficientnet_pretrain.pth.tar.notused' # the model pretrained on NIH dataset         
N_CLASSES = 2
CLASS_NAMES = ['COVID', 'Non-COVID']
IMG_SIZE = 480
#########################################################################
def train(df_train_val, model_index, BATCH_SIZE = 40, num_epochs = 20, lrate = 1e-4, folder_prefix = 'save_folder'):
    
       
    save_folder = './Data_imagenet_no_opacity_filter_480/' + folder_prefix + '/'
    if not os.path.isfile(save_folder):
        mkdir(save_folder)
    
   
    df_train, df_val = df_train_val_split_ratio(df_train_val,val_percent = 0.2, seed=model_index)
    df_train.to_csv(save_folder + 'Train_{}.csv'.format(model_index),index=False) 
    df_val.to_csv(save_folder + 'Val_{}.csv'.format(model_index),index=False)
    
    df_train = df_pos_neg_balance(df_train,random_state=model_index)
    df_val = df_pos_neg_balance(df_val,random_state=model_index)
    df_train = df_train.reset_index()
    df_val = df_val.reset_index()

    
    model_name = save_folder + 'EfficientNet_train_' + str(model_index) + '.pth.tar'
    savefig_name = save_folder + 'Loss_EfficientNet_train_' + str(model_index) + '.png'
    train_loss_name = save_folder + 'TrainLoss_EfficientNet_train_'+ str(model_index) + '.npy'
    val_loss_name = save_folder + 'ValLoss_EfficientNet_train_' + str(model_index) + '.npy'
    val_auroc_name = save_folder + 'ValAUROC_EfficientNet_train_' + str(model_index) + '.npy'
    cudnn.benchmark = True

    # initialize and load the model
    model = efficientnet_v2_m(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Sequential(nn.Linear(1280, N_CLASSES), nn.Softmax(dim=1))
    model = nn.DataParallel(model.cuda())
    # summary(model,input_size=(BATCH_SIZE, 3, 480, 480))
    # print(model)
    dev_count = torch.cuda.device_count()
    if os.path.isfile(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        print("=> loading NIH pretrained model")
        state_dict = removekey(checkpoint['state_dict'],['module.classifier.1.0.weight', 'module.classifier.1.0.bias'])
        model_dict = model.state_dict()    
        new_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(new_state_dict)
        model.load_state_dict(model_dict)
    else:
        print("=> no checkpoint found")

   # Build the traininig and validation Dataloader
    normalizer = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    transformList = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=30, fill=0),
        transforms.ColorJitter(brightness=(0.8,1.2), contrast=(0.8,1.2)),
        transforms.GaussianBlur(3, sigma=(0.1, 1.0)),
        transforms.RandomAdjustSharpness(1.1, 0.5),
        transforms.ToTensor(),
        transforms.Normalize(normalizer[0], normalizer[1])])
    
    transformList_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalizer[0], normalizer[1])])
 
    train_dataset = CV19DataSet(df=df_train, base_folder=DATA_DIR, transform=transformList, img_size=IMG_SIZE)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE*dev_count,
                             shuffle=True, num_workers=4, pin_memory=True, drop_last=False, persistent_workers=True)
    print("Training data size: ",len(train_loader.dataset))
    
    val_dataset = CV19DataSet(df=df_val, base_folder=DATA_DIR, transform=transformList_test, img_size=IMG_SIZE)
    
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE*dev_count,
                             shuffle=True, num_workers=4, pin_memory=True, drop_last=False, persistent_workers=True)
    print("Validation data size: ",len(val_loader.dataset))

    
    # Define the optimizer
    optimizer = optim.AdamW(model.parameters(), lrate)
    scheduler = ReduceLROnPlateau(optimizer, factor = 0.5, patience = 3, mode = 'min', min_lr = 1e-6)   
    # Define the LOSS 
    loss = torch.nn.BCELoss(reduction='mean')
    losstrain_list = [] 
    lossVal_list = [] 
    AUROCVal_list = []
    val_loss_min = 1000    
    save_epoch = 0
    nonsave_epoch = 0
    
    for epochID in range(num_epochs):     
        loss_train = epochTrain(model, train_loader, optimizer, loss)
        lossVal, losstensor, auroc_score = epochVal(model, val_loader, loss)
        lossVal = np.around(lossVal, decimals=6)
        auroc_score = np.around(auroc_score, decimals=3)

        losstrain_list.append(loss_train)
        lossVal_list.append(lossVal)
        AUROCVal_list.append(auroc_score)
        scheduler.step(losstensor.item())
        
        
        if (lossVal < val_loss_min):
            val_loss_min = lossVal
            torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossVal}, model_name)
            print ('Epoch [' + str(epochID + 1) + '] [save] Train loss = ' + str(losstrain_list[-1]))
            print ('Epoch [' + str(epochID + 1) + '] [save] Val loss = ' + str(lossVal))  
            print ('Epoch [' + str(epochID + 1) + '] [save] Val AUC = ' + str(auroc_score)) 
            save_epoch = epochID + 1
            
        else:
            print ('Epoch [' + str(epochID + 1) + '] [----] Train loss = ' + str(losstrain_list[-1]))
            print ('Epoch [' + str(epochID + 1) + '] [----] Val loss = ' + str(lossVal))
            print ('Epoch [' + str(epochID + 1) + '] [----] Val AUC = ' + str(auroc_score)) 
            nonsave_epoch = epochID + 1 

        if nonsave_epoch - save_epoch > 4:
            # For early stopping 
            fig = plt.subplots(1, 1, figsize=(5, 5))
            plt.plot(losstrain_list, label='Training Loss')
            plt.plot(lossVal_list, label='Validation Loss')
            plt.plot(AUROCVal_list, label='Validation AUROC')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend()
            plt.savefig(savefig_name, dpi=300, bbox_inches='tight')
            plt.close()
            np.save(train_loss_name, losstrain_list)
            np.save(val_loss_name, lossVal_list)
            np.save(val_auroc_name, AUROCVal_list)
            break
            
        torch.cuda.empty_cache()
        print('----------------------------------------------------------------------')

        if epochID % 5 == 4:
            fig = plt.subplots(1, 1, figsize=(5, 5))
            plt.plot(losstrain_list, label='Training Loss')
            plt.plot(lossVal_list, label='Validation Loss')
            plt.plot(AUROCVal_list, label='Validation AUROC')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend()
            plt.savefig(savefig_name, dpi=300, bbox_inches='tight')
            plt.close()
            np.save(train_loss_name, losstrain_list)
            np.save(val_loss_name, lossVal_list)
            np.save(val_auroc_name, AUROCVal_list)

#-------------------------------------------------------------------------------- 
def epochTrain (model, dataLoader, optimizer, loss):
    model.train()
    losstrain = 0
    losstrainNorm = 0  
    for batchID, (inp, target) in enumerate (dataLoader):
        input_var = Variable(inp).cuda()
        vartarget = Variable(target).cuda()     
        output = model(input_var)
        bce_loss = loss(output, vartarget)
       
        lossvalue = bce_loss   
        optimizer.zero_grad()
        lossvalue.backward()
        optimizer.step()
        
        torch.cuda.empty_cache()
        losstrain += lossvalue.data
        losstrainNorm += 1 
        
    outLoss = losstrain / losstrainNorm
    return outLoss.cpu().detach().numpy()
            
    #-------------------------------------------------------------------------------- 
def epochVal (model, dataLoader, loss):
    model.eval()
    lossVal = 0
    lossValNorm = 0    
    losstensorMean = 0
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
          
            losstensor = loss(varOutput, vartarget) 
            losstensorMean += losstensor
            lossVal += losstensor.data
            lossValNorm += 1
        del input_var, vartarget, varOutput, target
        torch.cuda.empty_cache()
            
    outLoss = lossVal / lossValNorm
    losstensorMean = losstensorMean / lossValNorm
    gt_np = gt.cpu().detach().numpy()
    pred_np = pred.cpu().detach().numpy()
    del gt, pred
    torch.cuda.empty_cache()
    gt_np = gt_np[1: gt_np.shape[0],:]
    pred_np = pred_np[1: pred_np.shape[0],:]
    AUROCs = compute_AUCs(gt_np, pred_np)
        
    return outLoss.data.cpu().detach().numpy(), losstensorMean, AUROCs[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('site', type = str)
    parser.add_argument('pos_patient', type = int)
    parser.add_argument('neg_patient', type = int)
    parser.add_argument('seed', type = int)
    
    args = parser.parse_args()
    site = args.site
    pos_patient = args.pos_patient
    neg_patient = args.neg_patient
    seed = args.seed
    
    
    debug = False
    delta_min = -7
    delta_max = 7
    save_folder = '{}_{}_{}_seed{}'.format(site,pos_patient,neg_patient,seed)
    mkdir('./Data_imagenet_480/' + save_folder )
    init_index = 0
    num_models = 5 
    csv_path = '../CSV_files/{}_opacity.csv'.format(site)
    df = pd.read_csv(csv_path)
    df = df[(df.delta<=delta_max) & (df.delta>=delta_min) & (df.Age>=18)]
    df, _ = df_sample_patient(df,pos_patient,neg_patient,seed=seed)
    df.to_csv('./Data_imagenet_480/' + save_folder + '/' + 'Total.csv',index=False) 
    

    for model_index in range(num_models):
        model_index = init_index + model_index
        train(df, model_index, BATCH_SIZE = 16, num_epochs = 40, lrate = 5e-5, folder_prefix = save_folder)
        
    


