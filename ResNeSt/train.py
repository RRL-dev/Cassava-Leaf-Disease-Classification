import os
import gc
import sys
import math
import time
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt


import torch
from torch import nn, optim
from torchvision import models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnest_path = '/home/roni/Dropbox/Code/Kaggle/Cassava-Leaf-Disease-Classification/Check_Point/'
sys.path.append(effnet_path)
from dataset import *
from resnets import resnest50


### Training helper functions ###
def accuracy(preds, target):
    preds = preds.argmax(dim=1)
    return (preds == target).float().mean()
    
def one_epoch(model, dl, loss_func, opt=None, lr_schedule=None):
    running_loss = 0.
    running_acc = 0
    
    for xb, yb in tqdm(dl):
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        loss = loss_func(preds, yb)
        
        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()
            if lr_schedule is not None:
                lr_schedule.step()
    
        running_acc += accuracy(preds, yb).item()
        running_loss += loss.item()
        
    return running_loss / len(dl), running_acc / len(dl)


def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']
    
def train_val(model, params):
   
    num_epochs = params["num_epochs"]
    loss_func = params["loss_func"]
    opt = params["optimizer"]
    train_dl = params["train_dl"]
    val_dl = params["val_dl"]
    lr_scheduler = params["lr_scheduler"]
    path2weights = params["path2weights"]
    one_cycle = params["one_cycle"]
    
    loss_history = {
        "train": [],
        "val": [],
    }
   
    metric_history = {
        "train": [],
        "val": [],
    }
    
    best_model_wts = copy.deepcopy(model.state_dict())
    
    val_min=0.0
    
    for epoch in range(num_epochs):
        start = time.time()
        current_lr = get_lr(opt)
        print(f'Epoch {epoch + 1}/{num_epochs}, current lr = {current_lr:5f}')
      
        model.train()
        train_loss, train_metric = one_epoch(model, train_dl, loss_func, opt, lr_scheduler if one_cycle else None)

        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)
  
        model.eval()
        with torch.no_grad():
            val_loss, val_metric = one_epoch(model, val_dl, loss_func, opt=None)
        
       
        if val_min < val_metric:
            val_min = val_metric
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), path2weights)
            print("Copied best model weights!")
    
        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)
        
        if not one_cycle:
            lr_scheduler.step(val_loss)
            if current_lr != get_lr(opt):
                print("Loading best model weights!")
                model.load_state_dict(best_model_wts) 
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n"
              f"Train Acc: {train_metric:.4f}, Val Acc: {val_metric:.4f}\n"
              f"Completed in {time.time() - start:.3f}")
        
        print("-"*10) 

    model.load_state_dict(best_model_wts)
        
    return model, loss_history, metric_history



class Model(nn.Module):3
    def __init__(self, original_model, num_classes):
        super(Model, self).__init__()
        # Everything except the last linear layer
        self.features    = nn.Sequential(*list(original_model.children())[:-1])
        self.classifier  = nn.Sequential(nn.Linear(2048, num_classes))

    def forward(self, x):
        f = self.features(x)        
        y = self.classifier(f)
        return y



model     = Model(original_model=resnest50(), num_classes=5).to(device) 
criterion = nn.CrossEntropyLoss()
opt       = optim.Adam(model.parameters())
epochs    = 35
lr_sch    = optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, epochs=epochs,
                                       steps_per_epoch=len(train_dl), pct_start=0.25,)


params_train = {
 "num_epochs":    epochs,
 "optimizer":     opt,
 "loss_func":     criterion,
 "train_dl":      train_dl,
 "val_dl":        valid_dl,
 "lr_scheduler":  lr_sch,
 "path2weights":  "effnet.pt",
 "one_cycle":     True
}

model, loss_hist, metric_hist = train_val(model, params_train)