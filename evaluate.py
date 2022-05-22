from tqdm import tqdm
import torch
import helpers
import os
import pandas as pd
import numpy as np
import normalizers
import json as js

def image_stream(test_data=R"data/test"):
    """
    Generator that yields image and label.
    """
    d = js.load(open(os.path.join(test_data, "layer_dist.json")))
    norm = normalizers.DistNormalizer(d)
    for i in range(4232):
        img = np.load(os.path.join("data","test","test_data","test_"+ str(i)+".npy"))
        yield (i, np.transpose(norm(img)))
        

def evaluate(model, device="cuda"):
    #model = os.path.join("models", "model_"+str(hash(model))+".pth")
    model.eval()
    fname = "result.csv"        
    with open(fname, "w+") as file:
        file.write("test_id,label\n")
        for (n, img) in tqdm(image_stream(), total=4232):
            
            label_idx = torch.argmax(model(torch.tensor([img]).to(device).float()), dim=1).item()
            label = helpers.label_classes_back_mapping[label_idx]
            file.write(str(n)+","+label+"\n")

def multi_evaluate(models, device="cuda"):
    
    fname = "multi_model_result.csv"        
    with open(fname, "w+") as file:
        file.write("test_id,label\n")
        img_predictions = {label:[] for label in models}
        for (n, img) in tqdm(image_stream(), total=4232):
            for label, model in models.items():
                img_predictions[label].append(model(torch.tensor([img]).to(device).float()).cpu().numpy())
        
        adjusted_predictions = {}
        for label, predictions in models.items():
            predictions = np.array(img_predictions[label])
            adjusted_predictions[label]  = (predictions / predictions.std()) # - predictions.mean() 
        
        for (n, img) in tqdm(image_stream(), total=4232):
            image_predictions = {label: adjusted_predictions[label][n] for label in models}
            final_label = max(image_predictions, key=image_predictions.get)
            file.write(str(n)+","+final_label+"\n")