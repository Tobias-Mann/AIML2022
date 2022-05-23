from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import helpers

def plot_matrix(preprocessors, dataset):
    y_true = []
    y_pred = []
    for inputs, labels in tqdm(torch.utils.data.DataLoader(dataset, shuffle=False, batch_size = 1, prefetch_factor=2)):
        single_predictions = {label:model(inputs).cpu().numpy() for  label, model in preprocessors.items()}
        final_prediction_l = max(single_predictions, key=single_predictions.get)
        pred_idx = helpers.classes.index(final_prediction_l)
        true_idx = labels[0].cpu().numpy()
        y_true.append(true_idx)
        y_pred.append(pred_idx)
        
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt="d", ax=ax, yticklabels=helpers.classes, xticklabels=helpers.classes)
    plt.savefig(R"images/confusion_matrix_train_data.png")
    plt.close(fig)
    
    y_true = []
    y_pred = []
    for inputs, labels in tqdm(torch.utils.data.DataLoader(dataset.test_set, shuffle=False, batch_size = 1, prefetch_factor=2)):
        single_predictions = {label:model(inputs).cpu().numpy() for  label, model in preprocessors.items()}
        final_prediction_l = max(single_predictions, key=single_predictions.get)
        pred_idx = helpers.classes.index(final_prediction_l)
        true_idx = labels[0].cpu().numpy()
        y_true.append(true_idx)
        y_pred.append(pred_idx)
    
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt="d", ax=ax, yticklabels=helpers.classes, xticklabels=helpers.classes)
    plt.savefig(R"images/confusion_matrix_train_data_test_set.png")
    plt.close(fig)
    
    plt.savefig(R"images/confusion_matrix.png")
    plt.close(fig)