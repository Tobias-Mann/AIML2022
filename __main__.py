import imgdataset
import conf_mat
import train
import helpers
import evaluate
import prepare
import torch
from torch import optim, nn
import ensamble
import os


def main(prep = False, mini_batch_size=32, train_preprocessors = False):
    """
    Main function of the programm, this will load the data, preprocess it, load a model archtecture and eventually train the model.
    Finally it will evaluate the model on the test data.

    Args:
        mini_batch_size (int, optional): Batch Size for training the model, this will be passed on to the other functions such as 'train'. Defaults to 32. 
        train_preprocessors (bool, optional): Boolean indicating wheather to create a new instance of the model or load state dicts from the local 'models' directory. Defaults to False.
    """
    if prep:
        prepare.convert_images()
        prepare.create_layer_dist_files()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset = imgdataset.CustomImageDataset(train_test_split=0.7) # load the train dataset 
    torch.cuda.empty_cache()
    #prep_data = train_dataset.cut_out(split=.5, delete_test_set=False, with_train=True)
    prep_data = train_dataset
    
    
    preprocessors = {}
    for label in helpers.classes:
        if train_preprocessors:
            """
            
            """
            # Create Specialized Dataset 50% of train_dataset
            train_data = prep_data.get_label_dataset(label)
            test_data = prep_data.test_set.get_label_dataset(label)
            # Load instance of binary classifier
            m = ensamble.Preprocessor42(12, classes= 1).to(device)
            criterion = nn.BCELoss().to(device) # earlier : criterion = nn.BCEWithLogitsLoss().to(device) # but the logit is now replaced with sigmoid in the model
            optimizer = optim.Adam(m.parameters(),lr=0.005,betas=(0.9,0.999),eps=1e-08,weight_decay=0)
            train_dataloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size = mini_batch_size, prefetch_factor=2)
            validation_dataloader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size = mini_batch_size, prefetch_factor=2)
            m, m_hash = train.train(model=m, train_loader=train_dataloader, test_loader=validation_dataloader, optimizer=optimizer, criterion=criterion, device=device, epochs=50, name="Preprocessor_"+label, preprocessing=True)
            m.eval()
            preprocessors[label] = m
        else:
            """
            Using the pretrained models from the local 'models' directory.
            """
            m = ensamble.Preprocessor42(12, classes= 1).to(device)
            state_dict = [file for file in os.listdir("models") if file.startswith("Preprocessor_"+label)][0]
            m.load_state_dict(torch.load("models/"+state_dict))
            preprocessors[label] = m.to(device)
    
    # Exclude the preprocessors from the training process (otherwise our GPU memory cannot deal with the size of the model)    
    """
    The commented lines below where used to combine the preprocessors into one model with adjacency layers. As we did not increase performance, we decided to keep the individual models.
    However, in case you are curious about our approach we decided to leave them in the code.
    """
    for m in preprocessors.values():
        for param in m.parameters():
            param.requires_grad = False

    
    # ensamble_model = ensamble.NewEnsamble42(12,10, preprocessors).to(device)
    # optimizer = optim.Adam(m.parameters(),lr=0.005,betas=(0.9,0.999),eps=1e-08,weight_decay=0)
    # criterion = nn.CrossEntropyLoss().to(device)
    
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size = mini_batch_size, prefetch_factor=2)
    # validation_dataloader = torch.utils.data.DataLoader(train_dataset.test_set, shuffle=True, batch_size = mini_batch_size, prefetch_factor=2)
    # m, m_hash = train.train(model=ensamble_model, train_loader=train_dataloader, test_loader=validation_dataloader, optimizer=optimizer, criterion=criterion, device=device, name="Ensamble", epochs=1)
    for name, model in preprocessors.items():
        model.eval()
    
    #evaluate.evaluate(m)
    conf_mat.plot_matrix(preprocessors, train_dataset)
    evaluate.multi_evaluate(preprocessors)
if __name__ == "__main__":
    main(False, 32, False)