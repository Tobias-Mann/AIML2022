import torch
import datetime
import numpy as np
import os


#optimizer = optim.Adam(model.parameters(),lr=0.005,betas=(0.9,0.999),eps=1e-08,weight_decay=0)

def train(model, train_loader, test_loader, optimizer, criterion, epochs, mini_batch_size = 256, early_stopping = 25, device="cuda", models_dir = "./models/", name = "model", preprocessing = False):
    """
    Train model.
    """
    
    os.makedirs(models_dir, exist_ok=True)
    m_hash = hash(str(model))
    name = name+"_"+str(m_hash)+".pth"
    save_path = os.path.join(models_dir, name)
    train_epoch_losses = []
    validation_epoch_losses = []
    count = 0
    last_validation= None
    model.to(device)
    for epoch in range(epochs):
        print("\n")
        if count >=early_stopping:
            break
        
        train_mini_batch_losses = []
        validation_mini_batch_losses = []
        train_loss = 0.0
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data= data.to(device)
            target  = target.float().to(device) if preprocessing else target.to(device)
            optimizer.zero_grad()
            output = model(data.float())
            loss = criterion(output, target.unsqueeze(1)) if preprocessing else criterion(output, target)
            if (not hasattr(model, "trainable")) or  model.trainable:
                loss.backward()
            optimizer.step()
            train_loss += loss.item()
            model.zero_grad()
            train_mini_batch_losses.append(loss.data.item())
        train_epoch_loss = np.mean(train_mini_batch_losses)
        #print("Train Loss: {}".format(train_loss))
        test_loss = 0.0
        model.eval()
        
        for batch_idx, (data, target) in enumerate(test_loader):
            data= data.to(device)
            target  = target.float().to(device) if preprocessing else target.to(device)
            output = model(data)
            loss = criterion(output, target.unsqueeze(1)) if preprocessing else criterion(output, target)
            test_loss += loss.item()
            validation_mini_batch_losses.append(loss.data.item())
            validation_epoch_loss = np.mean(validation_mini_batch_losses)
        #print("Test Loss: {}".format(test_loss))
        if last_validation is None or validation_epoch_loss < last_validation:
            last_validation = validation_epoch_loss
            torch.save(model.state_dict(), save_path)
            print("Model improved -> saved to {}".format(save_path))
            count=0
        else:
            count+=1
        now = datetime.datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
        print('[LOG {}] epoch: {} train-loss: {} validation-loss {}'.format(str(now), str(epoch), str(train_epoch_loss), str(validation_epoch_loss)), end=" ")

        # determine mean min-batch loss of epoch
        train_epoch_losses.append(train_epoch_loss)
        validation_epoch_losses.append(validation_epoch_loss)
    model.load_state_dict(torch.load(save_path))
    return model, m_hash