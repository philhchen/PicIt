import copy
import os
import time
import torch
import torch.optim as optim

MODEL_SAVE_DIR = os.path.join(
    os.path.dirname(__file__), '../models/saved_models/'
)

def get_optimizer(model, feature_extract):
    """
    Gather the parameters to be optimized/updated in this run. If we are
    finetuning we will be updating all parameters. However, if we are
    doing feature extract method, we will only update the parameters
    that we have just initialized, i.e. the parameters with requires_grad
    is True.
    """
    params_to_update = model.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # Observe that all parameters are being optimized
    return optim.SGD(params_to_update, lr=0.001, momentum=0.9)

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25,
                print_every=1000, save_dir=MODEL_SAVE_DIR):
    since = time.time()
    val_acc_history = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            iter = 0
            for iter, data in enumerate(dataloaders[phase]):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if iter % print_every == 0:
                    print('Iter: ', iter)
                    print('Running loss: {:4f}'.format(
                        running_loss / ((iter + 1) * inputs.size(0))))
                    print('Running accuracy: {:4f}'.format(
                        running_corrects.double() / ((iter + 1) * inputs.size(0))))
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                    }, os.path.join(save_dir,
                                    'checkpoint-epoch-{}-iter-{}'.format(epoch, iter)))

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load and save best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history
