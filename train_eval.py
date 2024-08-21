import torch.nn.functional as F
import torch

def train(model, optimizer, data, label, args, train_mask):
    
    model.train()
    optimizer.zero_grad()
    logits = model(data, args.dropout)
    train_logits = logits[train_mask]
    train_label = label
    train_loss = F.cross_entropy(train_logits, train_label,reduction='none')
    train_loss_mean = train_loss.mean()
    train_loss_mean.backward()
    optimizer.step()
    pred_train = logits[train_mask].argmax(1)
    train_acc = pred_train.eq(label).sum().item() / train_mask.shape[0]   
    return train_acc, train_loss_mean


def evaluate(model, data, label, args):

    with torch.no_grad():
        model.eval()
        logits = model(data, args.dropout)
        pred_val = torch.softmax(logits[data.val_mask],1).argmax(1)
        val_acc = pred_val.eq(label[data.val_mask]).sum().item() / data.val_mask.sum().item()
        pred_test = torch.softmax(logits[data.test_mask],1).argmax(1)
        test_acc = pred_test.eq(label[data.test_mask]).sum().item() / data.test_mask.sum().item()
    return  val_acc, test_acc

def predict(model, data, args):

    with torch.no_grad():
        model.eval()
        logits = model(data, args.dropout)
        preds = torch.softmax(logits,1).argmax(1)
    return preds 


   