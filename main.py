import argparse

def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    accs_by_val = []
    accs_latest = []
    
    for seed in tqdm(range(args.runs)):
            
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        #load dataset
        dataset, clean_label = get_planetoid_dataset(args.data_dir, args.dataset, args.normalize_features, type=args.noise_type, rate=args.noise_rate, seed=0, split_type=args.split_type)
        data = dataset[0]
        data.clean_y = clean_label
        data = data.to(device)
        data.train_idx = data.train_mask.nonzero().reshape(-1,)
        # pre-calculate the Personalized PageRank Matrix
        Pi = Personalized_PageRank(args,data)
        data.Pi = Pi.to(device)

        # calulate the CBC 
        node_difficulty = difficulty_measurer(data, data.train_idx).to(device)
        # handling outliers
        node_difficulty[node_difficulty==0] = float("inf")
        # sort dataset by CBC
        _, indices = torch.sort(node_difficulty[data.train_idx])
        sorted_trainset = data.train_idx[indices]
        
        #-----------------------Pre-train---------------------#
        pre_model = gcn(dataset.num_features,dataset.num_classes,args.hidden)
        pre_model.to(device).reset_parameters()
        pre_optimizer = optim.Adam(pre_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for _ in range(args.pretain_epoch):
            pre_model.train()
            pre_optimizer.zero_grad()
            logits = pre_model(data, args.dropout)
            loss = F.cross_entropy(logits[data.train_idx], data.y[data.train_idx])
            loss.backward()
            pre_optimizer.step()
            train_acc = logits[data.train_idx].argmax(1).eq(data.y[data.train_idx]).sum().item() / data.train_idx.shape[0]   
            val_acc, test_acc = evaluate(pre_model, data, data.y, args)
            
        # -----------------------TSS---------------------#
        model = gcn(dataset.num_features,dataset.num_classes,args.hidden)
        model.to(device).reset_parameters()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        train_subset = sorted_trainset[:int(args.lam * sorted_trainset.shape[0])]
        preds  = predict(pre_model, data, args)
        clean_idx = train_subset[preds[train_subset] == data.y[train_subset]]
        unlabeled_idx = train_subset[preds[train_subset] != data.y[train_subset]]
        if args.debug == True:
            return_statistics(clean_idx.cpu(),unlabeled_idx.cpu(),train_subset[(data.y[train_subset]  == data.clean_y[train_subset])].cpu(),train_subset[(data.y[train_subset]  != data.clean_y[train_subset])].cpu(),args)
        train_subset = clean_idx

        add_set = torch.LongTensor([]).cuda()
        later_size = args.lam
        val_accs = []
        test_accs = []
        for epoch in range(args.n_epoch):
            train_subset = torch.concat((train_subset,add_set))
            model.train()
            optimizer.zero_grad()
            logits = model(data, args.dropout)
            train_loss = F.cross_entropy(logits[train_subset], data.y[train_subset])   
            train_loss.backward()
            optimizer.step() 
            train_acc = logits[train_subset].argmax(1).eq(data.y[train_subset]).sum().item() / train_subset.shape[0]   
            val_acc, test_acc = evaluate(model, data, data.y, args)
            val_accs.append(val_acc)
            test_accs.append(test_acc)
            if args.debug == True:
                print(f'Train Epoch: {epoch},',f'Train loss: {float(train_loss):.4f},'f'Train acc: {float(train_acc):.4f},'f'Val acc: {float(val_acc):.4f},'f'Test acc: {float(test_acc):.4f}')

            if epoch < args.T:
                size = training_scheduler(args.lam, epoch+1, args.T, scheduler=args.scheduler)
                add_set = sorted_trainset[int(later_size * sorted_trainset.shape[0]):int(size * sorted_trainset.shape[0])]
                preds  = predict(pre_model, data, args)
                clean_idx = add_set[preds[add_set] == data.y[add_set]]
                unlabeled_idx = add_set[preds[add_set] != data.y[add_set]]
                add_set =  clean_idx
                later_size = size
            else:    
                add_set = torch.LongTensor([]).cuda()
        
        test_acc_by_val = test_accs[np.argmax(val_accs)]*100
        test_acc_lastest = test_accs[-1]*100
        accs_by_val.append(test_acc_by_val)
        accs_latest.append(test_acc_lastest)
    
    print('test_accuracy',np.mean(accs_by_val))
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type = str, help='dir to dataset', default = './dataset')
    parser.add_argument('--runs', type=int, default = 10)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--dataset', type = str, default = 'cora',choices = ['cora','citeseer','pubmed'])
    parser.add_argument('--split_type', type=str, default = 'full')
    # train
    parser.add_argument('--pretain_epoch', type=int, default=400)
    parser.add_argument('--n_epoch', type=int, default=500)
    parser.add_argument('--lr', type=int, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--normalize_features', type=bool, default=True)
    # label noise
    parser.add_argument('--noise_rate', type=float, default=0.5)
    parser.add_argument('--noise_type', type=str, default='sym',choices=['pair','sym','idn','random'])
    # curriculum learning
    parser.add_argument('--lam', default=0.25)
    parser.add_argument('--T', type=int, default=400)
    parser.add_argument('--scheduler', default='linear')
    # Pagerank 
    parser.add_argument('--pagerank-prob', default=0.85, type=float,help="probility of going down instead of going back to the starting position in the random walk")
    parser.add_argument('--ppr-topk', default=-1,type=int)
    args = parser.parse_args()

    
    import torch
    from torch import optim 
    from tqdm import tqdm
    import numpy as np
    from datasets import get_planetoid_dataset
    from model import gcn
    from train_eval import evaluate, predict
    from utils import difficulty_measurer,Personalized_PageRank,return_statistics,training_scheduler
    import torch.nn.functional as F
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    run(args)
