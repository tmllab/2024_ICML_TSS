# basic function
import numpy as np
import torch
from numpy.testing import assert_array_almost_equal
from math import inf
from scipy import stats
import torch.nn.functional as F

sym_count = 0
pair_count = 0


def multiclass_noisify(y, P, random_state):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    # print (np.max(y), P.shape[0])
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    # print(m)
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


# noisify_pairflip call the function "multiclass_noisify"
def noisify_pairflip(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes - 1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes - 1, nb_classes - 1], P[nb_classes - 1, 0] = 1. - n, n

        y_train = y_train.numpy()
        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise, end=' ')
        y_train = y_train_noisy

    return y_train, actual_noise


def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes - 1):
            P[i, i] = 1. - n
        P[nb_classes - 1, nb_classes - 1] = 1. - n
        y_train = y_train.numpy()
        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        # assert actual_noise > 0.0
        # print('Actual noise %.2f' % actual_noise, end=' ')
        y_train = y_train_noisy

    # print (P)

    return y_train, actual_noise

def get_instance_noisy_label(n, zipdata, labels, num_classes, feature_size, norm_std, seed): 
    # n -> noise_rate 
    # dataset -> mnist, cifar10 # not train_loader
    # labels -> labels (targets)
    # label_num -> class number
    # feature_size -> the size of input images (e.g. 28*28)
    # norm_std -> default 0.1
    # seed -> random_seed 
    print("building instance dependent noise dataset...")
    label_num = num_classes
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))

    P = []
    flip_distribution = stats.truncnorm((0 - n) / norm_std, (1 - n) / norm_std, loc=n, scale=norm_std)
    flip_rate = flip_distribution.rvs(labels.shape[0])

    if isinstance(labels, list):
        labels = torch.FloatTensor(labels)
    labels = labels.cuda()

    W = np.random.randn(label_num, feature_size, label_num)


    W = torch.FloatTensor(W).cuda()

    for i, (x, y) in enumerate(zipdata):
        # 1*m *  m*10 = 1*10
        x = x.cuda()
        A = x.view(1, -1).mm(W[y]).squeeze(0)
        A[y] = -inf
        A = flip_rate[i] * F.softmax(A, dim=0)
        A[y] += 1 - flip_rate[i]
        P.append(A)
    P = torch.stack(P, 0).cpu().numpy()
    l = [i for i in range(label_num)]
    new_label = [np.random.choice(l, p=P[i]) for i in range(labels.shape[0])]
    record = [[0 for _ in range(label_num)] for i in range(label_num)]

    for a, b in zip(labels, new_label):
        a, b = int(a), int(b)
        record[a][b] += 1


    pidx = np.random.choice(range(P.shape[0]), 1000)
    cnt = 0
    for i in range(1000):
        if labels[pidx[i]] == 0:
            a = P[pidx[i], :]
            cnt += 1
        if cnt >= 10:
            break
    return np.array(new_label)


def add_random_noise(label, noise_percentage, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    noisy_labels = label
    probs_to_change = torch.randint(100, (len(noisy_labels),))
    idx_to_change = probs_to_change >= (100.0 - noise_percentage*100)


    for n, label_i in enumerate(noisy_labels):
        if idx_to_change[n] == 1:
            set_labels = list(
                set(range(len(label.unique()))) - set([label_i]))  # this is a set with the available labels (without the current label)
            set_index = np.random.randint(len(set_labels))
            noisy_labels[n] = set_labels[set_index]

    # loader.sampler.data_source.train_data = images
    # loader.sampler.data_source.train_labels = noisy_labels

    return noisy_labels


def make_noise(dataset, type, rate, seed):
    if rate != 0:
        nb_classes = dataset.num_classes
        train_labels = dataset.data.y[dataset.data.train_mask]
        val_labels = dataset.data.y[dataset.data.val_mask]

        if type == 'sym':
            train_noisy_labels, actual_noise_rate = noisify_multiclass_symmetric(train_labels, rate, random_state=seed,nb_classes=nb_classes)
            val_noisy_labels,_ = noisify_multiclass_symmetric(val_labels, rate, random_state=seed,nb_classes=nb_classes)
            
        elif type == 'pair':
            # rate = 0.45 if rate == 0.5 else rate
            train_noisy_labels, actual_noise_rate = noisify_pairflip(train_labels, rate, random_state=seed,nb_classes=nb_classes)
            val_noisy_labels, _ = noisify_pairflip(val_labels, rate, random_state=seed,nb_classes=nb_classes)
        
        elif type == 'idn':
            zipdata_train = zip(dataset.data.x[dataset.data.train_mask],dataset.data.y[dataset.data.train_mask])
            train_noisy_labels = get_instance_noisy_label(n=rate, zipdata=zipdata_train, labels=train_labels, num_classes=nb_classes, feature_size=dataset.data.num_features, norm_std=0.1, seed=seed)
            zipdata_val = zip(dataset.data.x[dataset.data.val_mask],dataset.data.y[dataset.data.val_mask])
            val_noisy_labels = get_instance_noisy_label(n=rate, zipdata=zipdata_val, labels=val_labels, num_classes=nb_classes, feature_size=dataset.data.num_features, norm_std=0.1, seed=seed)
        
        elif type == 'random':
            train_label = dataset.data.y[dataset.data.train_mask]
            train_noisy_labels = add_random_noise(train_label, rate, seed=seed).numpy()
            val_label = dataset.data.y[dataset.data.val_mask]
            val_noisy_labels = add_random_noise(val_label, rate, seed=seed).numpy()
            
        clean_y = dataset.data.y.clone()
        dataset.data.y[dataset.data.train_mask] = torch.from_numpy(train_noisy_labels)
        dataset.data.y[dataset.data.val_mask] = torch.from_numpy(val_noisy_labels)
    else:   
        clean_y = dataset.data.y

    return clean_y

