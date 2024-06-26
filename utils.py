import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import LinearRegression as LinR
from sklearn.model_selection import cross_validate, cross_val_predict, StratifiedKFold, cross_val_score
from sklearn.preprocessing import normalize, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def normalize_adj(edge_sparse, mode='sym'):
    adj = edge_sparse
    if mode == "sym":
        inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1)) + 1e-15)
        return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
    elif mode == "row":
        inv_degree = 1. / (adj.sum(dim=1) + 1e-15)
        return inv_degree[:, None] * adj

def sparse_to_index(edge_sparse):
    t = edge_sparse.coo()
    edge_index = torch.stack([t[0], t[1]])
    edge_weight = t[2]
    return edge_index, edge_weight

def get_lp_res(auc_test, ap_test, auc_val, ap_val, is_print=False):
    res = pd.DataFrame([[auc_test, ap_test], [auc_val, ap_val]]).T
    res.columns = ['test', 'val']
    res.index = ['AUC', 'AP']
    
    if is_print:
        print(res)

    return res

def count_avg_lp(res_lp_list):
    arr = np.array([df.values for df in res_lp_list])
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)

    k = []
    for i in range(2):
        auc = f'{mean[0][i]:.2f} ± {std[0][i]:.2f}'
        ap = f'{mean[1][i]:.2f} ± {std[1][i]:.2f}'
        k.append([auc, ap])
    tr = pd.DataFrame(k).T
    tr.columns = ['test', 'val']
    tr.index = ['avg-auc', 'avg-ap']

    print(tr)
    
    return tr

def evaluate_dp(node_embedding, labels):

    X_, y_ = node_embedding.detach().cpu().numpy(), labels.detach().cpu().numpy()
    
    clf = LinR()
    scores = cross_validate(clf, X_, y_, cv=5, scoring={'r2': 'r2', 'mae': 'neg_mean_absolute_error'})
    r2 = scores['test_r2'].mean()
    mae = scores['test_mae'].mean()
    
    res = pd.DataFrame(np.zeros([2, 1]), columns=['test-dp'], index=['r2', 'mae'])
    res.iloc[0, 0], res.iloc[1, 0] = r2, -mae
    
    return res

def count_avg_dp(res_list):
    arr = np.array([df.values for df in res_list])

    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)

    r2 = f'{mean[0][0]:.3f} ± {std[0][0]:.3f}'
    mae = f'{mean[1][0]:.3f} ± {std[1][0]:.3f}'

    res = pd.DataFrame([r2, mae], columns=['test-dp'], index=['avg-r2', 'avg-mae'])
    
    print(f'Finnal MAE: {res.iloc[1, 0]}')
    
    return res


def evaluate_nc(embedding, labels, train_mask, val_mask, test_mask, is_print=True):
    
    X = embedding.detach().cpu().numpy()
    Y = labels.detach().cpu().numpy()
    X = normalize(X, norm='l2')
    
    X_train, X_val, X_test = X[train_mask.cpu()], X[val_mask.cpu()], X[test_mask.cpu()]
    y_train, y_val, y_test = Y[train_mask.cpu()], Y[val_mask.cpu()], Y[test_mask.cpu()]

    clf = LR(solver='lbfgs', max_iter=10000, multi_class='auto')
    c = 2.0 ** np.arange(-10, 10)
    clf = GridSearchCV(estimator=OneVsRestClassifier(clf), param_grid=dict(estimator__C=c), n_jobs=-1, cv=5, verbose=0)
  
    clf.fit(X_train, y_train)
    
    y_test_pred = clf.predict(X_test)
    acc_test = accuracy_score(y_test, y_test_pred)*100
    micro_test = f1_score(y_test, y_test_pred, average='micro')*100
    
    y_val_pred = clf.predict(X_val)
    acc_val = accuracy_score(y_val, y_val_pred)*100
    micro_val = f1_score(y_val, y_val_pred, average='micro')*100
    
    res = pd.DataFrame([[acc_test, micro_test], [acc_val, micro_val]]).T
    res.columns = ['test', 'val']
    res.index = ['accuracy', 'f1_micro']
    
    if is_print:
        print(res)

    return res


def count_avg_nc(res_nc_list):
    arr = np.array([df.values for df in res_nc_list])
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)

    k = []
    for i in range(2):
        accuracy = f'{mean[0][i]:.2f} ± {std[0][i]:.2f}'
        f1_micro = f'{mean[1][i]:.2f} ± {std[1][i]:.2f}'
        k.append([accuracy, f1_micro])
    tr = pd.DataFrame(k).T
    tr.columns = ['test', 'val']
    tr.index = ['avg-accuracy', 'avg-f1_micro']

    print(tr)
    
    return tr

def cut_vertex_detection(embedding, labels, cv=5, is_print=True):
    X = embedding.detach().cpu().numpy()
    Y = labels
    
    X = StandardScaler().fit_transform(X)
    
    k = []
    scores = cross_validate(LR(n_jobs=-1, class_weight='balanced', max_iter=10000), X, Y, cv=cv, scoring=['f1_micro', 'f1_macro'], n_jobs=-1, verbose=0)
    micro = f"{scores['test_f1_micro'].mean()*100:.2f} ± {scores['test_f1_micro'].std()*100:.2f}"
    macro = f"{scores['test_f1_macro'].mean()*100:.2f} ± {scores['test_f1_macro'].std()*100:.2f}"
    k.append([micro, macro])
    tr = pd.DataFrame(k).T
    tr.columns = ['test-cvd']
    tr.index = ['f1-micro', 'f1-macro']
    
    if is_print:
        print(tr)
        
    res = pd.DataFrame(np.zeros([2, 1]), columns=['test-cvd'], index=['micro', 'macro'])
    res.iloc[0, 0], res.iloc[1, 0] = scores['test_f1_micro'].mean()*100, scores['test_f1_macro'].mean()*100
    
    return res

def count_avg_cvd(res_list):
    arr = np.array([df.values for df in res_list])

    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)

    micro = f'{mean[0][0]:.2f} ± {std[0][0]:.2f}'
    macro = f'{mean[1][0]:.2f} ± {std[1][0]:.2f}'

    res = pd.DataFrame([micro, macro], columns=['test-cvd'], index=['avg-micro', 'avg-macro'])
    
    print(f'Finnal micro: {res.iloc[0, 0]}')
    
    return res

def cut_edge_detection(x, edge_index, bridges, seed):
    def train_epoch(loader):
        clf.train()
        for inputs, labels in loader:
            optimizer.zero_grad()
            criterion(clf(inputs), labels).backward()
            optimizer.step()

    @torch.no_grad()
    def test(loader):
        clf.eval()
        logits = []
        y_true = []
        for inputs, labels in loader:
            logits.append(clf(inputs))
            y_true.append(labels)
        logits = torch.cat(logits, dim=0).cpu().numpy()
        y_true = torch.cat(y_true, dim=0).cpu().numpy()
        logits = logits.argmax(1)
        
        return f1_score(y_true, logits, average='micro')*100, f1_score(y_true, logits, average='macro')*100

    cut_edges = torch.from_numpy(np.array(bridges).T).to(x.device)
    edge_embedding = torch.cat([x[edge_index[0], :], x[edge_index[1], :]], dim=-1).to(x.device)
    # print(edge_embedding.shape)

    value = cut_edges.t().unsqueeze(0).expand(edge_index.t().shape[0], -1, -1)
    labels = torch.any(torch.all(torch.eq(value, edge_index.t().unsqueeze(1)), dim=2), dim=1).long().to(x.device)

    X, y = edge_embedding, labels

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    micro_list, macro_list = [], []
    X_, y_ = X.detach().cpu().numpy(), y.detach().cpu().numpy()

    rus = RandomUnderSampler(random_state=seed)
    X_, y_ = rus.fit_resample(X_, y_)

    X_ = StandardScaler().fit_transform(X_)
    
    for train_index, test_index in kf.split(X_, y_):
        X_train, X_test = torch.from_numpy(X_[train_index]).to(X.device), torch.from_numpy(X_[test_index]).to(X.device)
        y_train, y_test = torch.from_numpy(y_[train_index]).to(X.device), torch.from_numpy(y_[test_index]).to(X.device)
        
        train_dataset, test_dataset = TensorDataset(X_train, y_train), TensorDataset(X_test, y_test)
        train_loader, test_loader = DataLoader(train_dataset, batch_size=512, shuffle=True), DataLoader(test_dataset, batch_size=20000, shuffle=False)
        
        clf = nn.Linear(X.shape[1], 2).to(X.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(clf.parameters(), lr=0.01)
        
        for epoch in range(10):
            train_epoch(train_loader)
        
        micro, macro = test(test_loader)
        micro_list.append(micro)
        macro_list.append(macro)
    
    res = pd.DataFrame(np.zeros([2, 1]), columns=['test-ced'], index=['micro', 'macro'])
    res.iloc[0, 0], res.iloc[1, 0] = np.array(micro_list).mean(), np.array(macro_list).mean()
    
    return res