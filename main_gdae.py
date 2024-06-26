import argparse

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerDecoder

from torch_geometric.utils import remove_self_loops, dense_to_sparse

from torch_sparse import SparseTensor, spspmm
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.linear_model import LogisticRegression

from datasets import *
from utils import *
from models.GDAE import *
from generate_te import *

def main(args):
    
    setup_seed(0)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    # load dataset
    data, articulation_points, bridges, components, y_ap = load_data(name=args.dataset, only_data=False)
    pe_path = f'./data/topology_encodings/{args.dataset}_te.pt'
    if os.path.exists(pe_path):
        pe = torch.load(pe_path).to(device)
    else:
        pe = generate_topology_encoding(data, 32)
        torch.save(pe, pe_path)
    data.pe = pe

    data = data.to(device)

    # partial edges for link prediction
    if 'lp' in args.task:
        train_data, val_data, test_data = T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True, split_labels=True, add_negative_train_samples=True)(data)

    all_res_nc, all_res_cvd, all_res_ced, all_res_lp, all_res_dp = [], [], [], [], []
    for run in range(args.runs):
        seed = run + 42
        
        print(f"【RUN:{run} | seed = {seed}】")
        
        setup_seed(seed)
        
        # masking A
        if args.mask_type == 'None':
            mask = None
        else:
            mask = Masker(p=args.p, num_nodes=data.num_nodes, walk_length=3)
            
        # Dual-view Encoder
        mlp_encoder = MLPEncoder(data.num_features 
                                , data.pe.shape[1]
                                , args.encoder_channels
                                , args.hidden_channels
                                , dropout=args.encoder_dropout
                                , norm=args.norm
                                , activation=args.encoder_activation)
        mpgnn_encoder = MPGNNEncoder(data.num_features
                            , args.encoder_channels
                            , args.hidden_channels
                            , num_layers=args.encoder_layers
                            , dropout=args.encoder_dropout
                            , norm=args.norm
                            , layer=args.layer
                            , activation=args.encoder_activation)
        
        # Dual-view Decoder
        edge_decoder = EdgeDecoder(args.hidden_channels, args.decoder_channels, num_layers=args.decoder_layers, dropout=args.decoder_dropout)
        node_decoder = NodeDecoder(args.hidden_channels, args.hidden_channels*2, data.x.shape[1] + data.pe.shape[1])
        degree_decoder = DegreeDecoder(args.hidden_channels*2, args.hidden_channels*4)
        
        # GDAE framework
        model = GDAEModel(mpgnn_encoder, edge_decoder, node_decoder, mlp_encoder, degree_decoder, mask).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        

        best_nc = pd.DataFrame(np.zeros([2, 2]))
        best_cvd = pd.DataFrame(np.zeros([2, 1]))
        best_ced = pd.DataFrame(np.zeros([2, 1]))
        best_lp = pd.DataFrame(np.zeros([2, 2]))
        best_dp = pd.DataFrame(np.vstack([np.ones([1, 1])*-1e9, np.ones([1, 1])*1e9]))
        pbar = tqdm(total=args.epochs)
        # start_time = time.time()
        for epoch in range(args.epochs):
            # training            
            model.train()
    
            # link prediction
            if 'lp' in args.task:
                loss = model.train_epoch(train_data, optimizer, batch_size=args.batch_size, lam1=args.lam1, lam2=args.lam2)
            else:
                loss = model.train_epoch(data, optimizer, batch_size=args.batch_size, lam1=args.lam1, lam2=args.lam2)
  
            pbar.set_description(f'Epoch {epoch:02d}')
            if 'lp' in args.task:
                pbar.set_postfix({"loss": f'{loss:.4f}', "best_val": f'{best_lp.iloc[0, 1]:.3f}', "best_test": f'{best_lp.iloc[0, 0]:.3f}'})
            elif 'dp' in args.task:
                pbar.set_postfix({"loss": f'{loss:.4f}', "best_mae": f'{best_dp.iloc[1, 0]:.4f}'})
            elif 'cvd' in args.task:
                pbar.set_postfix({"loss": f'{loss:.4f}', "best_acc": f'{best_cvd.iloc[0, 0]:.3f}'})
            elif 'nc' in args.task:
                pbar.set_postfix({"loss": f'{loss:.4f}', "best_val": f'{best_nc.iloc[0, 1]:.3f}', "best_test": f'{best_nc.iloc[0, 0]:.3f}'})
            pbar.update(1)
        
            if epoch and epoch % args.eval_steps == 0:
                # end_time = time.time()
                
                # epoch_time = (end_time - start_time) / args.eval_steps
                # print(f"Epoch Time: {epoch_time}")
                
                # eval downstream task
                model.eval()

                if 'lp' not in args.task:
                    embedding1 = model.mpgnn_encoder.get_node_embedding(data.x, data.edge_index)
                    embedding2, p = model.mlp_encoder.get_node_embedding(data.x, data.pe)
                    node_embedding = torch.cat([embedding1, embedding2], dim=-1)
                   
                if 'nc' in args.task:
                    cur = 0 if (data.train_mask.shape[1]==1) else (run % data.train_mask.shape[1])
                    res_nc = evaluate_nc(node_embedding, data.y, data.train_mask[:, cur], data.val_mask[:, cur], data.test_mask[:, cur], False)
                    if res_nc.iloc[0][1] > best_nc.iloc[0][1]:
                        best_nc = res_nc
                        
                if 'cvd' in args.task:
                    res_cvd = cut_vertex_detection(node_embedding, y_ap, is_print=False)
                    if best_cvd.iloc[0, 0] < res_cvd.iloc[0, 0]:
                        best_cvd = res_cvd
                        
                if 'lp' in args.task:
                    z0 = model.mpgnn_encoder(train_data.x, train_data.edge_index)
                    z1, p = model.mlp_encoder(train_data.x, train_data.pe)
                    auc_val, ap_val = model.eval_lp(z0, p, val_data.pos_edge_label_index, val_data.neg_edge_label_index)
                    auc_test, ap_test = model.eval_lp(z0, p, test_data.pos_edge_label_index, test_data.neg_edge_label_index)
                    res_lp = get_lp_res(auc_test, ap_test, auc_val, ap_val)
                    
                    if res_lp.iloc[0][1] > best_lp.iloc[0][1]:  # AUC
                        best_lp = res_lp
                        
                if 'dp' in args.task:
                    res_dp = evaluate_dp(node_embedding, data.degrees)
                    if best_dp.iloc[0, 0] < res_dp.iloc[0, 0] or best_dp.iloc[1, 0] > res_dp.iloc[1, 0]:
                        best_dp = res_dp
                    
        # current run              
        pbar.close()
        if 'nc' in args.task:
            all_res_nc.append(best_nc)
        if 'cvd' in args.task:
            all_res_cvd.append(best_cvd)
        if 'lp' in args.task:
            all_res_lp.append(best_lp)
        if 'dp' in args.task:
            all_res_dp.append(best_dp)
    
    # all runs results
    print(f"{args.task} on {args.dataset} by GDAE | all runs ({args.runs})")
    if 'nc' in args.task:
        count_avg_nc(all_res_nc)
    if 'cvd' in args.task:
        count_avg_cvd(all_res_cvd)
    if 'lp' in args.task:
        count_avg_lp(all_res_lp)
    if 'dp' in args.task:
        count_avg_dp(all_res_dp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ESSENTIAL
    parser.add_argument('-dataset', type=str, default='cora')
    parser.add_argument('-gpu_id', type=int, default=0)
    
    parser.add_argument('-task', nargs='+', type=str, default=['nc'])
    
    # Model Hyper-param
    parser.add_argument("-mask_type", nargs="?", default="Edges", help="Whether masking A, `Edges` or `None` (default: Edges)")

    parser.add_argument("-layer", nargs="?", default="gcn", help="MPGNN backbone, (default: gcn)")
    parser.add_argument("-encoder_activation", nargs="?", default="elu", help="Activation function for Dual-view encoder, (default: relu)")
    parser.add_argument('-encoder_channels', type=int, default=128, help='Channels of Dual-view encoder layers. (default: 128)')
    parser.add_argument('-hidden_channels', type=int, default=256, help='Channels of embedding size. (default: 256)')
    parser.add_argument('-decoder_channels', type=int, default=128, help='Channels of Dual-view decoder layers. (default: 128)')
    parser.add_argument('-encoder_layers', type=int, default=2, help='Number of layers for Dual-view encoder. (default: 2)')
    parser.add_argument('-decoder_layers', type=int, default=2, help='Number of layers for Dual-view decoders. (default: 2)')
    parser.add_argument('-encoder_dropout', type=float, default=0.8, help='Dropout probability of encoder. (default: 0.8)')
    parser.add_argument('-decoder_dropout', type=float, default=0.2, help='Dropout probability of decoder. (default: 0.2)')

    parser.add_argument('-lr', type=float, default=0.01, help='Learning rate for autoencoding. (default: 0.01)')
    parser.add_argument('-weight_decay', type=float, default=5e-5, help='weight_decay for autoencoding. (default: 5e-5)')
    parser.add_argument('-grad_norm', type=float, default=1.0, help='grad_norm for autoencoding. (default: 1.0)')
    parser.add_argument('-batch_size', type=int, default=2**16, help='Number of batch size for autoencoding. (default: 2**16)')

    parser.add_argument('-p', type=float, default=0.7, help='Mask ratio of Topology Masking')
    parser.add_argument('-norm', action='store_true', help='Whether to use batch normalization for Dual-view encoder. (default: False)')
    parser.add_argument('-lam1', type=float, default=0.1, help='Weight factor for attribute reconstruction. (default: 0.1)')
    parser.add_argument('-lam2', type=float, default=0.001, help='Weight factor for overall loss regularization. (default: 0.001)')

    parser.add_argument('-epochs', type=int, default=500, help='Number of training epochs. (default: 500)')
    parser.add_argument('-runs', type=int, default=10, help='Number of runs. (default: 10)')
    parser.add_argument('-eval_steps', type=int, default=25, help='Number of validation intervals. (default: 25)')
    
    args = parser.parse_args()

    print(args)
    main(args)
    