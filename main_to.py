from args import create_exp_dirs
from args import init_parser, init_sub_args
import torch
import random
import numpy as np
import csv

from dataset import get_dataset_and_loader
from utils.train_utils_token import dump_args, init_model_params, Trainer, init_optimizer, init_scheduler, CostumLoss

from utils.train_utils_token import load_trained_encoder, load_trained_CTR

from utils.data_utils import trans_list
from tqdm import tqdm
from utils.eval2 import score_dataset
import yaml
import os
from models import TransformerPredictor
from models import ReformerPredictor
from models_graph.gcae.gcae import GCAE, Encoder_T  # Import GCAE model
from torch.utils.tensorboard import SummaryWriter


def main ():
    print('Start training...')
    parser = init_parser()
    args = parser.parse_args()
    if args.seed == 999:  
        args.seed = torch.initial_seed()
        np.random.seed(0)
    else:
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(args.seed)
        np.random.seed(0)
    args, model_args = init_sub_args(args)
    args.ckpt_dir = create_exp_dirs(args.exp_dir, dirmap=args.dataset)
    
    pretrained_model = vars(args).get('model_ckpt_dir', None)
    dataset, loader = get_dataset_and_loader(args, trans_list=trans_list, only_test=(pretrained_model is not None))
    
    expand_ratio = 1
    extra_dim = 0
    if args.relative:
        expand_ratio = 2

    #graph_based configuration:
    # Define Transformer model input dimension based on token shape
    if model_args.token_config == "graph":
        input_dim = args.h_d * 18 *4   #Channels * number_of_keypoints from encoder output 
    

    elif model_args.token_config == "CTR":  
        input_dim = 5120 
    

    Transformer_model = TransformerPredictor(input_dim*expand_ratio+extra_dim, model_args.num_heads, model_args.latent_dim, model_args.num_layers, 1000, device=args.device, dropout=args.dropout)
                                            #(d_model, num_heads, d_feedforward, num_layers, max_len=1000, device='cuda:0', dropout=0.1):
    
    #Transformer_model = ReformerPredictor(d_model=input_dim*expand_ratio+extra_dim, num_heads=model_args.num_heads,num_layers=model_args.num_layers, device='cuda:0') #,d_feedforward=model_args.latent_dim
    
    
      
    #transformer_param = sum(p.numel() for p in Transformer_model.parameters() if p.requires_grad) + sum(p.numel() for p in Transformer_model.decoder.parameters() if p.requires_grad)
    #print("number of transformer param",transformer_param)
    if pretrained_model == None:  
        if not os.path.exists(args.model_save_dir):
            os.makedirs(args.model_save_dir)
        arguments = vars(args)
        with open(args.model_save_dir + '/' + 'arguments.yaml', 'w') as file:
            yaml.dump(arguments, file)
        ae_optimizer_f = init_optimizer(args.model_optimizer, lr=args.model_lr)
        ae_scheduler_f = init_scheduler(args.sched, lr=args.model_lr, epochs=args.epochs)
        trainer = Trainer(model_args, Transformer_model, loader['train'], loader['test'], optimizer_f=ae_optimizer_f,
                                scheduler_f=ae_scheduler_f)
        
        trained_model = trainer.train(checkpoint_filename='trans', args=args)
        
    else:   
        loss_func = CostumLoss(model_args.loss, a=model_args.a, b=model_args.b, c=model_args.c, d=model_args.d)
        checkpoint = torch.load(args.model_ckpt_dir)
        Transformer_model.load_state_dict(checkpoint['state_dict'])
        print('Model loaded successfully!')
        Transformer_model.to(args.device)
        eval_loss = []
        dataset_size = len(loader['test'].dataset)
        Transformer_model.eval()
        with torch.no_grad():
            for i, data_batch in enumerate(tqdm(loader['test'])):
                data = data_batch[0].to(args.device, non_blocking=True)
                data = data[:,0:2, :, :].to(torch.float32)
                batch_size, channels, seg_len, num_kp = data.shape
                
                if args.token_config == "graph":
                    checkpoint_path =args.encoder_path  # Set your checkpoint path
                    encoder = load_trained_encoder(checkpoint_path, args)  # Load the trained GCAE encoder
                    encoder.to(args.device)  
                    encoder.eval()  

                    encoded_tokens = encoder(data)
                    print("token shape:", encoded_tokens.shape )
                    batch_size, channels, seg_len, num_kp, _ = encoded_tokens.shape  
                    encoded_tokens = encoded_tokens.view(batch_size, seg_len, -1)  # Reshape to (batch, seq_len, features)
                    #Flattened feature dimension (C × num_kp × _)
                    data = encoded_tokens  # Replace input data with extracted tokens

                 # Use graph-based tokens if enabled
                elif args.token_config == "CTR":
                    checkpoint_path_CTR =args.CTR_path  
                    CTR = load_trained_CTR(checkpoint_path_CTR, args)  # Load the trained encoder
                    CTR.to(args.device)  
                    CTR.eval()  

                    encoded_tokens_CTR = CTR(data)
                    print("token shape:", encoded_tokens_CTR.shape )
                    batch_size, channels, seg_len, num_kp, _ = encoded_tokens_CTR.shape  
                    encoded_tokens_CTR = encoded_tokens_CTR.view(batch_size, seg_len, -1)  
                   
                    data = encoded_tokens_CTR  
                
                recon = Transformer_model.forward(data, data)
                loss = loss_func.calculate(data, recon)
                eval_loss.extend(loss.cpu().numpy())
        
        auc_roc, auc_pr, eer, eer_th, fpr_at_target_fnr, threshold_at_target_fnr = score_dataset(np.array(eval_loss), dataset['test'].metadata, args=args)
        print('AUC ROC: {}'.format(auc_roc))
        print('AUC PR: {}'.format(auc_pr))
        print('EER: {}'.format(eer))
        print('EER TH: {}'.format(eer_th))
        print('10ER: {}'.format(fpr_at_target_fnr))
        print('10ER TH: {}'.format(threshold_at_target_fnr))

    
if __name__ == '__main__':    
    main()