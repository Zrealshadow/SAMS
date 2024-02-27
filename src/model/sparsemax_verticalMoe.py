from copy import deepcopy
from typing import List
import torch
import torch.nn as nn


import argparse
from src.model.expert import initialize_expert
from src.model.gate_network import SparseVerticalGate

from third_party.utils.model_utils import EntmaxBisect






class SparseMax_VerticalSAMS(nn.Module):
    
    def __init__(self, args:argparse.Namespace):
        super().__init__()
        
        self.args = args
        
        col_cardinality_sum = args.nfield + args.nfeat + 1
        self.gate_input_size = args.nfield * args.sql_nemb
        
        self.input_size = args.nfield * args.data_nemb
        self.hidden_size = args.moe_hid_layer_len
        self.output_size = args.output_size
        self.num_experts = args.K

        
        self.gate_network = SparseVerticalGate(self.gate_input_size, self.num_experts ,args.hid_layer_len, 0.1)
        
        expert = initialize_expert(args)
        self.experts = nn.ModuleList([deepcopy(expert) for _ in range(self.num_experts)])

        
        self.sql_embedding = nn.Embedding(col_cardinality_sum, args.sql_nemb)
        self.input_embedding = nn.Embedding(args.nfeat, args.data_nemb)

        nn.init.xavier_uniform_(self.sql_embedding.weight)
        nn.init.xavier_uniform_(self.input_embedding.weight)

        self.alpha = args.alpha
        # self.alpha = nn.Parameter(torch.tensor(args.alpha, dtype=torch.float32), requires_grad=True)
        self.sparsemax = EntmaxBisect(alpha=self.alpha)
        

    
    def forward(self, x:torch.Tensor, sql:torch.Tensor):
        """
        Args:
        x:      [B, nfield] only Id
        sql:    [B, nfield]
        Returns:
        y:      [B]
        """
        if self.args.expert == "afn":
            # should clip the embedding
            # since the input will be transfered into log space, should be positive
            self.embedding_clip()
            
        x_emb = self.input_embedding(x)         
        sql_emb = self.sql_embedding(sql)
        
        B = x.shape[0]
        # x_emb = x_emb.view(B, -1)
        sql_emb = sql_emb.view(B, -1)
        
        gate_score = self.gate_network(sql_emb)
        gate_score = self.sparsemax(gate_score)
        
        
        # --------- Calculate Importance -----------
        importance = gate_score.sum(0)
        
        imp_loss = self.cv_squared(importance)
        
        # --------- Calculate Sparse Term ------------
        spa_loss = self.l2_regularization(gate_score)
        
        x_list = [ expert(x_emb) for expert in self.experts]
        x = torch.stack(x_list, dim = 1) 
        # [B, N, nhid]
        
        # sum according to weight
        x = gate_score.unsqueeze(-1) * x
        # [B, N, nhid]
        
        x = torch.sum(x, dim = 1)
        # [B, nhid]
        
        return x.squeeze(-1), (imp_loss, spa_loss)
    
    
    def l2_regularization(self, x):
        """
        Parameters
        ----------
        x : [B, K]
        """
        
        loss = torch.mean(torch.sqrt(torch.sum(x**2, dim = 1)))
        return loss
    
    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def embedding_clip(self):
        "keep AFN expert embeddings positive"
        with torch.no_grad():
            self.input_embedding.weight.abs_()
            self.input_embedding.weight.clamp_(min=1e-4)
            
            
    
    def cal_gate_score(self, sql:torch.Tensor):
        B = sql.shape[0]
        sql_emb = self.sql_embedding(sql).view(B, -1)
        gate_score = self.gate_network(sql_emb)
        gate_score = self.sparsemax(gate_score)
        return gate_score
     

    def tailor_by_sql(self, sql:torch.Tensor):
        """
        sql: [1, F]
        """
        B = sql.shape[0]
        sql_emb = self.sql_embedding(sql).view(B, -1)
        gate_score = self.gate_network(sql_emb)
        gate_score = self.sparsemax(gate_score)
        
        expert_score = gate_score.squeeze(0)
        non_zero_indices = torch.nonzero(expert_score, as_tuple=False).t()
        non_zero_index = non_zero_indices.squeeze(0).cpu()
        non_zero_index = non_zero_index.numpy().tolist()
        
        non_zero_scores = expert_score[non_zero_indices] #[B,K], K-> none zero scores
        
        selected_experts = [ deepcopy(self.experts[i]) for i in non_zero_index]
        embedding = deepcopy(self.input_embedding)
        
        return SliceModel(embedding, selected_experts, non_zero_scores)
    

class SliceModel(nn.Module):
    
    def __init__(self, embeding:nn.Module, experts:List[nn.Module], weight:torch.Tensor):
        super().__init__()
        
        self.embedding = embeding
        self.experts = nn.ModuleList(experts)
        self.weight = nn.Parameter(weight) # [1, K]
    
    def forward(self, x, _):
        """
        x : [B,F]
        """
        x_emb = self.embedding(x)   # [B,F,E]
        x_list = [expert(x_emb) for expert in self.experts]
        x = torch.stack(x_list, dim = 1)    # [B, K, nhid]
        x = self.weight.unsqueeze(-1) * x
        x = torch.sum(x, dim = 1)           # [B, nhid]
        return x.squeeze(-1)