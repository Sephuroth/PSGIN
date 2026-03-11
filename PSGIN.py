# -*- coding:utf-8 -*-
"""
Filename: PSGIN.py
Time: 2025-10-11
"""
import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    """Temporal Attention"""

    def __init__(self, dim_in, dim_out=None):
        super().__init__()
        dim_out = dim_out or dim_in * 4

        self.query_proj = nn.Linear(dim_in, dim_out, bias=False)
        self.key_proj = nn.Linear(dim_in, dim_out, bias=True)
        self.score_proj = nn.Linear(dim_out, 1, bias=False)

    def forward(self, query, keys):
        # query: [batch, num_nodes, features]
        # keys:  [batch, num_time, num_nodes, features]

        q = self.query_proj(query).unsqueeze(1).unsqueeze(3)  # [B, 1, N, 1, D_out]
        k = self.key_proj(keys).unsqueeze(2)  # [B, T, 1, N, D_out]

        attn_scores = self.score_proj(torch.tanh(q + k)).squeeze(-1)  # [B, T, N, N]

        return torch.softmax(attn_scores, dim=-1)


class DynamicGconv(nn.Module):
    """Graph convolution for adaptive/dynamic adjacency matrices"""

    def __init__(self):
        super(DynamicGconv, self).__init__()

    def forward(self, x, A):
        # x: [batch, num_nodes, features]
        # A: [batch, num_nodes, num_nodes] or [num_nodes, num_nodes]

        if A.dim() == 2:
            A = A.unsqueeze(0).expand(x.size(0), -1, -1)
        elif A.size(0) != x.size(0):
            A = A.expand(x.size(0), -1, -1)

        return torch.bmm(A.transpose(1, 2), x)


class StaticGconv(nn.Module):
    """Graph convolution for predefine/static adjacency matrices"""

    def __init__(self):
        super(StaticGconv, self).__init__()

    def forward(self, x, A):
        # x: [batch, num_nodes, features], A: [num_nodes, num_nodes]
        x = torch.einsum('nvc,vw->nwc', x, A)
        return x.contiguous()


class GCN(nn.Module):
    """Graph convolution network"""

    def __init__(self, dims, gdep, dropout, alpha, beta, gamma, type=None):
        super(GCN, self).__init__()
        self.type_GNN = type
        self.gdep = gdep
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.dropout = dropout

        if type == 'gru':
            self.gconv = DynamicGconv()
            self.gconv_preA = StaticGconv()
        elif type == 'hyper':
            self.gconv_preA = StaticGconv()
        elif type == 'ss_gal':
            self.gconv = DynamicGconv()
        else:
            raise ValueError(f"Unsupported GCN type: {type}")

        self.mlp = nn.Linear((gdep + 1) * dims[0], dims[1])
        self.dropout_layer = nn.Dropout(p=dropout) if dropout else nn.Identity()

    def forward(self, x, adj):
        h = x
        out = [h]
        for _ in range(self.gdep):
            if self.type_GNN == 'gru':
                # adj[0]: adaptive graph
                # adj[1]: predefined graph
                h = self.alpha * x + self.beta * self.gconv(h, adj[0]) + self.gamma * self.gconv_preA(h, adj[1])
            elif self.type_GNN == 'hyper':
                h = self.alpha * x + self.gamma * self.gconv_preA(h, adj)
            elif self.type_GNN == 'ss_gal':
                h = self.alpha * x + self.gamma * self.gconv(h, adj)
            out.append(h)

        ho = torch.cat(out, dim=-1)
        ho = self.mlp(ho)
        return self.dropout_layer(ho)


class PSGIN(nn.Module):
    """Pattern-Decoupled Spatiotemporal Graph Imputation Network (PSGIN) main module"""

    def __init__(self, input_dim, hyper_gcn_params, ss_gal_params, hidden_dim, dropout=0.2, tanh_alpha=2,
                 gcn_weights=[0.05, 0.95, 0.95]):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.tanh_alpha = tanh_alpha

        # ==========================================
        # 1. Short-term component: Short-term Spatiotemporal Graph Attention Layer (SS-GAL)
        # ==========================================
        self.ss_gal_layers = nn.ModuleList()
        for layer_name, layer_param in ss_gal_params.items():
            # 确定输入维度
            ss_gal_input_dim = input_dim if layer_param['dims_ss_gal'][0] == -1 else layer_param['dims_ss_gal'][0]

            ss_gal_block = nn.ModuleDict({
                'temporal_attention': TemporalAttention(ss_gal_input_dim),
                'spatial_gcn': GCN([ss_gal_input_dim, layer_param['dims_ss_gal'][1]], layer_param['depth_ss_gal'], dropout,
                                   *gcn_weights, type='ss_gal'),
            })
            self.ss_gal_layers.append(ss_gal_block)

        self.num_ss_gal_layers = len(self.ss_gal_layers)
        last_ss_gal_dim = ss_gal_params[list(ss_gal_params.keys())[-1]]['dims_ss_gal'][-1]

        # ==========================================
        # 2. Inductive Dynamic Graph Generator (IDGG)
        # ==========================================
        dims_hyper = hyper_gcn_params['dims_hyper']
        dims_hyper[0] = hidden_dim
        gcn_depth = hyper_gcn_params['depth_GCN']

        self.source_graph_generator = GCN(dims_hyper, gcn_depth, dropout, *gcn_weights, type='hyper')
        self.target_graph_generator = GCN(dims_hyper, gcn_depth, dropout, *gcn_weights, type='hyper')

        self.source_node_embedding = nn.Linear(self.input_dim, dims_hyper[-1])
        self.target_node_embedding = nn.Linear(self.input_dim, dims_hyper[-1])

        # ==========================================
        # 3. Long-term component: Long-term Spatiotemporal Graph Recurrent Unit (LS-GRU)
        # ==========================================
        gru_input_dim = [self.input_dim + self.hidden_dim, self.hidden_dim]

        self.gru_update_gate = GCN(gru_input_dim, gcn_depth, dropout, *gcn_weights, type='gru')
        self.gru_reset_gate = GCN(gru_input_dim, gcn_depth, dropout, *gcn_weights, type='gru')
        self.gru_candidate_state = GCN(gru_input_dim, gcn_depth, dropout, *gcn_weights, type='gru')

        # ==========================================
        # 4. Final Predictors
        # ==========================================
        self.short_term_predictor = nn.Linear(last_ss_gal_dim, self.input_dim)
        self.long_term_predictor = nn.Linear(self.hidden_dim, self.input_dim)
        self.fusion_predictor = nn.Linear(last_ss_gal_dim + self.hidden_dim, self.input_dim)

    def forward(self, input_sequence, static_adj):
        """
        forward
        :param input_sequence: [batch_size, num_timesteps, num_nodes, features]
        :param static_adj: predefined adjacent matrix [num_nodes, num_nodes]
        """
        batch_size, num_timesteps, num_nodes = input_sequence.shape[0], input_sequence.shape[1], input_sequence.shape[2]
        device = input_sequence.device

        # 初始化 Graph GRU 的全局隐藏状态 [batch * num_nodes, hidden_dim]
        global_hidden_state = torch.zeros(batch_size * num_nodes, self.hidden_dim, device=device)

        # =========================================================================
        # Stage 1：Long-term Spatiotemporal Graph Recurrent Unit (LS-GRU)
        # =========================================================================
        for t_step in range(0, max(1, num_timesteps - 4), 4):
            current_step_data = input_sequence[:, t_step]

            # reshape [batch, num_nodes, hidden_dim]
            node_hidden_states = global_hidden_state.view(-1, num_nodes, self.hidden_dim)

            # --- 1. generate adaptive adjacency matrix ---
            source_graph_context = self.source_graph_generator(node_hidden_states, static_adj)
            target_graph_context = self.target_graph_generator(node_hidden_states, static_adj)

            source_node_features = torch.tanh(
                self.tanh_alpha * torch.mul(self.source_node_embedding(current_step_data), source_graph_context))
            target_node_features = torch.tanh(
                self.tanh_alpha * torch.mul(self.target_node_embedding(current_step_data), target_graph_context))

            # Calculate antisymmetric similarity
            asymmetric_similarity = torch.matmul(source_node_features, target_node_features.transpose(2, 1)) - \
                                    torch.matmul(target_node_features, source_node_features.transpose(2, 1))

            adaptive_adj = torch.relu(torch.tanh(self.tanh_alpha * asymmetric_similarity))

            # Add self-loops and normalization
            adaptive_adj = adaptive_adj + torch.eye(num_nodes, device=device)
            adaptive_adj = adaptive_adj / (adaptive_adj.sum(dim=-1, keepdim=True) + 1e-8)
            combined_adj_matrices = [adaptive_adj, static_adj]

            # --- 2. LS-GRU status update ---
            gru_input = torch.cat((current_step_data, node_hidden_states), dim=-1)

            update_gate = torch.sigmoid(self.gru_update_gate(gru_input, combined_adj_matrices))
            reset_gate = torch.sigmoid(self.gru_reset_gate(gru_input, combined_adj_matrices))

            reset_gru_input = torch.cat((current_step_data, torch.mul(reset_gate, node_hidden_states)), dim=-1)
            candidate_state = torch.tanh(self.gru_candidate_state(reset_gru_input, combined_adj_matrices))

            # Hidden status update： H_t = Z * H_{t-1} + (1 - Z) * \tilde{H}
            node_hidden_states = torch.mul(update_gate, node_hidden_states) + torch.mul(1 - update_gate,
                                                                                        candidate_state)

            # reshape [batch * num_nodes, hidden_dim]
            global_hidden_state = node_hidden_states.view(batch_size * num_nodes, self.hidden_dim)

        # =========================================================================
        # Stage 2：Short-term Spatiotemporal Graph Attention Layer (SS-GAL)
        # =========================================================================
        all_temporal_attn_scores = []
        recent_history_features = input_sequence[:, -4:]
        target_step_features = input_sequence[:, -1]

        for i in range(self.num_tgn_layers):
            # 1. Calculate attention score
            temporal_attn_scores = self.tgn_layers[i]['temporal_attention'](target_step_features,
                                                                            recent_history_features)
            all_temporal_attn_scores.append(temporal_attn_scores)

            # 2. Generate dynamic spatiotemporal graphs
            recent_history_features = recent_history_features.reshape(-1, num_nodes, recent_history_features.shape[-1])
            dynamic_spatiotemporal_adj = static_adj.unsqueeze(0) + temporal_attn_scores.reshape(-1, num_nodes,
                                                                                                num_nodes)

            # 3. Graph convolution
            recent_history_features = self.tgn_layers[i]['spatial_gcn'](recent_history_features,
                                                                        dynamic_spatiotemporal_adj)
            recent_history_features = recent_history_features.reshape(batch_size, -1, num_nodes,
                                                                      recent_history_features.shape[-1])
            recent_history_features = torch.relu(recent_history_features)

            target_step_features = torch.sum(recent_history_features, dim=1)

        short_term_output = self.short_term_predictor(target_step_features).view(batch_size, num_nodes, -1)

        # =========================================================================
        # Stage 3：Mix outputs
        # =========================================================================
        current_step_data = short_term_output
        node_hidden_states = global_hidden_state.view(-1, num_nodes, self.hidden_dim)

        source_graph_context = self.source_graph_generator(node_hidden_states, static_adj)
        target_graph_context = self.target_graph_generator(node_hidden_states, static_adj)

        source_node_features = torch.tanh(
            self.tanh_alpha * torch.mul(self.source_node_embedding(current_step_data), source_graph_context))
        target_node_features = torch.tanh(
            self.tanh_alpha * torch.mul(self.target_node_embedding(current_step_data), target_graph_context))

        asymmetric_similarity = torch.matmul(source_node_features, target_node_features.transpose(2, 1)) - \
                                torch.matmul(target_node_features, source_node_features.transpose(2, 1))
        adaptive_adj = torch.relu(torch.tanh(self.tanh_alpha * asymmetric_similarity))

        adaptive_adj = adaptive_adj + torch.eye(num_nodes, device=device)
        adaptive_adj = adaptive_adj / (adaptive_adj.sum(dim=-1, keepdim=True) + 1e-8)
        combined_adj_matrices = [adaptive_adj, static_adj]

        gru_input = torch.cat((current_step_data, node_hidden_states), dim=-1)
        update_gate = torch.sigmoid(self.gru_update_gate(gru_input, combined_adj_matrices))
        reset_gate = torch.sigmoid(self.gru_reset_gate(gru_input, combined_adj_matrices))

        reset_gru_input = torch.cat((current_step_data, torch.mul(reset_gate, node_hidden_states)), dim=-1)
        candidate_state = torch.tanh(self.gru_candidate_state(reset_gru_input, combined_adj_matrices))

        node_hidden_states = torch.mul(update_gate, node_hidden_states) + torch.mul(1 - update_gate, candidate_state)

        final_long_hidden = node_hidden_states.view(batch_size, num_nodes, -1)
        long_term_output = self.long_term_predictor(final_long_hidden)

        fused_output = self.fusion_predictor(torch.cat([target_step_features, final_long_hidden], dim=2))

        return short_term_output, long_term_output, fused_output, all_temporal_attn_scores[
            0] if all_temporal_attn_scores else None