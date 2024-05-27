import math
import torch
import torch.nn.functional as F


def compute_cluster_loss(outputs, labels):
    unique_labels = torch.unique(labels)
    label_num = len(unique_labels)

    if label_num == 1:
        return 0.0
    
    for label in unique_labels:
        if torch.sum(labels == label) == 1:
            return 0.0
    
    group_std = torch.stack([torch.std(outputs[labels == label], 0) for label in unique_labels])
    group_avg = torch.stack([torch.mean(outputs[labels == label], 0) for label in unique_labels])

    L_intra = torch.mean(group_std)
    L_inter = 0.0
    for i in range(label_num - 1):
        for j in range(i + 1, label_num):
            L_inter += (group_avg[i] - group_avg[j]).abs().mean()
    loss = math.log(math.exp(L_intra) / math.exp(L_inter / (label_num * (label_num - 1) / 2)))
    return loss

def compute_cosine_similarity(tensor):
    # Normalize the input tensor along the last dimension (d)
    normalized_tensor = F.normalize(tensor, p=2, dim=1)

    # Compute the dot product between each pair of normalized vectors
    similarity_matrix = torch.matmul(normalized_tensor, normalized_tensor.t())

    return similarity_matrix

# def compute_contrast_loss(self, label, k=100, metric='sim', mix_group=True, tau=0.02, contrast_filter=False):
#     losses = []
#     for l in range(self.n_layers):
#         reprs = self.reprs[l]
#         if contrast_filter:
#             contrast_adj = self.transformers[l].contrast_adj.clone()
#             threshold = sorted(contrast_adj.flatten().abs().tolist())[500]
#             mask = torch.zeros_like(contrast_adj)
#             mask[contrast_adj >= threshold] = 1
#             contrast_adj = contrast_adj * mask
#             _, top_contrast_node_idx = torch.topk(contrast_adj.sum(dim=0), k=k, largest=False)
#             # _, top_contrast_node_idx = torch.topk(contrast_adj.abs().sum(dim=0), k=k, largest=True)
#         bz = reprs.shape[0] // self.node_num
#         if metric == 'sim':
#             if mix_group:
#                 batch_sim_matrix = self.compute_cosine_similarity(reprs)
#                 batch_pos_loss, batch_neg_loss = 0.0, 0.0
#                 node_list = top_contrast_node_idx.tolist() if contrast_filter else range(self.node_num)
#                 for i in node_list:
#                     aligned_node_idx = [j * self.node_num + i for j in range(bz)]

#                     sim_matrix = batch_sim_matrix[aligned_node_idx, :][:, aligned_node_idx].clone()
#                     # pos_dis, _ = torch.topk(sim_matrix, k=k, largest=False)  # the node itself has the largest sim
#                     # pos_loss = pos_dis.mean()
#                     pos_loss = sim_matrix.mean()
#                     batch_pos_loss += math.exp(pos_loss / tau)

#                     global_sim_matrix = batch_sim_matrix[aligned_node_idx, :].clone()
#                     global_sim_matrix[:, aligned_node_idx] = 0.0
#                     # neg_dis, _ = torch.topk(global_sim_matrix, k=k, largest=True)
#                     # neg_loss = neg_dis.mean()
#                     neg_loss = global_sim_matrix.mean()
#                     batch_neg_loss += math.exp(neg_loss / tau)
#                 losses.append(-math.log(batch_pos_loss/batch_neg_loss))
#             else:
#                 batch_sim_matrix = self.compute_cosine_similarity(reprs)
#                 batch_pos_loss, batch_neg_loss = 0.0, 0.0
#                 unique_labels = torch.unique(label)
#                 node_list = top_contrast_node_idx.tolist() if contrast_filter else range(self.node_num)
#                 for i in node_list:
#                     aligned_node_idx = [j * self.node_num + i for j in range(bz)]
#                     global_sim_matrix = batch_sim_matrix[aligned_node_idx, :]
#                     global_sim_matrix[:, aligned_node_idx] = 0.0
#                     # neg_dis, _ = torch.topk(global_sim_matrix, k=k, largest=True)
#                     # neg_loss = neg_dis.mean()
#                     neg_loss = global_sim_matrix.mean()
#                     batch_neg_loss += math.exp(neg_loss / tau)
#                     for ul in unique_labels.tolist():
#                         pos_idx = [j * self.node_num + i for j in (label == ul).nonzero().T.squeeze(dim=0).tolist()]
#                         # neg_idx = [j * self.node_num + i for j in (label != ul).nonzero().T.squeeze().tolist()]
#                         pos_sim_matrix = batch_sim_matrix[pos_idx, :][:, pos_idx]
#                         # neg_sim_matrix = batch_sim_matrix[neg_idx, :][:, neg_idx]
#                         pos_loss = pos_sim_matrix.mean()
#                         # neg_loss = neg_sim_matrix.mean()
#                         batch_pos_loss += math.exp(pos_loss / tau)
#                         # batch_neg_loss += neg_loss
#                 losses.append(-math.log(batch_pos_loss / (bz * self.node_num) / batch_neg_loss / (bz * self.node_num)))
#         else:
#             raise NotImplementedError
#     return sum(losses) / self.n_layers
