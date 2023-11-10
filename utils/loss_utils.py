import torch
import torch.nn as nn
import torch.nn.functional as F


# template supervised loss
# template-reaction mutual information maximization


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, reduction='mean', apply_logsoftmax=True, ignore_index=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.apply_logsoftmax = apply_logsoftmax
        self.ignore_idx = ignore_index

    def forward(self, logits, label):
        if logits.shape != label.shape and self.ignore_idx != -1:
            logits = logits[label != self.ignore_idx]
            label = label[label != self.ignore_idx]

        # Apply Label Smoothing:
        with torch.no_grad():
            if logits.shape != label.shape:
                new_label = torch.zeros(logits.shape)
                indices = torch.Tensor([[torch.arange(len(label))[i].item(),
                                         label[i].item()] for i in range(len(label))]).long()
                value = torch.ones(indices.shape[0])
                label = new_label.index_put_(tuple(indices.t()), value).to(label.device)
                label = label * (1 - self.smoothing) + self.smoothing / logits.shape[-1]
                label = label / label.sum(-1)[:, None]

            elif self.ignore_idx != -1:  # for context alignment loss
                label_lengths = (label != 2).sum(dim=-1)
                valid_indices = label_lengths != 0

                exist_align = (label == 1).sum(dim=-1) > 0
                smoothed_logits_addon = self.smoothing / label_lengths
                smoothed_logits_addon[smoothed_logits_addon > 1] = 0

                tmp = label.clone()
                tmp = tmp * (1 - self.smoothing) + smoothed_logits_addon.unsqueeze(1)
                tmp[label == 2] = 0

                label = tmp[valid_indices & exist_align]
                logits = logits[valid_indices & exist_align]

            else:
                label = label * (1 - self.smoothing) + self.smoothing / logits.shape[-1]
                label = label / label.sum(-1)[:, None]

        if self.apply_logsoftmax:
            logs = self.log_softmax(logits)
        else:
            logs = logits

        loss = -torch.sum(logs * label, dim=1)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


# NCE loss within a graph
def intra_NCE_loss(node_reps, node_modify_reps, batch, tau=0.1, epsilon=1e-6):
    node_reps_norm = torch.norm(node_reps, dim = 1).unsqueeze(-1)
    node_modify_reps_norm = torch.norm(node_modify_reps, dim = 1).unsqueeze(-1)
    sim = torch.mm(node_reps, node_modify_reps.t()) / (
        torch.mm(node_reps_norm, node_modify_reps_norm.t()) + epsilon)
    exp_sim = torch.exp(sim / tau)

    mask = torch.stack([(batch.batch == i).float() for i in batch.batch.tolist()], dim = 1)
    exp_sim_mask = exp_sim * mask
    exp_sim_all = torch.index_select(exp_sim_mask, 1, batch.masked_node_indices)
    exp_sim_positive = torch.index_select(exp_sim_all, 0, batch.masked_node_indices)
    positive_ratio = exp_sim_positive.sum(0) / (exp_sim_all.sum(0) + epsilon)

    NCE_loss = -torch.log(positive_ratio).sum() / batch.masked_node_indices.shape[0]
    mask_select = torch.index_select(mask, 1, batch.masked_node_indices)
    thr = 1. / mask_select.sum(0)
    correct_cnt = (positive_ratio > thr).float().sum()

    return NCE_loss, correct_cnt

# NCE loss across different graphs
def inter_NCE_loss(graph_reps, graph_modify_reps, device, tau=0.1, epsilon=1e-6):
    graph_reps_norm = torch.norm(graph_reps, dim = 1).unsqueeze(-1)
    graph_modify_reps_norm = torch.norm(graph_modify_reps, dim = 1).unsqueeze(-1)
    sim = torch.mm(graph_reps, graph_modify_reps.t()) / (
        torch.mm(graph_reps_norm, graph_modify_reps_norm.t()) + epsilon)
    exp_sim = torch.exp(sim / tau)

    mask = torch.eye(graph_reps.shape[0]).to(device)
    positive = (exp_sim * mask).sum(0)
    negative = (exp_sim * (1 - mask)).sum(0)
    positive_ratio = positive / (positive + negative + epsilon)

    NCE_loss = -torch.log(positive_ratio).sum() / graph_reps.shape[0]
    thr = 1. / ((1 - mask).sum(0) + 1.)
    correct_cnt = (positive_ratio > thr).float().sum()

    return NCE_loss, correct_cnt


def supervised_NCE_loss(graph_reps, labels, device, tau=0.1, epsilon=1e-6):
    """
        graph_reps: (batch_size, 2*d_model)
        labels: (batch_size)
    """
    graph_reps_norm = torch.norm(graph_reps, dim = 1).unsqueeze(-1)
    sim = torch.mm(graph_reps, graph_reps.t()) / (
        torch.mm(graph_reps_norm, graph_reps_norm.t()) + epsilon)  # cosine similarity
    exp_sim = torch.exp(sim / tau)

    mask = labels.unsqueeze(1).eq(labels.unsqueeze(0)).float().to(device)

    positive = (exp_sim * mask).sum(0)
    negative = (exp_sim * (1 - mask)).sum(0)
    positive_ratio = positive / (positive + negative + epsilon)

    NCE_loss = -torch.log(positive_ratio).sum() / graph_reps.shape[0]
    thr = 1. / ((1 - mask).sum(0) + 1.)
    correct_cnt = (positive_ratio > thr).float().sum()

    return NCE_loss, correct_cnt


# NCE loss for global-local mutual information maximization
def gl_NCE_loss(node_reps, graph_reps, batch, tau=0.1, epsilon=1e-6):
    node_reps_norm = torch.norm(node_reps, dim = 1).unsqueeze(-1)
    graph_reps_norm = torch.norm(graph_reps, dim = 1).unsqueeze(-1)
    sim = torch.mm(node_reps, graph_reps.t()) / (
            torch.mm(node_reps_norm, graph_reps_norm.t()) + epsilon)
    exp_sim = torch.exp(sim / tau)

    mask = torch.stack([(batch.batch == i).float() for i in range(graph_reps.shape[0])], dim = 1)
    positive = exp_sim * mask
    negative = exp_sim * (1 - mask)
    positive_ratio = positive / (positive + negative.sum(0).unsqueeze(0) + epsilon)

    NCE_loss = -torch.log(positive_ratio + (1 - mask)).sum() / node_reps.shape[0]
    thr = 1. / ((1 - mask).sum(0) + 1.).unsqueeze(0)
    correct_cnt = (positive_ratio > thr).float().sum()

    return NCE_loss, correct_cnt

# NCE loss between graphs and prototypes
def proto_NCE_loss(graph_reps, proto, proto_connection, tau=0.1, epsilon=1e-6):
    # similarity for original and modified graphs
    graph_reps_norm = torch.norm(graph_reps, dim=1).unsqueeze(-1)
    exp_sim_list = []
    mask_list = []
    NCE_loss = 0

    for i in range(len(proto)-1, -1, -1):
        tmp_proto = proto[i]
        proto_norm = torch.norm(tmp_proto, dim=1).unsqueeze(-1)

        sim = torch.mm(graph_reps, tmp_proto.t()) / (
                torch.mm(graph_reps_norm, proto_norm.t()) + epsilon)
        exp_sim = torch.exp(sim / tau)

        if i != (len(proto) - 1):
            # apply the connection mask
            exp_sim_last = exp_sim_list[-1]
            idx_last = torch.argmax(exp_sim_last, dim = 1).unsqueeze(-1)
            connection = proto_connection[i]
            connection_mask = (connection.unsqueeze(0) == idx_last.float()).float()
            exp_sim = exp_sim * connection_mask

            # define NCE loss between prototypes from consecutive layers
            upper_proto = proto[i+1]
            upper_proto_norm = torch.norm(upper_proto, dim=1).unsqueeze(-1)
            proto_sim = torch.mm(tmp_proto, upper_proto.t()) / (
                    torch.mm(proto_norm, upper_proto_norm.t()) + epsilon)
            proto_exp_sim = torch.exp(proto_sim / tau)

            proto_positive_list = [proto_exp_sim[j, connection[j].long()] for j in range(proto_exp_sim.shape[0])]
            proto_positive = torch.stack(proto_positive_list, dim=0)
            proto_positive_ratio = proto_positive / (proto_exp_sim.sum(1) + epsilon)
            NCE_loss += -torch.log(proto_positive_ratio).mean()

        mask = (exp_sim == exp_sim.max(1)[0].unsqueeze(-1)).float()

        exp_sim_list.append(exp_sim)
        mask_list.append(mask)

    # define NCE loss between graph embedding and prototypes
    for i in range(len(proto)):
        exp_sim = exp_sim_list[i]
        mask = mask_list[i]

        positive = exp_sim * mask
        negative = exp_sim * (1 - mask)
        positive_ratio = positive.sum(1) / (positive.sum(1) + negative.sum(1) + epsilon)
        NCE_loss += -torch.log(positive_ratio).mean()

    return NCE_loss