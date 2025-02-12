import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.modules.loss import _Loss
# from pytorch_metric_learning import losses

# template supervised loss
# template-reaction mutual information maximization

def unique(x, dim=None):
    """Unique elements of x and indices of those unique elements
    https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810

    e.g.

    unique(tensor([
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 3],
        [1, 2, 5]
    ]), dim=0)
    => (tensor([[1, 2, 3],
                [1, 2, 4],
                [1, 2, 5]]),
        tensor([0, 1, 3]))
    """
    unique, inverse = torch.unique(
        x, sorted=True, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype,
                        device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    return unique, inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)


def is_normalized(feature: Tensor, dim=1):
    norms = feature.norm(dim=dim)
    return torch.allclose(norms, torch.ones_like(norms))


def exp_sim_temperature(proj_feat1: Tensor, proj_feat2: Tensor, t: float):
    projections = torch.cat([proj_feat1, proj_feat2], dim=0) #* [batch_size*2, hidden_dim]
    sim_logits = torch.mm(projections, projections.t().contiguous()) / t
    max_value = sim_logits.max().detach()
    sim_logits -= max_value
    sim_exp = torch.exp(sim_logits)
    return sim_exp, sim_logits


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
            logits = logits[label != self.ignore_idx]  #! remove padding tokens
            label = label[label != self.ignore_idx]

        # Apply Label Smoothing:
        with torch.no_grad():
            if logits.shape != label.shape:
                new_label = torch.zeros(logits.shape)
                indices = torch.Tensor([[torch.arange(len(label))[i].item(),
                                         label[i].item()] for i in range(len(label))]).long()
                value = torch.ones(indices.shape[0])  # len_tokens
                label = new_label.index_put_(tuple(indices.t()), value).to(label.device)  # one-hot label
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


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, supcon_level="rxn"):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.supcon_level = supcon_level

    def forward(self, features, labels=None, mask=None, return_all=False):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, f_dim] #! L2-normalized in f_dim (token_length*model_dim)
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        # device = (torch.device('cuda')
        #           if features.is_cuda
        #           else torch.device('cpu'))
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:  #! self-supervised con
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            assert labels.shape[1] == 3
            if self.supcon_level == "rxn":
                labels = labels[:, 2]
            elif self.supcon_level == "template":
                labels = labels[:, 1]
            elif self.supcon_level == "superclass":
                labels = labels[:, 0]
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)   #* [batch_size, batch_size]
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1] 
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) #*[2N, 2N]

        # compute mean of log-likelihood over positive 
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1) #*[2N]

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size)

        if return_all:
            return log_prob, mask
        else:
            return loss.mean()


class HMLC(nn.Module):
    def __init__(self, temperature=0.07,
                 base_temperature=0.07, layer_penalty=None, loss_type='hmce'):
        super(HMLC, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        if layer_penalty == "pow2":
            self.layer_penalty = self.pow_2
        elif layer_penalty == "exp":
            self.layer_penalty = self.exp
        self.sup_con_loss = SupConLoss(temperature=temperature, contrast_mode="all", base_temperature=base_temperature)
        self.loss_type = loss_type

    def pow_2(self, value):
        return torch.pow(2, value)

    def exp(self, value):
        return torch.exp(value)

    def forward(self, features, labels):
        #* features.shape: N*n_view*d
        #* labels.shape: N*L
        #* mask.shape: N*L

        # device = (torch.device('cuda')
        #           if features.is_cuda
        #           else torch.device('cpu'))
        device = features.device
        mask = torch.ones(labels.shape).to(device)  
        mask_labels_lower_layer = None
        cumulative_loss = torch.tensor(0.0).to(device)
        max_loss_lower_layer = torch.tensor(float('-inf')).to(device)
        
        #* L=labels.shape[1]
        #* default implementation: 2-hierar:l=1&2
        #* try: 3-hierar:l=0&1&2 (seems not work)
        #* baselines: 1-hierar:l=0(self-supervised con)/1(template-based supcon)/2(namerxn-based supcon)-->implement using supcon
        
        for l in range(1,labels.shape[1]): 
            mask[:, labels.shape[1]-l:] = 0  #! computing layer_loss in decreasing order of l from L to 0 (layer_loss↑，penalty↓)
            layer_labels = labels * mask
            mask_labels = torch.stack([torch.all(torch.eq(layer_labels[i], layer_labels), dim=1)
                                       for i in range(layer_labels.shape[0])]).type(torch.uint8).to(device)  #* select positive samples with same labels at all levels

            # layer_loss = self.sup_con_loss(features, mask=mask_labels, return_all=False) 

            log_prob, layer_mask = self.sup_con_loss(features, mask=mask_labels, return_all=True) 
            # if mask_labels_lower_layer is not None:
            #     log_prob = log_prob * mask_labels_lower_layer
            #* mask pos samples in lower layer (no need to select unique index)
            # mask_labels_lower_layer = 1 - layer_mask
            
            layer_loss = -((layer_mask * log_prob).sum(1) / layer_mask.sum(1)).mean()

            if self.loss_type == 'hmc':
                cumulative_loss += self.layer_penalty(torch.tensor(
                  1/l).type(torch.float))/2 * layer_loss
            elif self.loss_type == 'hce':
                #! origin code
                layer_loss = torch.max(max_loss_lower_layer.to(layer_loss.device), layer_loss)
                
                #! new implementation
                # layer_loss = torch.max(max_loss_lower_layer, -log_prob)
                # layer_loss = ((layer_mask * layer_loss).sum(1) / layer_mask.sum(1)).mean()

                cumulative_loss += layer_loss
            elif self.loss_type == 'hmce':
                #! origin code
                layer_loss = torch.max(max_loss_lower_layer.to(layer_loss.device), layer_loss)

                #! new implementation
                # layer_loss = torch.max(max_loss_lower_layer, -log_prob)
                # layer_loss = ((layer_mask * layer_loss).sum(1) / layer_mask.sum(1)).mean()

                cumulative_loss += self.layer_penalty(torch.tensor(
                    1/l).type(torch.float))/2 * layer_loss
            elif self.loss_type == 'supcon':
                cumulative_loss += layer_loss
            else:
                raise NotImplementedError('Unknown loss')
            
            #! origin
            max_loss_lower_layer = torch.max(
                max_loss_lower_layer.to(layer_loss.device), layer_loss)

            #! new
            # max_loss_lower_layer = (-(layer_mask * log_prob)).max() #! max layer loss from all positive pairs at layer l
            
            _, unique_indices = unique(layer_labels, dim=0) 
            labels = labels[unique_indices]
            mask = mask[unique_indices]
            features = features[unique_indices]

        return cumulative_loss / labels.shape[1] 


class SelfPacedSupConLoss(nn.Module):
    def __repr__(self):
        message = f"{self.__class__.__name__} with T: {self._t}, method: {self._weight_update} gamma: {self.__gamma}"
        return message

    def __init__(self, temperature=0.07, weight_update="hard", correct_grad=False, **kwargs):
        super().__init__()
        self._t = temperature
        self._weight_update = weight_update
        self.__gamma = 1e6
        self._correct_grad = correct_grad
        print(f"initializing {self.__class__.__name__} with t: {self._t}, cor_grad: {self._correct_grad} ")

    def forward(self, features, target=None, mask: Tensor = None, **kwargs):
        proj_feat1 = features[:, 0, :]
        proj_feat2 = features[:, 1, :]
        batch_size = proj_feat1.shape[0]  #*[batchsize, hidden_dim]
        if mask is not None:
            assert mask.shape == torch.Size([batch_size, batch_size])
            pos_mask = mask == 1
            neg_mask = mask == 0

        elif target is not None:  # supervised mask
            if isinstance(target, list):
                target = torch.Tensor(target).to(device=proj_feat2.device)
            mask = torch.eq(target[..., None], target[None, ...])

            pos_mask = mask == True
            neg_mask = mask == False
        else:
            # only postive masks are diagnal of the sim_matrix
            pos_mask = torch.eye(batch_size, dtype=torch.float, device=proj_feat2.device)  # SIMCLR
            neg_mask = 1 - pos_mask
        gamma = self.__gamma

        return self._forward(proj_feat1, proj_feat2, pos_mask.float(), neg_mask.float(), gamma=gamma, **kwargs)

    def _forward(self, proj_feat1, proj_feat2, pos_mask, neg_mask, gamma=1e6, **kwargs):
        """
        Here the proj_feat1 and proj_feat2 should share the same mask within and cross proj_feat1 and proj_feat2
        :param proj_feat1:
        :param proj_feat2:
        :return:
        """
        assert is_normalized(proj_feat1) and is_normalized(proj_feat2), f"features need to be normalized first"
        assert proj_feat1.shape == proj_feat2.shape, (proj_feat1.shape, proj_feat2.shape)

        batch_size = len(proj_feat1)
        unselect_diganal_mask = 1 - torch.eye(
            batch_size * 2, batch_size * 2, dtype=torch.float, device=proj_feat2.device
        )

        # upscale
        pos_mask = pos_mask.repeat(2, 2)
        neg_mask = neg_mask.repeat(2, 2)

        pos_mask *= unselect_diganal_mask
        neg_mask *= unselect_diganal_mask

        # 2n X 2n
        sim_exp, sim_logits = exp_sim_temperature(proj_feat1, proj_feat2, self._t)
        assert pos_mask.shape == sim_exp.shape == neg_mask.shape, (pos_mask.shape, sim_exp.shape, neg_mask.shape)

        # =============================================
        # in order to have a hook for further processing
        self.sim_exp = sim_exp
        self.sim_logits = sim_logits
        self.pos_mask = pos_mask
        self.neg_mask = neg_mask
        # ================= end =======================
        pos_count, neg_count = pos_mask.sum(1), neg_mask.sum(1)
        pos_sum = (sim_exp * pos_mask).sum(1, keepdim=True).repeat(1, batch_size * 2)
        neg_sum = (sim_exp * neg_mask).sum(1, keepdim=True).repeat(1, batch_size * 2)

        log_pos_div_sum_pos_neg = sim_logits - torch.log(pos_sum + neg_sum + 1e-16)
        assert log_pos_div_sum_pos_neg.shape == torch.Size([batch_size * 2, batch_size * 2])

        self_paced_mask = self._self_paced_mask(log_pos_div_sum_pos_neg, gamma, pos_mask=pos_mask)  #* log_pos_div_sum_pos_neg=-loss_ij
        self.sp_mask = self_paced_mask
        batch_downgrade_ratio = torch.masked_select(self_paced_mask, pos_mask.bool()).mean().item()

        self.downgrade_ratio = batch_downgrade_ratio

        log_pos_div_sum_pos_neg *= self_paced_mask

        # over positive mask
        loss = (log_pos_div_sum_pos_neg * pos_mask).sum(1) / pos_count
        loss = -loss.mean()

        if self._correct_grad:
            if batch_downgrade_ratio > 0:
                loss /= batch_downgrade_ratio

        if torch.isnan(loss):
            raise RuntimeError(loss)
        return loss

    @torch.no_grad()
    def _self_paced_mask(self, llh_matrix, gamma, *, pos_mask):
        l_i_j = -llh_matrix
        if self._weight_update == "hard":
            _weight = (l_i_j <= gamma).float()
        else:
            _weight = torch.max(1 - 1 / gamma * l_i_j, torch.zeros_like(l_i_j))
        return torch.max(_weight, 1 - pos_mask)

    def set_gamma(self, gamma):
        print(f"{self.__class__.__name__} set gamma as {gamma}")
        self.__gamma = float(gamma)

    @property
    def age_param(self):
        return self.__gamma


class SelfPacedHMLC(nn.Module):
    def __init__(self, temperature=0.07,
                 base_temperature=0.07, layer_penalty=None, loss_type='hmce', weight_update="hard", correct_grad=False):
        super(SelfPacedHMLC, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        if not layer_penalty:
            self.layer_penalty = self.pow_2
        else:
            self.layer_penalty = layer_penalty
        self.sup_con_loss = SupConLoss(temperature=temperature, contrast_mode="all", base_temperature=base_temperature)
        self.loss_type = loss_type

        self._weight_update = weight_update
        self.__gamma = 1e6
        self._correct_grad = correct_grad

    def pow_2(self, value):
        return torch.pow(2, value)

    def forward(self, features, labels):
        #* features.shape: N*n_view*d
        #* labels.shape: N*L
        #* mask.shape: N*L

        device = features.device
        mask = torch.ones(labels.shape).to(device)  
        cumulative_loss = torch.tensor(0.0).to(device)
        max_loss_lower_layer = torch.tensor(float('-inf')).to(device)

        for l in range(1,labels.shape[1]): 
            mask[:, labels.shape[1]-l:] = 0  #! computing layer_loss in decreasing order of l from L to 0 (layer_loss↑，penalty↓)
            layer_labels = labels * mask
            mask_labels = torch.stack([torch.all(torch.eq(layer_labels[i], layer_labels), dim=1)
                                       for i in range(layer_labels.shape[0])]).type(torch.uint8).to(device)  #* select positive samples with same labels at all levels
            log_prob, layer_mask = self.sup_con_loss(features, mask=mask_labels, return_all=True) 
            sp_mask = self._self_paced_mask(llh_matrix=log_prob, gamma=self.__gamma, pos_mask=layer_mask)
            batch_downgrade_ratio = torch.masked_select(sp_mask, layer_mask.bool()).mean().item()
            log_prob *= sp_mask
            layer_loss = -((layer_mask * log_prob).sum(1) / layer_mask.sum(1)).mean()
            if self._correct_grad:
                if batch_downgrade_ratio > 0:
                    layer_loss /= batch_downgrade_ratio

            if self.loss_type == 'hmc':
                cumulative_loss += self.layer_penalty(torch.tensor(
                  1/(l)).type(torch.float)) * layer_loss
            elif self.loss_type == 'hce':
                layer_loss = torch.max(max_loss_lower_layer.to(layer_loss.device), layer_loss)
                cumulative_loss += layer_loss
            elif self.loss_type == 'hmce':
                layer_loss = torch.max(max_loss_lower_layer.to(layer_loss.device), layer_loss)
                cumulative_loss += self.layer_penalty(torch.tensor(
                    1/l).type(torch.float)) * layer_loss
            else:
                raise NotImplementedError('Unknown loss')
            
            max_loss_lower_layer = torch.max(
                max_loss_lower_layer.to(layer_loss.device), layer_loss)
            
            _, unique_indices = unique(layer_labels, dim=0) 
            labels = labels[unique_indices]
            mask = mask[unique_indices]
            features = features[unique_indices]

        return cumulative_loss / labels.shape[1]
    
    @torch.no_grad()
    def _self_paced_mask(self, llh_matrix, gamma, *, pos_mask):
        l_i_j = -llh_matrix
        if self._weight_update == "hard":
            _weight = (l_i_j <= gamma).float()
        else:
            _weight = torch.max(1 - 1 / gamma * l_i_j, torch.zeros_like(l_i_j))
        return torch.max(_weight, 1 - pos_mask)

    def set_gamma(self, gamma):
        print(f"{self.__class__.__name__} set gamma as {gamma}")
        self.__gamma = float(gamma)

    @property
    def age_param(self):
        return self.__gamma


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
    graph_reps_norm = torch.norm(graph_reps, dim = 1).unsqueeze(-1)  #! check the unit norm is correct?
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
    #! info-NCE loss (derived from MOCO) is only applicable for one positive sample
    #! Ref of SupCon loss: https://proceedings.neurips.cc/paper_files/paper/2020/file/d89a66c7c80a29b1bdbab0f2a1a94af8-Paper.pdf
    #! Meta-label: NameRXN classfication; reaction templates; maybe other meta-data in open reaction database?
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

def weighted_mse_loss(inputs, targets, weights=None):
    loss = (inputs - targets) ** 2
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    # loss = torch.sum(loss)
    return loss

def weighted_focal_mse_loss(inputs, targets, weights=None, activate='sigmoid', beta=.2, gamma=1):
    loss = (inputs - targets) ** 2
    # loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
    #     (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    focal_weight = (inputs - targets) ** 2 / ((inputs - targets) ** 2).sum()
    loss *= focal_weight
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    # loss = torch.sum(loss)
    return loss

class focal_loss(nn.Module):
    def __init__(self, alpha=None, gamma=2, num_classes=3, size_average=True, device="cuda"):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(focal_loss,self).__init__()
        self.size_average = size_average
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中第一类为背景类
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma
        self.alpha = self.alpha.to(device)
        
        print('Focal Loss:')
        print('Alpha = {}'.format(self.alpha))
        print('Gamma = {}'.format(self.gamma))
        
    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1) # log_softmax
        preds_softmax = torch.exp(preds_logsoft)    # softmax

        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class BMCLoss(_Loss):
    def __init__(self, init_noise_sigma=1.0):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        return self.bmc_loss(pred.view(-1, 1), target.view(-1, 1), noise_var)
    
    def bmc_loss(self, pred, target, noise_var):
        """Compute the Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
        Args:
        pred: A float tensor of size [batch, 1].
        target: A float tensor of size [batch, 1].
        noise_var: A float number or tensor.
        Returns:
        loss: A float tensor. Balanced MSE Loss.
        """
        logits = - (pred - target.T).pow(2) / (2 * noise_var)   # logit size: [batch, batch]
        loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).to(logits.device))   # contrastive-like loss
        loss = loss * (2 * noise_var).detach()  # optional: restore the loss scale, 'detach' when noise is learnable 

        return loss