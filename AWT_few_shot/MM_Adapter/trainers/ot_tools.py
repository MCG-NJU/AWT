import torch
import torch.nn.functional as F


def calculate_batch_entropy(logits):
    return -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1)

@torch.no_grad()
def get_entropy_weight(output, img_t=0.5, text_t=0.5):
    # output: bs x aug_time x n_des x n_cls
    bs, aug_time, n_des, n_cls = output.shape
    with torch.cuda.amp.autocast():
        # get weights for images
        image_entropy = calculate_batch_entropy(output.mean(-2))
        image_weights = F.softmax(-image_entropy/img_t, dim=-1) # bs x aug_time
        image_weights = image_weights[:, None, :].repeat(1, n_cls, 1).reshape(bs*n_cls, aug_time)

        # get weights for descriptors
        anchor = output[:, :1].mean(-2, keepdim=True).repeat(1, n_des, n_cls, 1) # bs x n_des x n_cls x n_cls
        output_des = output[:, 0].unsqueeze(-1) # bs x n_des x n_cls x 1
        scatter_indices = torch.arange(n_cls)[None, None, :, None].repeat(bs, n_des, 1, 1).cuda() # bs x n_des x n_cls x 1
        anchor.scatter_(dim=-1, index=scatter_indices, src=output_des) # bs x n_des x n_cls x n_cls
        text_entropy = calculate_batch_entropy(anchor)
        text_weights = F.softmax(-text_entropy/text_t, dim=1) # bs x n_des x n_cls
        text_weights = text_weights.permute(0, 2, 1).reshape(-1, n_des)

    return image_weights, text_weights

def Sinkhorn(K, u, v):
    r = torch.ones_like(u)
    c = torch.ones_like(v)
    thresh = 1e-2
    for i in range(100):
        r0 = r
        r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
        c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
        err = (r - r0).abs().mean()
        if err.item() < thresh:
            break
    T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K
    return T

def optimal_transport(logits, logit_scale, image_weights, text_weights):
    # logits: bs x aug_time x n_des x n_cls
    bs, aug_time, n_des, n_cls = logits.shape
    eps = 0.1
    sim = logits / logit_scale.exp()
    sim = sim.permute(0, 3, 1, 2).reshape(bs*n_cls, aug_time, n_des) # (bs*n_cls) x aug_time x n_des

    wdist = 1.0 - sim
    with torch.no_grad():
        KK = torch.exp(-wdist / eps)
        T = Sinkhorn(KK, image_weights, text_weights)
        T = T.reshape(bs, n_cls, aug_time, n_des).permute(0, 2, 3, 1)
    assert not torch.isnan(T).any()

    wass_dist = torch.sum(T * logits, dim=(1, 2))

    return wass_dist

def Wasserstein_Distance(logits, logit_scale, is_train):
    # logits: bs x aug_time x n_des x n_cls
    # calculate wasserstein distance for every batch
    logits = logits.float()
    if is_train:
        img_t, text_t=1.0, 1.0
    else:
        img_t, text_t=0.5, 0.5
        
    image_weights, text_weights = get_entropy_weight(logits, img_t, text_t)
    wass_dist = optimal_transport(logits, logit_scale.float(), image_weights, text_weights)

    return wass_dist