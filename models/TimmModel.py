import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models import load_checkpoint
from timm.models.layers import SelectAdaptivePool2d

class TimmModel(nn.Module):
    def __init__(self, n_classes, in_chans, model_name):
        super(TimmModel, self).__init__()
        self.model_name = model_name
        self.net = timm.create_model(model_name, pretrained=True, in_chans=in_chans, num_classes=n_classes)
        if model_name.startswith('resnet'):
            self.global_pool = SelectAdaptivePool2d(pool_type='max', flatten=True)
        

    def forward(self, x):
        outputs = self.net(x)   # [batch_size, n_classes]
        Y_hat = torch.argmax(outputs, dim=1)
        Y_prob = F.softmax(outputs, dim=1)
        results_dict = {'logits': outputs, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return results_dict

    def forward_features(self, x):
        features = self.net.forward_features(x)
        if self.model_name.startswith('resnet'):
            features = self.global_pool(features)
        return features

    def forward_imgs(self, imgs, nviews=None, block_num = 150):
        B, N, H, W = imgs.shape
        if nviews is not None:
            assert N == nviews
        else:
            assert B == 1
        imgs_sequence = imgs.view(B*N, 1, H, W)
        features = []; pred_prob = []; pred_logits = []
        for i in range(B*N//block_num):
            features.append(self.forward_features(imgs_sequence[i*block_num:(i+1)*block_num]))
            output = self.forward(imgs_sequence[i*block_num:(i+1)*block_num])
            pred_logits.append(output['logits'])
            pred_prob.append(output['Y_prob'])
        if (B*N % block_num != 0):
            i = B*N // block_num
            features.append(self.forward_features(imgs_sequence[i*block_num:]))
            output = self.forward(imgs_sequence[i*block_num:])
            pred_logits.append(output['logits'])
            pred_prob.append(output['Y_prob'])
        features = torch.cat(features, dim=0).view(B, N, -1)
        pred_prob = torch.cat(pred_prob, dim=0).view(B, N, -1)
        pred_logits = torch.cat(pred_logits, dim=0).view(B, N, -1)

        final_index = torch.argmax(pred_prob[:, :, 1], dim=-1)
        final_logits, final_prob = [], []
        for b in range(B):
            final_logits.append(pred_logits[b, final_index[b].item(), :].unsqueeze(0))
            final_prob.append(pred_prob[b, final_index[b].item(), :].unsqueeze(0))
        final_logits = torch.cat(final_logits, dim=0)
        final_prob = torch.cat(final_prob, dim=0)
        # final_logits = pred_logits[final_index, :]
        # final_prob = pred_prob[:, final_index, :]
        results_dict = {'logits': final_logits, 'Y_prob': final_prob, 'Y_hat': torch.argmax(final_prob, dim=-1),
                        'images_prob': pred_prob}
        return results_dict