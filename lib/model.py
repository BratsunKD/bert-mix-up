import torch
import torch.nn as nn

class BertMixUp(nn.Module):
    def __init__(self, pretrained_model, num_classes=2, mixup_type='none'):
        super().__init__()
        self.backbone = pretrained_model
        self.linear = nn.Linear(pretrained_model.config.hidden_size, num_classes)
        self.mixup_type = mixup_type
        
    def forward(self, x, mask, lam=None, x2=None, mask2=None):
        if self.mixup_type == 'embedding' and x2 is not None and mask2 is not None and lam is not None:
            return self.forward_with_embedding_mixup(x, x2, mask, mask2, lam)
        elif self.mixup_type == 'sentences' and x2 is not None and mask2 is not None and lam is not None:
            return self.forward_with_sentence_mixup(x, x2, mask, mask2, lam)
        else:
            outputs = self.backbone(input_ids=x, attention_mask=mask)
            cls_output = outputs.last_hidden_state[:,0,:] 
            logits = self.linear(cls_output)
            return logits

    def forward_with_embedding_mixup(self, x1, x2, mask1, mask2, lam):
        embeddings1 = self.backbone.embeddings(input_ids=x1)
        embeddings2 = self.backbone.embeddings(input_ids=x2)
        mixed_embeddings = lam*embeddings1+(1-lam)*embeddings2
        mask = mask2[:] | mask1[:]
        outputs = self.backbone(inputs_embeds=mixed_embeddings, attention_mask=mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.linear(cls_output)
        return logits

    def forward_with_sentence_mixup(self, x1, x2, mask1, mask2, lam):
        outputs1 = self.backbone(input_ids=x1, attention_mask=mask1)
        outputs2 = self.backbone(input_ids=x2, attention_mask=mask2)
        cls_output1 = outputs1.last_hidden_state[:,0,:]
        cls_output2 = outputs2.last_hidden_state[:,0,:]
        mixed_cls_output = lam*cls_output1 + (1-lam)*cls_output2
        logits = self.linear(mixed_cls_output)
        return logits
