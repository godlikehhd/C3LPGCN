
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform



class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        hidden_state, prediction_scores = self.predictions(sequence_output)
        return prediction_scores, hidden_state


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states_trans = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states_trans)
        return hidden_states_trans, hidden_states


class CLPTGCN(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.bert = bert
        self.opt = opt
        self.layers = opt.num_layers
        self.mem_dim = opt.bert_dim // 2
        self.attention_heads = opt.attention_heads
        self.bert_dim = opt.bert_dim
        self.bert_drop = nn.Dropout(opt.bert_dropout)
        self.pooled_drop = nn.Dropout(opt.bert_dropout)
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)
        self.lstm_drop = nn.Dropout(opt.input_dropout)
        self.layernorm = LayerNorm(opt.bert_dim)
        if self.opt.dataset != 'restaurant':
            self.lm_head = BertOnlyMLMHead(config=opt.config)
        self.lstm = nn.LSTM(opt.bert_dim, self.mem_dim, opt.rnn_layers, batch_first=True,
                            dropout=opt.rnn_dropout, bidirectional=opt.bidirect)

        self.W = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.bert_dim
            self.W.append(nn.Linear(input_dim, self.bert_dim))

        self.dense_sent = nn.Linear(self.bert_dim, self.bert_dim)
        self.dense_aspect = nn.Linear(self.bert_dim, self.bert_dim)
        self.dense_mask = nn.Linear(self.bert_dim, self.bert_dim)

        self.sim = Similarity(temp=self.opt.temp)
        self.cls = nn.Linear(opt.bert_dim * 3, opt.polarities_dim)

    def forward(self, inputs):
        tids_mask, tids_pos, tids_neg_1, tids_neg_2, bert_segments_ids_aul, attention_mask_aul, \
        adj_dep, src_mask, aspect_mask, label_id, loc_mask = inputs

        output = self.bert(tids_mask, attention_mask=attention_mask_aul,
                           token_type_ids=bert_segments_ids_aul, return_dict=False)

        out_pos = self.bert(tids_pos, attention_mask=attention_mask_aul,
                            token_type_ids=bert_segments_ids_aul, return_dict=False)

        out_neg_1 = self.bert(tids_neg_1, attention_mask=attention_mask_aul,
                              token_type_ids=bert_segments_ids_aul, return_dict=False)

        out_neg_2 = self.bert(tids_neg_2, attention_mask=attention_mask_aul,
                              token_type_ids=bert_segments_ids_aul, return_dict=False)

        sequence_output_mask = output[0]
        sequence_output_pos = out_pos[0]
        if self.opt.dataset != 'restaurant':
            scores, hidden_states = self.lm_head(sequence_output_mask)
        else:
            scores = None

        loc_mask_bert_dim = loc_mask.unsqueeze(-1).repeat(1, 1, self.opt.bert_dim)
        mask_state_ori = (sequence_output_mask * loc_mask_bert_dim).sum(dim=1)
        pos_state_ori = (sequence_output_pos * loc_mask_bert_dim).sum(dim=1)

        pooled_output = self.pooled_drop(output[1])
        pooled_output_pos = self.pooled_drop(out_pos[1])
        pooled_output_neg_1 = self.pooled_drop(out_neg_1[1])
        pooled_output_neg_2 = self.pooled_drop(out_neg_2[1])

        cos_sim = self.sim(pooled_output.unsqueeze(1), pooled_output_pos.unsqueeze(0))
        sample_mask = torch.eye(cos_sim.size(0)).cuda()
        cos_sim = (cos_sim * sample_mask).sum(dim=1).unsqueeze(-1)
        cos_sim_n1 = self.sim(pooled_output.unsqueeze(1), pooled_output_neg_1.unsqueeze(0))
        cos_sim_n2 = self.sim(pooled_output.unsqueeze(1), pooled_output_neg_2.unsqueeze(0))
        cos_sim = torch.cat([cos_sim, cos_sim_n1, cos_sim_n2], dim=-1)
        # labels = torch.arange(cos_sim.size(0)).long().cuda()
        labels = torch.zeros(cos_sim.size(0)).long().cuda()
        loss_fct = nn.CrossEntropyLoss()

        loss_cl = loss_fct(cos_sim, labels)

        sequence_output_mask = self.bert_drop(sequence_output_mask)
        sequence_output_pos = self.bert_drop(sequence_output_pos)
        src_mask_cl = src_mask.unsqueeze(-1).repeat(1, 1, self.opt.bert_dim)
        h_s_mask = sequence_output_mask * src_mask_cl
        h_s_pos = sequence_output_pos * src_mask_cl

        h_s_mask_log_softmax = torch.log_softmax(h_s_mask, dim=-1)
        label_h_s_pos = torch.softmax(h_s_pos, dim=-1)
        kl_fct = torch.nn.KLDivLoss(reduction='batchmean')
        loss_kl = kl_fct(h_s_mask_log_softmax, label_h_s_pos)

        lstm_input_mask, _ = self.lstm(h_s_mask, None)
        lstm_input_pos, _ = self.lstm(h_s_pos, None)

        outputs_dep_mask = self.lstm_drop(lstm_input_mask)
        outputs_dep_pos = self.lstm_drop(lstm_input_pos)

        # outputs_dep_mask = h_s_mask
        # outputs_dep_pos = h_s_pos
        denom_dep = adj_dep.sum(2).unsqueeze(2) + 1
        for l in range(self.layers):
            # ************SynGCN_mask*************
            Ax_dep = adj_dep.bmm(outputs_dep_mask)
            AxW_dep = self.W[l](Ax_dep)
            AxW_dep = AxW_dep / denom_dep
            gAxW_dep = F.relu(AxW_dep)
            outputs_dep_mask = self.gcn_drop(gAxW_dep) if l < self.layers - 1 else gAxW_dep

            # ************SynGCN_pos*************
            Ax_dep_pos = adj_dep.bmm(outputs_dep_pos)
            AxW_dep_pos = self.W[l](Ax_dep_pos)
            AxW_dep_pos = AxW_dep_pos / denom_dep
            gAxW_dep_pos = F.relu(AxW_dep_pos)
            outputs_dep_pos = self.gcn_drop(gAxW_dep_pos) if l < self.layers - 1 else gAxW_dep_pos
        asp_wn = aspect_mask.sum(dim=1).unsqueeze(-1)
        aspect_mask = aspect_mask.unsqueeze(-1).repeat(1, 1, self.opt.bert_dim)
        outputs_asp_dep_mask = (outputs_dep_mask * aspect_mask).sum(dim=1) / asp_wn
        outputs_asp_dep_pos = (outputs_dep_pos * aspect_mask).sum(dim=1) / asp_wn

        outputs_dep_mask_d = self.dense_sent(outputs_dep_mask)

        outputs_asp_dep = self.dense_aspect(outputs_asp_dep_mask)
        mask_state = self.dense_mask(mask_state_ori)
        temp_rep = (outputs_asp_dep + mask_state).unsqueeze(1)
        outputs_dep_t = torch.transpose(outputs_dep_mask_d, 1, 2)
        weight = torch.bmm(temp_rep, outputs_dep_t).squeeze(1)
        weight = weight.masked_fill(src_mask == 0, -1e9)
        target_weight = torch.softmax(weight, dim=-1).unsqueeze(1)
        target_representation = torch.bmm(target_weight, outputs_dep_mask_d).squeeze(1)

        final_rep = torch.cat((target_representation, pooled_output, mask_state_ori), dim=-1)

        outputs_dep_pos_d = self.dense_sent(outputs_dep_pos)

        outputs_asp_dep_pos = self.dense_aspect(outputs_asp_dep_pos)
        pos_state = self.dense_mask(pos_state_ori)
        temp_rep_pos = (outputs_asp_dep_pos + pos_state).unsqueeze(1)
        outputs_dep_t_pos = torch.transpose(outputs_dep_pos_d, 1, 2)
        weight_pos = torch.bmm(temp_rep_pos, outputs_dep_t_pos).squeeze(1)
        weight_pos = weight_pos.masked_fill(src_mask == 0, -1e9)
        target_weight_pos = torch.softmax(weight_pos, dim=-1).unsqueeze(1)
        target_representation_pos = torch.bmm(target_weight_pos, outputs_dep_pos_d).squeeze(1)

        final_rep_pos = torch.cat((target_representation_pos, pooled_output_pos, pos_state_ori), dim=-1)

        logits = self.cls(final_rep)
        logits_pos = self.cls(final_rep_pos)
        labels_pos = torch.argmax(logits_pos, dim=-1)
        loss_dl = loss_fct(logits, labels_pos)

        return logits, logits_pos, scores, label_id, loss_kl, loss_cl, loss_dl, pooled_output










