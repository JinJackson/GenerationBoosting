from transformers import BertModel, AlbertModel, RobertaModel, BertPreTrainedModel, AlbertPreTrainedModel
from transformers import AutoModel, PreTrainedModel, AutoModelForPreTraining, AutoModelForQuestionAnswering
import torch.nn as nn

# BERT
class BertMatchModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_attentions=True)
        pooler_output = outputs[1]
        attention_matrix = outputs[-1]
        logits = self.linear(self.dropout(pooler_output))
        # logits = self.linear(pooler_output)
        if labels is None:
            return logits, attention_matrix
        loss = self.loss_func(logits, labels.float())
        return loss, logits, attention_matrix

# ERNIE
class ErnieMatchModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooler_output = outputs[1]
        logits = self.linear(self.dropout(pooler_output))
        # logits = self.linear(pooler_output)
        if labels is None:
            return logits
        loss = self.loss_func(logits, labels.float())
        return loss, logits

# Roberta
class RobertaMatchModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooler_output = outputs[1]
        logits = self.linear(self.dropout(pooler_output))
        if labels is None:
            return logits
        loss = self.loss_func(logits, labels.float())
        return loss, logits



class AlbertMatchModel(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        outputs = self.albert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooler_output = outputs[1]
        logits = self.linear(self.dropout(pooler_output))
        if labels is None:
            return logits
        loss = self.loss_func(logits, labels.float())
        return loss, logits
