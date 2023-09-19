# coding: utf-8

# ----------------------------------------------------------------
# Author:   Du Mingzhe (mingzhe@nus.edu.sg)
# Date:     29/10/2022
# ---------------------------------------------------------------- 

import torch
from torch import nn
from transformers import BertModel
from typing import List, Optional, Tuple, Union
from torch_geometric.nn import GATConv, Linear, HGTConv
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import AutoModelForQuestionAnswering
from transformers.models.bert.modeling_bert import BertPreTrainedModel, QuestionAnsweringModelOutput

# Pytorch Geometric provides three ways for the user to create models on heterogeneous graph data:

# 1) Automatically convert a homogenous GNN model to a heterogeneous GNN model by making use of torch_geometric.nn.to_hetero() or torch_geometric.nn.to_hetero_with_bases().

# 2) Define inidividual functions for different types using PyGs wrapper torch_geometric.nn.conv.HeteroConv for heterogeneous convolution.

# 3) Deploy existing (or write your own) heterogeneous GNN operators.

# More on https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html

class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, metadata):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return self.lin(x_dict['context'])

class GraphQA(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        
        # Hyper Parameters
        self.num_labels = config.num_labels
        self.node_types = ["token", "leaf", "constituent"]
        self.metadata = (
            # Node
            ["token", "leaf", "constituent"],
            # Edge
            [
                ('token', 'connect', 'token'),
                ('token', 'connect', 'leaf'),
                ('leaf', 'connect', 'constituent'),
            ]
        )
        self.graph_hidden_channels = 768
        self.graph_layer = 2
        self.graph_head = 2

        # BERT Backbone
        self.bert = BertModel(config, add_pooling_layer=False)
        self.qa_outputs = Linear(config.hidden_size, 2)

        # Heterogenous Graph
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in self.node_types:
            self.lin_dict[node_type] = Linear(-1, self.graph_hidden_channels)
        
        self.convs = torch.nn.ModuleList()
        for _ in range(self.graph_layer):
            conv = HGTConv(self.graph_hidden_channels, self.graph_hidden_channels, self.metadata, self.graph_head, group='sum')
            self.convs.append(conv)

        self.graph_qa_outputs = Linear(self.graph_hidden_channels, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        graph_data: Optional[dict] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Ablation
        # sequence_output = outputs[0]
        
        # Graph
        # x_dict: A dictionary holding input node features for each individual node type.
        # edge_index_dict: A dictionary holding graph connectivity information for each individual edge type.
        
        def get_embedding(node):
            if not node:
                return list()
            else:
                pooled_embedding = torch.zeros(768)
                for tid in node.tids:
                    pooled_embedding += outputs[0][tid]
                return [pooled_embedding] + [get_embedding(child) for child in node.children]
            
        # Get x_dict
        x_dict = {
            "token": outputs[0],
            # "leaf": outputs[0], # Connect token and constituent
            "constituent": torch.stack(get_embedding(graph_data["graph"])),
        }

        # Get edge_index_dict
        edge_index_dict = graph_data["edge_index"]
        
        # node_type in ["token", "leaf", "constituent"]
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()
        
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        logits = self.graph_qa_outputs(x_dict['token'])
        # logits = self.qa_outputs(sequence_output)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits   = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
