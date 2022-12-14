import abc
import numpy as np
from padertorch import Model
import torch
from torch import nn
from padertorch.contrib.je.modules.hybrid import CNN
from padertorch.contrib.je.modules.rnn import GRU

class LinearNorm(nn.Module):
     def __init__(self, in_features, out_features, bias=False):
          super(LinearNorm, self).__init__()
          self.linear = nn.Linear(in_features, out_features, bias)

          nn.init.xavier_uniform_(self.linear.weight)
          if bias:
               nn.init.constant_(self.linear.bias, 0.0)
     
     def forward(self, x):
          x = self.linear(x)
          return x

class Swish(nn.Module):
     def __init__(self):
          super(Swish, self).__init__()
     
     def forward(self, inputs):
          return inputs * inputs.sigmoid()

class FeedForwardModule(nn.Module):
     def __init__(
               self,
               encoder_dim: int = 512,
               expansion_factor: int = 4,
               dropout_p: float = 0.1,
     ) -> None:
          super(FeedForwardModule, self).__init__()
          self.sequential = nn.Sequential(
               nn.LayerNorm(encoder_dim),
               LinearNorm(encoder_dim, encoder_dim * expansion_factor, bias=True),
               Swish(),
               nn.Dropout(p=dropout_p),
               LinearNorm(encoder_dim * expansion_factor, encoder_dim, bias=True),
               nn.Dropout(p=dropout_p),
          )

     def forward(self, inputs):
          return self.sequential(inputs)
          
class ResidualConnectionModule(nn.Module):

     def __init__(self, module, module_factor = 1.0, input_factor = 1.0):
          super(ResidualConnectionModule, self).__init__()
          self.module = module
          self.module_factor = module_factor
          self.input_factor = input_factor

     def forward(self, inputs):
          return (self.module(inputs) * self.module_factor) + (inputs * self.input_factor)

class ConformerConvModule(nn.Module):
     def __init__(
               self,
               in_channels: int,
               kernel_size: int = 31,
               expansion_factor: int = 2,
               dropout_p: float = 0.1,
     ) -> None:
          super(ConformerConvModule, self).__init__()
          assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
          assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

          self.sequential = nn.Sequential(
               nn.LayerNorm(in_channels),
               Transpose(shape=(1, 2)),
               PointwiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True),
               GLU(dim=1),
               DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
               nn.BatchNorm1d(in_channels),
               Swish(),
               PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True),
               nn.Dropout(p=dropout_p),
          )

     def forward(self, inputs: Tensor) -> Tensor:
          return self.sequential(inputs).transpose(1, 2)

class MultiHeadedSelfAttentionModule(nn.Module):
     def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.1, position_enc: Optional[Tensor] = None, max_seq_len: int = 10000):
          super(MultiHeadedSelfAttentionModule, self).__init__()
          self.d_model = d_model
          self.max_seq_len = max_seq_len
          self.positional_encoding = position_enc
          self.layer_norm = nn.LayerNorm(d_model)
          self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p)
          self.dropout = nn.Dropout(p=dropout_p)

     def forward(self, inputs: Tensor, mask: Optional[Tensor] = None):
          batch_size, seq_length, _ = inputs.size()

          # -- Forward
          if not self.training and seq_length > self.max_seq_len:
               pos_embedding = get_sinusoid_encoding_table(
                    seq_length, self.d_model
               )[: seq_length, :].unsqueeze(0).expand(batch_size, -1, -1).to(
                    inputs.device
               )
          else:
               pos_embedding = self.positional_encoding[
                    :, :seq_length, :
               ].expand(batch_size, -1, -1)

          inputs = self.layer_norm(inputs)
          outputs = self.attention(inputs, inputs, inputs, pos_embedding=pos_embedding, mask=mask)

          return self.dropout(outputs)
          
class ConformerBlock(torch.nn.Module):
     def __init__(
               self,
               encoder_dim: int = 512,
               num_attention_heads: int = 8,
               feed_forward_expansion_factor: int = 4,
               conv_expansion_factor: int = 2,
               feed_forward_dropout_p: float = 0.1,
               attention_dropout_p: float = 0.1,
               conv_dropout_p: float = 0.1,
               conv_kernel_size: int = 31,
               half_step_residual: bool = True,
               position_enc= None,
               max_seq_len: int = 10000,
     ):
          super(ConformerBlock, self).__init__()
          if half_step_residual:
               self.feed_forward_residual_factor = 0.5
          else:
               self.feed_forward_residual_factor = 1

          self.sequential = nn.Sequential(
               ResidualConnectionModule(
                    module=FeedForwardModule(
                         encoder_dim=encoder_dim,
                         expansion_factor=feed_forward_expansion_factor,
                         dropout_p=feed_forward_dropout_p,
                    ),
                    module_factor=self.feed_forward_residual_factor,
               ),
               ResidualConnectionModule(
                    module=MultiHeadedSelfAttentionModule(
                         d_model=encoder_dim,
                         num_heads=num_attention_heads,
                         dropout_p=attention_dropout_p,
                         position_enc=position_enc,
                         max_seq_len=max_seq_len,
                    ),
               ),
               ResidualConnectionModule(
                    module=ConformerConvModule(
                         in_channels=encoder_dim,
                         kernel_size=conv_kernel_size,
                         expansion_factor=conv_expansion_factor,
                         dropout_p=conv_dropout_p,
                    ),
               ),
               ResidualConnectionModule(
                    module=FeedForwardModule(
                         encoder_dim=encoder_dim,
                         expansion_factor=feed_forward_expansion_factor,
                         dropout_p=feed_forward_dropout_p,
                    ),
                    module_factor=self.feed_forward_residual_factor,
               ),
               torch.nn.LayerNorm(encoder_dim),
          )

     def forward(self, inputs, mask):
          output = self.sequential(inputs)
          if mask is not None:
               output = output.masked_fill(mask.unsqueeze(-1), 0)
          return output

class CRNN(Model, abc.ABC):
     
     def __init__(
               self, feature_extractor, cnn, rnn_fwd, rnn_bwd, *,
               minimum_score=1e-5, label_smoothing=0.,
               labelwise_metrics=(), label_mapping=None, test_labels=None,
               slat=False, strong_fwd_bwd_loss_weight=1., class_weights=None,
     ):
          super().__init__(
               labelwise_metrics=labelwise_metrics,
               label_mapping=label_mapping,
               test_labels=test_labels,
          )
          self.feature_extractor = feature_extractor
          self.cnn = cnn
          self.rnn_fwd = rnn_fwd
          self.rnn_bwd = rnn_bwd
          self.minimum_score = minimum_score
          self.label_smoothing = label_smoothing
          self.slat = slat
          self.strong_fwd_bwd_loss_weight = strong_fwd_bwd_loss_weight
          self.class_weights = None if class_weights is None else torch.Tensor(class_weights)

     def sigmoid(self, y):
          return self.minimum_score + (1-2*self.minimum_score) * nn.Sigmoid()(y)

     def fwd_tagging(self, h, seq_len):
          y, seq_len_y = self.rnn_fwd(h, seq_len=seq_len)
          return self.sigmoid(y), seq_len_y

     def bwd_tagging(self, h, seq_len):
          y, seq_len_y = self.rnn_bwd(h, seq_len=seq_len)
          return self.sigmoid(y), seq_len_y

     def forward(self, inputs):

          if self.training:
               x = inputs.pop('stft')
          else:
               x = inputs['stft']
          seq_len = np.array(inputs['seq_len'])
          if "weak_targets" in inputs:
               targets = self.read_targets(inputs)
               x, seq_len_x, targets = self.feature_extractor(
                    x, seq_len=seq_len, targets=targets
               )
          else:
               x, seq_len_x = self.feature_extractor(x, seq_len=seq_len)
               targets = None

          h, seq_len_h = self.cnn(x, seq_len_x)
          y_fwd, seq_len_y = self.fwd_tagging(h, seq_len_h)
          if self.rnn_bwd is None:
               y_bwd = None
          else:
               y_bwd, seq_len_y_ = self.bwd_tagging(h, seq_len_h)
               assert (seq_len_y_ == seq_len_y).all()
          return y_fwd, y_bwd, seq_len_y, x, seq_len_x, targets
     
     

if __name__=="__main__":
     a=0
     config = CRNN.get_config({\
               'cnn': {\
                    'factory': CNN,\
                    'cnn_2d': {'out_channels':[32,32,32], 'kernel_size': 3},\
                    'cnn_1d': {'out_channels':[32,32], 'kernel_size': 3},\
               },\
               'rnn_fwd': {'factory': GRU, 'hidden_size': 64, 'output_net': {'out_channels':[32,10], 'kernel_size': 1}},\
               'feature_extractor': {\
                    'sample_rate': 16000,\
                    'stft_size': 512,\
                    'number_of_filters': 80,\
               },\
          })
     crnn = CRNN.from_config(config)
     inputs = {'stft': torch.randn((4, 1, 15, 257, 2)), 'seq_len': [15, 14, 13, 12], 'weak_targets': torch.zeros((4,10)), 'boundary_targets': torch.zeros((4,10,15))}
     np.random.seed(3)
     outputs = crnn({**inputs})
     outputs[0].shape
     torch.Size([4, 10, 15])
     # review = crnn.review(inputs, outputs)
     print(outputs)
