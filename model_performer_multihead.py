from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal

from IPython import embed

from functools import partial
from einops import rearrange, repeat

import numpy as np




def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4, device = None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    ratio = (projection_matrix.shape[0] ** -0.5)

    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data -
                    torch.max(data_dash, dim=-1, keepdim=True).values) + eps)
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(data_dash)) + eps)

    return data_dash.type_as(data)



def orthogonal_matrix_chunk(cols, qr_uniform_q = False, device = None):
    unstructured_block = torch.randn((cols, cols), device = device)
    q, r = torch.qr(unstructured_block.cpu(), some = True)
    q, r = map(lambda t: t.to(device), (q, r))

    # proposed by @Parskatt
    # to make sure Q is uniform https://arxiv.org/pdf/math-ph/0609050.pdf
    if qr_uniform_q:
        d = torch.diag(r, 0)
        q *= d.sign()
    return q.t()

def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling = 0, qr_uniform_q = False, device = None):
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, qr_uniform_q = qr_uniform_q, device = device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, qr_uniform_q = qr_uniform_q, device = device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device = device).norm(dim = 1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device = device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix



class SubNet(nn.Module):
    '''
    The subnetwork that is used in LMF for video and audio in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, dropout):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(SubNet, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        normed = self.norm(x)
        dropped = self.drop(normed)
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.relu(self.linear_2(y_1))
        y_3 = F.relu(self.linear_3(y_2))

        return y_3
    
class _SubNet(nn.Module):
    '''
    The subnetwork that is used in LMF for video and audio in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, dropout):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(_SubNet, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_len, in_size)
        '''
        #normed = self.norm(x)
        dropped = self.drop(x)
        
        
        
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.relu(self.linear_2(y_1))
        y_3 = F.relu(self.linear_3(y_2))

        return y_3

    
    


class TextSubNet(nn.Module):
    '''
    The LSTM-based subnetwork that is used in LMF for text
    '''

    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(TextSubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_len, in_size)
        '''
        hidden_out, final_states = self.rnn(x)
        #h = self.dropout(final_states[0].squeeze())
        h = self.dropout(hidden_out)
        y_1 = self.linear_1(h)
        return y_1


class LMF(nn.Module):
    '''
    Low-rank Multimodal Fusion
    '''

    def __init__(self, input_dims, input_lens, hidden_dims, text_out, dropouts, output_dim, rank1, rank2, use_softmax=False):
        '''
        Args:
            input_dims - a length-3 tuple, contains (audio_dim, video_dim, text_dim)
            hidden_dims - another length-3 tuple, hidden dims of the sub-networks
            text_out - int, specifying the resulting dimensions of the text subnetwork
            dropouts - a length-4 tuple, contains (audio_dropout, video_dropout, text_dropout, post_fusion_dropout)
            output_dim - int, specifying the size of output
            rank - int, specifying the size of rank in LMF
        Output:
            (return value in forward) a scalar value between -3 and 3
        '''
        super(LMF, self).__init__()

        # dimensions are specified in the order of audio, video and text
        self.audio_in = input_dims[0]
        self.video_in = input_dims[1]
        self.text_in = input_dims[2]
        
        
        self.audio_len = input_lens[0]
        self.video_len = input_lens[1]
        self.text_len = input_lens[2]

        self.audio_hidden = hidden_dims[0]
        self.video_hidden = hidden_dims[1]
        self.text_hidden = hidden_dims[2]
        self.text_out = text_out
        self.output_dim = output_dim
        #self.rank1 = rank1
        #self.rank2 = rank2
        self.use_softmax = use_softmax
        
        
        self.middle1 = rank1
        self.middle2 = rank2

        self.audio_prob = dropouts[0]
        self.video_prob = dropouts[1]
        self.text_prob = dropouts[2]
        self.post_fusion_prob = dropouts[3]
        
        self.num_head = 4

        # define the pre-fusion subnetworks
        self.audio_subnet = _SubNet(self.audio_in, self.audio_hidden, self.audio_prob)
        self.video_subnet = _SubNet(self.video_in, self.video_hidden, self.video_prob)
        self.text_subnet = TextSubNet(self.text_in, self.text_hidden, self.text_out, dropout=self.text_prob)

        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
       
        self.a_fm1 = Parameter(torch.Tensor(self.audio_hidden + 1, self.num_head * self.middle1).cuda())
        self.v_fm1 = Parameter(torch.Tensor(self.video_hidden + 1, self.num_head * self.middle1).cuda())
        self.t_fm1 = Parameter(torch.Tensor(self.text_out + 1, self.num_head * self.middle1).cuda())
        
        self.a_fm2 = Parameter(torch.Tensor(self.audio_hidden + 1, self.num_head * self.middle2).cuda())
        self.v_fm2 = Parameter(torch.Tensor(self.video_hidden + 1, self.num_head * self.middle2).cuda())
        self.t_fm2 = Parameter(torch.Tensor(self.text_out + 1, self.num_head * self.middle2).cuda())
        
        
        self.out_weights = Parameter(torch.Tensor(self.num_head * self.middle1, self.output_dim).cuda())
        
        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows = middle2 * 2, nb_columns = middle2, scaling = 0, qr_uniform_q = False, device= self.a_fm1.device)
        projection_matrix = self.create_projection()
        
        self.register_buffer('projection_matrix', projection_matrix)
        
        
        xavier_normal(self.a_fm1)
        xavier_normal(self.v_fm1)
        xavier_normal(self.t_fm1)
        
        xavier_normal(self.a_fm2)
        xavier_normal(self.v_fm2)
        xavier_normal(self.t_fm2)
        
        xavier_normal(self.out_weights)
        
    
    
    def forward(self, audio_x, video_x, text_x):
        '''
        Args:
            audio_x: tensor of shape (batch_size, audio_in)
            video_x: tensor of shape (batch_size, video_in)
            text_x: tensor of shape (batch_size, sequence_len, text_in)
            
        Args:
            audio_x: tensor of shape (batch_size, audio_len, audio_in)
            video_x: tensor of shape (batch_size, video_len, video_in)
            text_x: tensor of shape (batch_size, sequence_len, text_in)
            
       The shape of audio_h (batch_size, audio_len, hidden_size)
       The shape of video_h (batch_size, video_len, hidden_size)
       The shape of text_h  (batch_size, text_len, hidden_size)                       
        '''
        
        device = audio_x.device
        
        
        audio_h = self.audio_subnet(audio_x)
        video_h = self.video_subnet(video_x)
        text_h = self.text_subnet(text_x)
        batch_size = audio_h.data.shape[0]
        
        
        if audio_h.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        _audio_h = torch.cat((Variable(torch.ones(batch_size, self.audio_len, 1).type(DTYPE), requires_grad=False), audio_h), dim=2)
        _video_h = torch.cat((Variable(torch.ones(batch_size, self.video_len, 1).type(DTYPE), requires_grad=False), video_h), dim=2)
        _text_h = torch.cat((Variable(torch.ones(batch_size, self.text_len, 1).type(DTYPE), requires_grad=False), text_h), dim=2)
        
        
        ### embedding
        fusion_audio = torch.matmul(_audio_h, self.a_fm1)     #[batch_size, audio_len, h * middle1]
        fusion_video = torch.matmul(_video_h, self.v_fm1)     #[batch_size, video_len, h * middle1]
        fusion_text = torch.matmul(_text_h, self.t_fm1)       #[batch_size, text_len, h * middle1]
        
        
        
        batch_size, lenth, _ = fusion_audio.size()
        
        fusion_audio = fusion_audio.view(batch_size, lenth, self.num_head, -1)
        fusion_audio = fusion_audio.transpose(1, 2)
        
        fusion_video = fusion_video.view(batch_size, lenth, self.num_head, -1)
        fusion_video = fusion_video.transpose(1, 2)
        
        fusion_text = fusion_text.view(batch_size, lenth, self.num_head, -1)
        fusion_text = fusion_text.transpose(1, 2)
        
        
        
        ### attention weights
        temp_audio = torch.matmul(_audio_h, self.a_fm2)     #[batch_size, audio_len, h * middle2]   
        temp_video = torch.matmul(_video_h, self.v_fm2)     #[batch_size, video_len, h * middle2]
        temp_text = torch.matmul(_text_h, self.t_fm2)      #[batch_size, text_len, h * middle2] 
        
        temp_audio = temp_audio.view(batch_size, lenth, self.num_head, -1)
        temp_audio = temp_audio.transpose(1, 2)
        
        temp_video = temp_video.view(batch_size, lenth, self.num_head, -1)
        temp_video = temp_video.transpose(1, 2)
        
        temp_text = temp_text.view(batch_size, lenth, self.num_head, -1)
        temp_text = temp_text.transpose(1, 2)
        
        
        #temp_audio = temp_audio.unsqueeze(1)
        #temp_video = temp_video.unsqueeze(1)
        #temp_text = temp_text.unsqueeze(1)
        
        
        
        create_kernel = partial(softmax_kernel, projection_matrix = self.projection_matrix, device = device)
        temp_audio = create_kernel(temp_audio, is_query = True)
        temp_video = create_kernel(temp_video, is_query = True)
        temp_text = create_kernel(temp_text, is_query = True)
        
        
        domi = torch.sum(temp_audio, 2) * torch.sum(temp_video, 2)*torch.sum(temp_text, 2)
        domi = torch.sum(domi, 2)
        domi = domi.view(-1)
 
        
        temp_audio = temp_audio.view(batch_size * self.num_head, lenth, -1)
        temp_video = temp_video.view(batch_size * self.num_head, lenth, -1)
        temp_text = temp_text.view(batch_size * self.num_head, lenth, -1)
        
        
        fusion_audio = fusion_audio.reshape(batch_size * self.num_head, lenth, -1)
        fusion_video = fusion_video.reshape(batch_size * self.num_head, lenth, -1)
        fusion_text = fusion_text.reshape(batch_size * self.num_head, lenth, -1)
        
        
        audio_rep = torch.matmul(temp_audio.transpose(1, 2), fusion_audio)
        video_rep = torch.matmul(temp_video.transpose(1, 2), fusion_video)
        text_rep = torch.matmul(temp_text.transpose(1, 2), fusion_text)
        
        
        
        final_rep = audio_rep * video_rep * text_rep
        
        final_rep = torch.sum(final_rep, 1)
        
        
        final_rep = final_rep / (domi.unsqueeze(1))
        
        final_rep = final_rep.view(batch_size, self.num_head, -1)
        final_rep = final_rep.view(batch_size, -1)
        
        
        output = torch.matmul(final_rep, self.out_weights)
        
        
        
        
        
        if self.use_softmax:
            output = F.softmax(output)
        return output

    
    
    
    def forward_lsc(self, audio_x, video_x, text_x, temp_factor):
        '''
        Args:
            audio_x: tensor of shape (batch_size, audio_in)
            video_x: tensor of shape (batch_size, video_in)
            text_x: tensor of shape (batch_size, sequence_len, text_in)
            
        Args:
            audio_x: tensor of shape (batch_size, audio_len, audio_in)
            video_x: tensor of shape (batch_size, video_len, video_in)
            text_x: tensor of shape (batch_size, sequence_len, text_in)
            
       The shape of audio_h (batch_size, audio_len, hidden_size)
       The shape of video_h (batch_size, video_len, hidden_size)
       The shape of text_h  (batch_size, text_len, hidden_size)                       
        '''
        
        
        device = audio_x.device
        
        
        audio_h = self.audio_subnet(audio_x)
        video_h = self.video_subnet(video_x)
        text_h = self.text_subnet(text_x)
        batch_size = audio_h.data.shape[0]
        
        
        if audio_h.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        _audio_h = torch.cat((Variable(torch.ones(batch_size, self.audio_len, 1).type(DTYPE), requires_grad=False), audio_h), dim=2)
        _video_h = torch.cat((Variable(torch.ones(batch_size, self.video_len, 1).type(DTYPE), requires_grad=False), video_h), dim=2)
        _text_h = torch.cat((Variable(torch.ones(batch_size, self.text_len, 1).type(DTYPE), requires_grad=False), text_h), dim=2)
        
        
        ### embedding
        fusion_audio = torch.matmul(_audio_h, self.a_fm1)     #[batch_size, audio_len, h * middle1]
        fusion_video = torch.matmul(_video_h, self.v_fm1)     #[batch_size, video_len, h * middle1]
        fusion_text = torch.matmul(_text_h, self.t_fm1)       #[batch_size, text_len, h * middle1]
        
        
        #print (fusion_audio)
        
        
        batch_size, lenth, _ = fusion_audio.size()
        
        fusion_audio = fusion_audio.view(batch_size, lenth, self.num_head, -1)
        fusion_audio = fusion_audio.transpose(1, 2)
        
        fusion_video = fusion_video.view(batch_size, lenth, self.num_head, -1)
        fusion_video = fusion_video.transpose(1, 2)
        
        fusion_text = fusion_text.view(batch_size, lenth, self.num_head, -1)
        fusion_text = fusion_text.transpose(1, 2)
        
        
        
        ### attention weights
        temp_audio = torch.matmul(_audio_h, self.a_fm2)     #[batch_size, audio_len, h * middle2]   
        temp_video = torch.matmul(_video_h, self.v_fm2)     #[batch_size, video_len, h * middle2]
        temp_text = torch.matmul(_text_h, self.t_fm2)      #[batch_size, text_len, h * middle2] 
        
        temp_audio = temp_audio.view(batch_size, lenth, self.num_head, -1)
        temp_audio = temp_audio.transpose(1, 2)
        
        temp_video = temp_video.view(batch_size, lenth, self.num_head, -1)
        temp_video = temp_video.transpose(1, 2)
        
        temp_text = temp_text.view(batch_size, lenth, self.num_head, -1)
        temp_text = temp_text.transpose(1, 2)
        
        
        temp_factor = temp_factor.unsqueeze(0)
        temp_factor = temp_factor.unsqueeze(2)
        
        temp_factor = temp_factor.repeat(batch_size, 1, self.num_head, 1)
        temp_audio = torch.cat((temp_audio, temp_factor), -1)
        temp_video = torch.cat((temp_video, temp_factor), -1)
        temp_text = torch.cat((temp_text, temp_factor),-1)
        
        
        
        create_kernel = partial(softmax_kernel, projection_matrix = self.projection_matrix, device = device)
        temp_audio = create_kernel(temp_audio, is_query = True)
        temp_video = create_kernel(temp_video, is_query = True)
        temp_text = create_kernel(temp_text, is_query = True)
        
        
        domi = torch.sum(temp_audio, 2) * torch.sum(temp_video, 2)*torch.sum(temp_text, 2)
        domi = torch.sum(domi, 2)
        domi = domi.view(-1)
        
        
        
        temp_audio = temp_audio.view(batch_size * self.num_head, lenth, -1)
        temp_video = temp_video.view(batch_size * self.num_head, lenth, -1)
        temp_text = temp_text.view(batch_size * self.num_head, lenth, -1)
        
        
        fusion_audio = fusion_audio.reshape(batch_size * self.num_head, lenth, -1)
        fusion_video = fusion_video.reshape(batch_size * self.num_head, lenth, -1)
        fusion_text = fusion_text.reshape(batch_size * self.num_head, lenth, -1)
        
        
        audio_rep = torch.matmul(temp_audio.transpose(1, 2), fusion_audio)
        video_rep = torch.matmul(temp_video.transpose(1, 2), fusion_video)
        text_rep = torch.matmul(temp_text.transpose(1, 2), fusion_text)
        
        
        #audio_rep = torch.sum(audio_rep, 1)
        #video_rep = torch.sum(video_rep, 1)
        #text_rep = torch.sum(text_rep, 1)
        
        final_rep = audio_rep * video_rep * text_rep
        final_rep = torch.sum(final_rep, 1)
        
        final_rep = final_rep / (domi.unsqueeze(1))
        
        final_rep = final_rep.view(batch_size, self.num_head, -1)
        final_rep = final_rep.view(batch_size, -1)
        
        
        output = torch.matmul(final_rep, self.out_weights)
        
        
        
        
        
        if self.use_softmax:
            output = F.softmax(output)
        return output

        
        
        

    def forward_vanilla(self, audio_x, video_x, text_x):
        '''
        Args:
            audio_x: tensor of shape (batch_size, audio_in)
            video_x: tensor of shape (batch_size, video_in)
            text_x: tensor of shape (batch_size, sequence_len, text_in)
            
        Args:
            audio_x: tensor of shape (batch_size, audio_len, audio_in)
            video_x: tensor of shape (batch_size, video_len, video_in)
            text_x: tensor of shape (batch_size, sequence_len, text_in)
            
       The shape of audio_h (batch_size, audio_len, hidden_size)
       The shape of video_h (batch_size, video_len, hidden_size)
       The shape of text_h  (batch_size, text_len, hidden_size)                       
        '''
        
        
        audio_h = self.audio_subnet(audio_x)
        video_h = self.video_subnet(video_x)
        text_h = self.text_subnet(text_x)
        batch_size = audio_h.data.shape[0]
        
        
        if audio_h.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        _audio_h = torch.cat((Variable(torch.ones(batch_size, self.audio_len, 1).type(DTYPE), requires_grad=False), audio_h), dim=2)
        _video_h = torch.cat((Variable(torch.ones(batch_size, self.video_len, 1).type(DTYPE), requires_grad=False), video_h), dim=2)
        _text_h = torch.cat((Variable(torch.ones(batch_size, self.text_len, 1).type(DTYPE), requires_grad=False), text_h), dim=2)
        
        
        
        fusion_audio = torch.matmul(_audio_h, self.audio_factor)     #[batch_size, audio_len, rank * output_dims]
        fusion_video = torch.matmul(_video_h, self.video_factor)     #[batch_size, video_len, rank * output_dims]
        fusion_text = torch.matmul(_text_h, self.text_factor)       #[batch_size, text_len, rank * output_dims]
        
        audio_att = F.tanh(torch.matmul(fusion_audio, self.attention_audio1) + self.attention_audio_bias)
        
        audio_att = F.softmax((torch.matmul(audio_att, self.attention_audio2)).permute(0,2,1) , 2)
        
        fusion_audio = torch.matmul(audio_att, fusion_audio) 
        
        
        video_att = F.tanh(torch.matmul(fusion_video, self.attention_video1) + self.attention_video_bias)
        
        video_att = F.softmax((torch.matmul(video_att, self.attention_video2)).permute(0,2,1) , 2)
        
        fusion_video = torch.matmul(video_att, fusion_video)
        
        
        text_att = F.tanh(torch.matmul(fusion_text, self.attention_text1) + self.attention_text_bias)
        
        text_att = F.softmax((torch.matmul(text_att, self.attention_text2)).permute(0,2,1) , 2)
        
        fusion_text = torch.matmul(text_att, fusion_text) 
        
        
        
        fusion_audio = fusion_audio.view(batch_size, self.rank2, self.rank1, self.output_dim)
        fusion_video = fusion_video.view(batch_size, self.rank2, self.rank1, self.output_dim)
        fusion_text = fusion_text.view(batch_size, self.rank2, self.rank1, self.output_dim)
        
        
        
        
        fusion_zy = fusion_audio * fusion_video * fusion_text
        
        fusion_zy = fusion_zy.view(batch_size, -1, self.output_dim)
        
        #output = torch.mean(fusion_zy, 1)
        
        #embed()
        
        # output = torch.sum(fusion_zy, dim=0).squeeze()
        # use linear transformation instead of simple summation, more flexibility
        #output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = torch.matmul(self.fusion_weights, fusion_zy) + self.fusion_bias
        output = output.view(-1, self.output_dim)
        if self.use_softmax:
            output = F.softmax(output)
        return output
