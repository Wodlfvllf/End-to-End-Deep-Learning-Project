from utils_2 import *
from dataset_2 import *

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, dim):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim 
        self.head_dim = dim//n_heads
        # Query Key Value Weight Matrices
        
        self.W_q = nn.Linear(self.dim, self.n_heads*self.head_dim, bias = False)
        self.W_k = nn.Linear(self.dim, self.n_heads*self.head_dim, bias = False)
        self.W_v = nn.Linear(self.dim, self.n_heads*self.head_dim, bias = False)
        
    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        
        # Stacked Query Key Value Vectors for each word in input sequence
        x_q = self.W_q(x)
        x_k = self.W_k(x)
        x_v = self.W_v(x)
        
        x_q = x_q.contiguous().view(batch_size, seq_len, self.n_heads, self.head_dim)  # [batch_size, seq_len, n_heads, head_dim]
        x_k = x_k.contiguous().view(batch_size, seq_len, self.n_heads, self.head_dim)  #[batch_size, seq_len, n_heads, head_dim]
        x_v = x_v.contiguous().view(batch_size, seq_len, self.n_heads, self.head_dim)  #[batch_size, seq_len, n_heads, head_dim]
        
        # [batch_size, n_heads, seq_len, head_dim]
        x_q = x_q.transpose(1,2)
        x_k = x_k.transpose(1,2)
        x_v = x_v.transpose(1,2)
        
        # [batch_size, n_heads, head_dim, seq_len]
        x_k = x_k.transpose(2,3)
        
        # [batch_size, n_heads, seq_len, seq_len]
        attention_sores = F.softmax(torch.matmul(x_q, x_k)/math.sqrt(self.head_dim), dim = -1)
        
        # [batch_size, n_heads, head_dim, seq_len]
        out = torch.matmul(attention_sores, x_v)
        
        # [batch_size, seq_len, n_heads * head_dim]
        out = out.contiguous().view(batch_size, seq_len, -1)
        
        return out

class Attention_block(nn.Module):
    def __init__(self, dim, n_heads, hidden_dim, seq_len):
        super().__init__()
        
        self.dim = dim
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.dropout = 0.2
        
        self.layer_norm = nn.LayerNorm(self.dim)
        self.attn = MultiHeadAttention(self.n_heads, self.dim)
        self.layer_norm_2 = nn.LayerNorm(self.dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.dim),
            nn.Dropout(self.dropout)
        )
        
    def forward(self, x):
        x = x + self.attn(self.layer_norm(x))
        x = x + self.ffn(self.layer_norm_2(x))
        return x

class Encoder(nn.Module):
    def __init__(self, dim, n_heads, hidden_dim, seq_len, blocks, input_dim):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.input_dim = input_dim
        
        #Embediing Layer
        self.linear = nn.Linear(in_features = self.input_dim, out_features = self.dim)
        
        self.patch_embed = PatchEmbed(125, 5, 3, self.dim)
        num_patches = self.patch_embed.num_patches
        # Multi Head Attention 
        attn_blocks = []
        for i in range(blocks):
            attn_blocks.append(Attention_block(self.dim, self.n_heads, self.hidden_dim, self.seq_len))
           
        # Combining All MultiHead self Attention as a transformer blocks.
        self.transformer = nn.Sequential(*attn_blocks)
        
        # Initialising class token to be added with embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        
        # Positional embeddings to give positional information to transformer
        self.pos_embedding = nn.Parameter(torch.randn(1, 1+self.seq_len, self.dim))
        self.init_parameters()
        
    # Initialising Parameters
    def init_parameters(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)
        
    def patchify(self, imgs):
        """
        imgs: (N, 8, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 8, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
        
    def forward(self, x):
        
        x = self.patchify(x)
        
        batch, seq_len, output_dim = x.shape
        
        # Passing through Linear Projection layer or Embedding Layer
        x = self.linear(x)
        
        #Adding Class Token
        cls_token = self.cls_token.repeat(batch, 1, 1)
        x = torch.concat([cls_token, x], dim = 1)
        
        #Adding Positional Embedding
        x = x + self.pos_embedding[:, :seq_len + 1, :]
        
        # Passing through transformer block
        x = self.transformer(x)
        
        return x



class VIT(nn.Module):
    def __init__(self, n_classes, seq_len, input_dim, Encoder):
        super(VIT, self).__init__()
        self.num_classes = n_classes
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.encoder = Encoder(ModelArgs.dim, ModelArgs.n_heads, ModelArgs.hidden_dim, self.seq_len, ModelArgs.n_layers, self.input_dim)
        self.avg_pool = nn.AvgPool1d(ModelArgs.dim, stride = 1)
        self.fc_1 = nn.Linear(in_features = self.seq_len+1, out_features = 64)
        self.fc_2 = nn.Linear(in_features = 64, out_features = 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.avg_pool(x)
        # x = x[:,:1,:]
        x = x.view(x.size(0), -1)
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        x = self.sigmoid(x)
        
        return x