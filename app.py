import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

class ModelConfig:
    vocab_size = 100277
    n_embd = 768
    n_head = 12
    n_layer = 12
    block_size = 128
    dropout = 0.1

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Custom100MLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.6, top_k=5):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

TRAIN_DATA = """
User: やっほー！
Xqon: やっほー！今日も調子はどう？
User: Hey!
Xqon: Hey there! How's your day going?
User: 今日疲れたなー
Xqon: お疲れ様！今日はどんな一日だったの？
User: I'm feeling a bit tired today.
Xqon: I hear you. Make sure to get some rest tonight!
User: Xqonって誰？
Xqon: 僕はXqon（クオン）だよ！君とお話しするAIアシスタントさ。
User: Who are you?
Xqon: I'm Xqon, your friendly AI companion!
User: 何か楽しいことないかな？
Xqon: 一緒に何か面白い話でもしようか！何が好き？
User: I'm bored.
Xqon: Let's chat! Tell me about your favorite hobbies.
User: ありがとう！
Xqon: どういたしまして！いつでも気軽に話しかけてね。
User: Thanks!
Xqon: You're welcome! I'm always here to talk.
"""

@st.cache_resource
def setup_model():
    cfg = ModelConfig()
    enc = tiktoken.get_encoding("cl100k_base")
    
    model = Custom100MLanguageModel(cfg)
    
    tokens = enc.encode(TRAIN_DATA)
    data = torch.tensor(tokens, dtype=torch.long)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    model.train()
    batch_size = 2
    block_size = cfg.block_size
    
    for step in range(150):
        if len(data) <= block_size:
            break
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
    model.eval()
    return cfg, model, enc

st.set_page_config(page_title="Xqon s-demo")
st.title("Xqon s-demo")

config, model, enc = setup_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    formatted_input = f"\nUser: {prompt}\nXqon:"
    input_ids = enc.encode(formatted_input)
    
    if len(input_ids) > config.block_size:
        input_ids = input_ids[-config.block_size:]
        
    context = torch.tensor([input_ids], dtype=torch.long)
    
    out_ids = model.generate(context, max_new_tokens=40, temperature=0.6, top_k=5)[0].tolist()
    generated_text = enc.decode(out_ids[len(input_ids):])
    
    response_text = generated_text.split("\n")[0].strip()

    if response_text:
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.chat_message("assistant").write(response_text)
