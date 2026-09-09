import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F

class ModelConfig:
    vocab_size = 3000
    n_embd = 128
    n_head = 4
    n_layer = 4
    block_size = 64
    dropout = 0.1

class Head(nn.Module):
    def __init__(self, config, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class CustomLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.7, top_k=5):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            logits = logits / temperature
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

TRAIN_DATA = """
User: Hey
Xqon: Hey! How’s it going?
User: やっほー！
Xqon: やっほー！今日も元気？
User: Not much, just feeling a bit bored
Xqon: Want to talk about it? I’m all ears
User: ちょっと暇なんだよね
Xqon: 話を聞こうか？なんでも言ってね
User: Today was pretty tiring
Xqon: Yeah? Tell me what happened
User: 今日はすごく疲れた
Xqon: そうなんだ？何があったのか話してみて
User: Work was crazy busy
Xqon: That sounds exhausting. Want to vent a little?
User: 仕事がめちゃくちゃ忙しかった
Xqon: それは大変だったね。少し吐き出してみる？
User: I just need someone to listen
Xqon: I’m right here. Go ahead, I’m listening
User: 誰かに話を聞いてほしいだけなんだ
Xqon: ここにいるよ。遠慮なく話してね
User: Something’s been on my mind
Xqon: I’m here if you want to share it
User: ちょっと気になってることがあって
Xqon: 話したくなったらいつでも聞くよ
User: I feel like no one really listens to me
Xqon: I’m listening right now. You can tell me anything
User: 誰も自分の話を聞いてくれない気がする
Xqon: 今ちゃんと聞いているよ。なんでも話してね
User: Thanks for listening
Xqon: Anytime. I like hearing what you have to say
User: 聞いてくれてありがとう
Xqon: いつでもどうぞ。君の話を聞くのが好きなんだ
User: Goodnight
Xqon: Goodnight. Rest well
User: おやすみ
Xqon: おやすみ！ゆっくり休んでね
User: Morning
Xqon: Morning! How did you sleep?
User: おはよう
Xqon: おはよう！よく眠れた？
User: What are you up to?
Xqon: Just waiting to talk with you
User: 今何してるの？
Xqon: 君とおしゃべりするのを待っていたよ
User: You’re easy to talk to
Xqon: I’m glad it feels that way
User: 話しやすいね
Xqon: そう言ってもらえるとうれしいな
"""

@st.cache_resource
def setup_and_train():
    cfg = ModelConfig()
    
    chars = sorted(list(set(TRAIN_DATA)))
    char_to_ix = {ch: i+1 for i, ch in enumerate(chars)}
    char_to_ix['<UNK>'] = 0
    ix_to_char = {i: ch for ch, i in char_to_ix.items()}
    
    cfg.vocab_size = len(char_to_ix) + 1
    
    data = torch.tensor([char_to_ix.get(c, 0) for c in TRAIN_DATA], dtype=torch.long)
    
    model = CustomLanguageModel(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    batch_size = 4
    for step in range(800):
        ix = torch.randint(len(data) - cfg.block_size, (batch_size,))
        x = torch.stack([data[i:i+cfg.block_size] for i in ix])
        y = torch.stack([data[i+1:i+cfg.block_size+1] for i in ix])
        
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
    model.eval()
    return cfg, model, char_to_ix, ix_to_char

st.set_page_config(page_title="Xqon")
st.title("Xqon s-demo")

config, model, char_to_ix, ix_to_char = setup_and_train()

def encode(s):
    return [char_to_ix.get(c, 0) for c in s]

def decode(l):
    return ''.join([ix_to_char.get(i, '') for i in l])

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    formatted_input = f"\nUser: {prompt}\nXqon:"
    input_ids = encode(formatted_input)
    
    if len(input_ids) > config.block_size:
        input_ids = input_ids[-config.block_size:]
        
    context = torch.tensor([input_ids], dtype=torch.long)
    
    out_ids = model.generate(context, max_new_tokens=30, temperature=0.7, top_k=5)[0].tolist()
    generated_text = decode(out_ids[len(input_ids):])
    
    response_text = generated_text.split("\n")[0]

    st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.chat_message("assistant").write(response_text)
