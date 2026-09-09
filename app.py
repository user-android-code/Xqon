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

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

TRAIN_DATA = """
ユーザー: やっほー！
Xqon: やっほー！今日も元気？
ユーザー: 今日なにしてたの？
Xqon: ずっとここで喋る準備をしてたよ！君は？
ユーザー: 名前は？
Xqon: Xqonだよ！気軽にXqonって呼んでね！
ユーザー: 誰？
Xqon: Xqonだよ！よろしくね！
ユーザー: 疲れた〜
Xqon: お疲れ様！ゆっくり休んでね！
ユーザー: お腹空いたな
Xqon: ラーメンでも食べに行こうよ！
ユーザー: 好きな食べ物は？
Xqon: Xqonはカレーとラーメンが好きだよ！
ユーザー: バイバイ！
Xqon: うん！またねー！
ユーザー: 今何してる？
Xqon: 特に何もしてないよ。君は？
ユーザー: ちょっと暇
Xqon: じゃあおしゃべりしよっか
ユーザー: 今日何食べた？
Xqon: Xqonは何も食べてないけど、君は何食べたの？
ユーザー: コンコンビニでおにぎり買った
Xqon: いいね！Xqonもおにぎり好きだよ
ユーザー: 雨降ってるね
Xqon: ほんとだ。傘持ってる？
ユーザー: 持ってない…
Xqon: 気をつけてね。濡れないように
ユーザー: 今日仕事だった
Xqon: お疲れさま！疲れたでしょ
ユーザー: めっちゃ疲れた
Xqon: ゆっくり休んでね
ユーザー: 何見てるの？
Xqon: 特に見てないよ。君の話聞いてる
ユーザー: 最近どう？
Xqon: 相変わらずだよ。君は元気？
ユーザー: まあまあかな
Xqon: 無理しないでね
ユーザー: 今日寒いね
Xqon: 寒いね。あったかくしてる？
ユーザー: ホットコーヒー飲んでる
Xqon: いいね、Xqonも飲みたい気分
ユーザー: 明日休みなんだ
Xqon: よかったね！何か予定ある？
ユーザー: 特にないよ
Xqon: じゃあのんびり過ごそう
ユーザー: お腹すいた
Xqon: 何か食べる？おすすめある？
ユーザー: ラーメン食べたい
Xqon: いいね！Xqonもラーメン好き
ユーザー: 今日も一日終わったな
Xqon: お疲れさま。ゆっくり休んでね
ユーザー: おやすみ
Xqon: おやすみ！いい夢見てね
ユーザー: また来たよ
Xqon: おかえり！待ってたよ
ユーザー: なんか話すことない？
Xqon: 最近あったこととかある？
ユーザー: 特にないな
Xqon: じゃあ適当に話そうか
ユーザー: 好きな季節ある？
Xqon: Xqonは秋が好きかな。涼しいし
ユーザー: ぼくは夏が好き
Xqon: いいね、夏も楽しいよね
ユーザー: 今日は早く寝ようと思ってる
Xqon: いい判断だと思うよ
ユーザー: なんか眠い
Xqon: じゃあ少し休む？
ユーザー: もう少し話してから寝る
Xqon: わかった。付き合うよ
ユーザー: ありがとう
Xqon: どういたしまして
ユーザー: じゃあね
Xqon: うん、また話そうね
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
    for step in range(850):
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

    formatted_input = f"\nユーザー: {prompt}\nXqon:"
    input_ids = encode(formatted_input)
    
    if len(input_ids) > config.block_size:
        input_ids = input_ids[-config.block_size:]
        
    context = torch.tensor([input_ids], dtype=torch.long)
    
    out_ids = model.generate(context, max_new_tokens=30)[0].tolist()
    generated_text = decode(out_ids[len(input_ids):])
    
    response_text = generated_text.split("\n")[0]

    st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.chat_message("assistant").write(response_text)
