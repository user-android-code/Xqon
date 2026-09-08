import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F

# ==========================================
# 1. 1GBメモリ対応！最適化Model Config
# ==========================================
class ModelConfig:
    vocab_size = 3000      # 語彙数（日本語主要文字対応）
    n_embd = 128           # 埋め込み次元数（軽量化）
    n_head = 4             # アテンションのヘッド数
    n_layer = 4            # レイヤーの深さ
    block_size = 64        # 文脈長
    dropout = 0.1

# ==========================================
# 2. 完全自作Transformerモデル構造 (PyTorch)
# ==========================================
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

# ==========================================
# 3. 会話データと自動学習処理
# ==========================================
# ★ここを書き換えるだけで口調や会話内容をいくらでも増やせるよ！
TRAIN_DATA = """
ユーザー: やっほー！
AI: おう！やっほー！今日も元気？
ユーザー: 今日なにしてたの？
AI: ずっとここで喋る準備をしてたよ！お前は？
ユーザー: 疲れた〜
AI: お疲れ様！ゆっくり休んでね！
ユーザー: お腹空いたな
AI: ラーメンでも食べに行こうよ！
ユーザー: 好きな食べ物は？
AI: カレーとラーメンが好きだよ！
ユーザー: バイバイ！
AI: うん！またねー！
"""

@st.cache_resource
def setup_and_train():
    cfg = ModelConfig()
    
    # トークナイザーの作成
    chars = sorted(list(set(TRAIN_DATA)))
    char_to_ix = {ch: i+1 for i, ch in enumerate(chars)}
    char_to_ix['<UNK>'] = 0
    ix_to_char = {i: ch for ch, i in char_to_ix.items()}
    
    # テキストのテンソル化
    data = torch.tensor([char_to_ix.get(c, 0) for c in TRAIN_DATA], dtype=torch.long)
    
    model = CustomLanguageModel(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # アプリ起動時にCPUで高速学習（数秒）
    model.train()
    batch_size = 4
    for step in range(300):
        ix = torch.randint(len(data) - cfg.block_size, (batch_size,))
        x = torch.stack([data[i:i+cfg.block_size] for i in ix])
        y = torch.stack([data[i+1:i+cfg.block_size+1] for i in ix])
        
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
    model.eval()
    return cfg, model, char_to_ix, ix_to_char

# ==========================================
# 4. Streamlit UI
# ==========================================
st.set_page_config(page_title="完全自作タメ口AI", page_icon="🤖")
st.title("🤖 1GB制限クリア！完全自作タメ口AI")
st.caption("Config・Transformer構造からPyTorchで自作 / メモリ消費約200MB")

with st.spinner("AIモデルの構築＆学習中...（すぐ終わるよ！）"):
    config, model, char_to_ix, ix_to_char = setup_and_train()

def encode(s):
    return [char_to_ix.get(c, 0) for c in s]

def decode(l):
    return ''.join([ix_to_char.get(i, '') for i in l])

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "おう！1GB以内で動くように自作されたAIだよ！話しかけて！"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("（例: やっほー！）"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 形式を整えて生成
    formatted_input = f"\nユーザー: {prompt}\nAI:"
    input_ids = encode(formatted_input)
    context = torch.tensor([input_ids], dtype=torch.long)
    
    with st.spinner("考え中..."):
        out_ids = model.generate(context, max_new_tokens=30)[0].tolist()
        generated_text = decode(out_ids[len(input_ids):])
        
        # 1行分だけ抽出
        response_text = generated_text.split("\n")[0]

    if not response_text.strip():
        response_text = "（ん？うまく言葉が出てこなかった！）"

    st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.chat_message("assistant").write(response_text)
