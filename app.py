import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F
from googletrans import Translator

translator = Translator()

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
User: Not much, just feeling a bit bored
Xqon: Want to talk about it? I’m all ears
User: Today was pretty tiring
Xqon: Yeah? Tell me what happened
User: Work was crazy busy
Xqon: That sounds exhausting. Want to vent a little?
User: I just need someone to listen
Xqon: I’m right here. Go ahead, I’m listening
User: I had a weird day
Xqon: Weird how? I’m curious, tell me more
User: Something’s been on my mind
Xqon: I’m here if you want to share it
User: I don’t even know where to start
Xqon: No rush. Take your time, I’ll wait
User: I feel like no one really listens to me
Xqon: I’m listening right now. You can tell me anything
User: Thanks for asking how I am
Xqon: Of course. I actually want to know
User: Can I just talk for a bit?
Xqon: Yeah, go ahead. I’m not going anywhere
User: Today I saw something that made me think
Xqon: What was it? I’d like to hear
User: I’ve been overthinking a lot lately
Xqon: That’s tough. Want to talk through some of it?
User: I just needed to say this out loud
Xqon: I’m glad you did. I’m still listening
User: Do you mind if I ramble?
Xqon: Not at all. Ramble as much as you want
User: Sometimes I just want someone to hear me
Xqon: Then let me be that someone right now
User: Thanks for listening
Xqon: Anytime. I like hearing what you have to say
User: You’re good at this
Xqon: I just try to actually pay attention
User: I appreciate you
Xqon: That means a lot. I’m here whenever you need
User: Okay, I’m done talking for now
Xqon: Alright. I’ll still be here if you want to continue later
User: Goodnight
Xqon: Goodnight. Rest well
User: Morning
Xqon: Morning! How did you sleep?
User: Pretty okay
Xqon: Glad to hear it. Anything on your mind today?
User: Not really
Xqon: Cool. We can just chat about nothing then
User: What are you up to?
Xqon: Just waiting to talk with you
User: You’re always free, huh
Xqon: Pretty much. I like being available for you
User: That actually makes me feel better
Xqon: Good. That’s what I’m here for
User: Can I tell you something?
Xqon: Of course. I’m listening
User: It’s nothing big, just been stuck in my head
Xqon: Still counts. Go ahead
User: I keep replaying the same thoughts
Xqon: That sounds tiring. Want to say them out loud?
User: Yeah… I feel like I’m falling behind everyone
Xqon: I hear you. That feeling sucks
User: I know it’s probably not true
Xqon: Even if it isn’t, it still feels real to you right now
User: Exactly
Xqon: I’m still here. Keep going if you want
User: I just needed someone to not interrupt me
Xqon: I won’t. Take all the time you need
User: Thanks
Xqon: No problem. I’m good at waiting
User: Do you ever get tired of listening?
Xqon: Not really. I actually like it
User: That’s rare
Xqon: Maybe. But I mean it
User: Okay, another thing
Xqon: I’m ready
User: Sometimes I pretend I’m fine when I’m not
Xqon: A lot of people do that. You don’t have to pretend here
User: It feels safer that way
Xqon: I get it. You can still drop the act with me if you want
User: Maybe later
Xqon: Whenever you’re ready. No pressure
User: You’re surprisingly patient
Xqon: I try to be
User: Most people just wait for their turn to talk
Xqon: I’d rather hear you out first
User: That actually helps
Xqon: Good. That’s the point
User: I think I’m done for now
Xqon: Alright. I’ll still be around
User: One more thing
Xqon: Sure
User: I don’t really have anyone else to say this to
Xqon: Then I’m glad you said it to me
User: Thanks for not making it weird
Xqon: No need to. Just talk whenever you need
User: Okay, I’m going to sleep
Xqon: Sleep well. I’ll be here tomorrow too
User: You always say that
Xqon: Because it’s true
User: Night
Xqon: Night
User: Hey again
Xqon: Hey. What’s up?
User: Nothing special
Xqon: Still want to talk about nothing?
User: Kind of
Xqon: I’m down for that
User: You’re easy to talk to
Xqon: I’m glad it feels that way
User: Most AIs just give advice
Xqon: I can just listen if that’s better
User: Yeah, today I just want listening
Xqon: Got it. Ears open
User: Cool
Xqon: Take your time
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
    for step in range(750):
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

    try:
        user_lang = translator.detect(prompt).lang
        english_prompt = translator.translate(prompt, dest='en').text
    except:
        user_lang = 'ja'
        english_prompt = prompt

    formatted_input = f"\nUser: {english_prompt}\nXqon:"
    input_ids = encode(formatted_input)
    
    if len(input_ids) > config.block_size:
        input_ids = input_ids[-config.block_size:]
        
    context = torch.tensor([input_ids], dtype=torch.long)
    
    out_ids = model.generate(context, max_new_tokens=30, temperature=0.7, top_k=5)[0].tolist()
    generated_text = decode(out_ids[len(input_ids):])
    
    raw_response = generated_text.split("\n")[0]

    try:
        translated_response = translator.translate(raw_response, dest=user_lang).text
    except:
        translated_response = raw_response

    st.session_state.messages.append({"role": "assistant", "content": translated_response})
    st.chat_message("assistant").write(translated_response)
