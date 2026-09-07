import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
import time

# --- 1. モデル定義 (256x256 軽量GAN) ---
class Generator(nn.Module):
    def __init__(self, z_dim=100, feature_maps=32):
        super().__init__()
        self.feature_maps = feature_maps
        self.fc = nn.Sequential(
            nn.Linear(z_dim, feature_maps * 8 * 4 * 4),
            nn.BatchNorm1d(feature_maps * 8 * 4 * 4),
            nn.ReLU(True)
        )
        def up_block(in_c, out_c):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(True)
            )
        self.net = nn.Sequential(
            up_block(feature_maps * 8, feature_maps * 8), # 4->8
            up_block(feature_maps * 8, feature_maps * 4), # 8->16
            up_block(feature_maps * 4, feature_maps * 4), # 16->32
            up_block(feature_maps * 4, feature_maps * 2), # 32->64
            up_block(feature_maps * 2, feature_maps),     # 64->128
            nn.Upsample(scale_factor=2, mode='nearest'),  # 128->256
            nn.Conv2d(feature_maps, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, self.feature_maps * 8, 4, 4)
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, feature_maps=32):
        super().__init__()
        def down_block(in_c, out_c, norm=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
            if norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.net = nn.Sequential(
            down_block(3, feature_maps, norm=False),
            down_block(feature_maps, feature_maps * 2),
            down_block(feature_maps * 2, feature_maps * 4),
            down_block(feature_maps * 4, feature_maps * 4),
            down_block(feature_maps * 4, feature_maps * 8),
            down_block(feature_maps * 8, feature_maps * 8),
            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.net(img).view(-1, 1)

# --- 2. テンソルをPIL画像に変換するヘルパー関数 ---
def tensor_to_pil(tensor):
    img_np = tensor.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    img_np = ((img_np + 1) / 2.0 * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_np)

# --- 3. セッション状態の初期化 ---
if "netG" not in st.session_state:
    st.session_state.netG = Generator()
    st.session_state.netD = Discriminator()
    st.session_state.fixed_z = torch.randn(1, 100) # 生成画像確認用の固定ノイズ

# --- 4. UI 画面構成 ---
st.title("🎨 オールインワン CPU 256x256 GAN")
st.write("学習から推論（画像生成）まで、これ1つのファイルで完結します。")

# サイドバー操作
st.sidebar.header("設定パネル")
mode = st.sidebar.radio("モード選択", ["一瞬で画像生成 (推論)", "CPUで簡易学習 (トレーニング)"])

# ----------------------------------------------------
# モード1: 推論 (一瞬で画像生成)
# ----------------------------------------------------
if mode == "一瞬で画像生成 (推論)":
    st.subheader("⚡ 瞬時画像生成モード")
    
    # 潜在変数を動かすスライダー
    slider_val = st.slider("潜在変数の操作 (z[0])", -3.0, 3.0, 0.0, 0.1)
    
    z = st.session_state.fixed_z.clone()
    z[0, 0] = slider_val

    # 一瞬で計算（勾配計算なし）
    with torch.no_grad():
        st.session_state.netG.eval()
        fake_img_tensor = st.session_state.netG(z)
        image = tensor_to_pil(fake_img_tensor)

    st.image(image, caption="生成された 256x256 画像", width=350)

# ----------------------------------------------------
# モード2: CPU学習 (その場でトレーニング)
# ----------------------------------------------------
else:
    st.subheader("🏋️‍♂️ CPU 簡易学習モード")
    st.write("ダミーデータ（ランダム画像）を使って、モデルの学習ステップを回します。")

    epochs = st.number_input("ステップ数 (Epochs)", min_value=1, max_value=100, value=10)
    
    if st.button("学習を開始する"):
        netG = st.session_state.netG
        netD = st.session_state.netD
        netG.train()
        netD.train()

        optG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
        criterion = nn.BCELoss()

        # 表示領域の作成
        img_placeholder = st.empty()
        status_text = st.empty()
        progress_bar = st.progress(0)

        # 簡易学習ループ (CPU用にバッチサイズ4)
        batch_size = 4
        for step in range(epochs):
            # ダミーの本物画像 (256x256) とラベル
            real_imgs = torch.randn(batch_size, 3, 256, 256)
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            # --- D の更新 ---
            optD.zero_grad()
            z = torch.randn(batch_size, 100)
            fake_imgs = netG(z)
            
            loss_d_real = criterion(netD(real_imgs), real_labels)
            loss_d_fake = criterion(netD(fake_imgs.detach()), fake_labels)
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optD.step()

            # --- G の更新 ---
            optG.zero_grad()
            loss_g = criterion(netD(fake_imgs), real_labels)
            loss_g.backward()
            optG.step()

            # UI更新 (学習中の生成画像を表示)
            with torch.no_grad():
                netG.eval()
                sample_tensor = netG(st.session_state.fixed_z)
                sample_img = tensor_to_pil(sample_tensor)
                netG.train()

            img_placeholder.image(sample_img, caption=f"ステップ {step+1}/{epochs} の生成結果", width=300)
            status_text.text(f"Step [{step+1}/{epochs}] | Loss D: {loss_d.item():.4f} | Loss G: {loss_g.item():.4f}")
            progress_bar.progress((step + 1) / epochs)
            time.sleep(0.1)

        st.success("学習ステップが完了しました！「一瞬で画像生成」モードに変えると、学習結果を試せます。")
