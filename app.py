import streamlit as st
import torch
import torch.nn as nn
import torchvision.utils as vutils
from PIL import Image
import hashlib
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="GAN .pth 読み込み計算アプリ", layout="wide")

st.title("🎨 GAN (.pth) 推論アプリ")
st.write("公開リポジトリから学習済み `.pth` をダウンロードして、画像生成計算を実行します。")

# ---------------------------------------------------------
# 1. 計算方法（Generatorのネットワーク構造）の定義
# ---------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 入力: 潜在ノイズ z (100次元)
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # 出力: 3 x 64 x 64 (RGB画像)
        )

    def forward(self, input):
        return self.main(input)

# ---------------------------------------------------------
# 2. Hugging Face から .pth を自動ダウンロードして重みをセット
# ---------------------------------------------------------
@st.cache_resource
def load_gan_model(repo_id: str, filename: str):
    # .pth ファイルの自動ダウンロード
    weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
    
    # モデルのインスタンス化
    model = Generator()
    
    # 学習済み重み (.pth) の読み込み計算
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    
    # 重みデータのキー調整（もし不要なプレフィックスがあれば自動除外）
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
        
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# ---------------------------------------------------------
# サイドバー設定（公開リポジトリの指定）
# ---------------------------------------------------------
st.sidebar.header("🤗 Hugging Face 設定")
repo_id = st.sidebar.text_input("Repo ID", value="csinva/gan-pretrained-pytorch")
filename = st.sidebar.text_input("ファイル名", value="celebA_dcgan.pth")

# モデルの準備
try:
    with st.spinner(".pth ファイルをダウンロード＆計算準備中..."):
        model = load_gan_model(repo_id, filename)
    st.sidebar.success("モデル準備完了！")
except Exception as e:
        st.error(f"モデルのロードに失敗しました: {e}")
        st.stop()

# ---------------------------------------------------------
# 3. メイン画面：プロンプト入力からノイズ計算・推論
# ---------------------------------------------------------
prompt = st.text_input("生成プロンプト（例: 'cool character', 'sunset view'）", value="cute anime face")

if st.button("画像生成（推論を実行）"):
    if not prompt:
        st.warning("プロンプトを入力してね！")
    else:
        with st.spinner("生成計算中..."):
            # プロンプト文字をハッシュ化してランダムシード（100次元ノイズ）にする
            seed = int(hashlib.md5(prompt.encode('utf-8')).hexdigest(), 16) % (2**32)
            torch.manual_seed(seed)
            
            # 100次元の潜在ベクトル z (1, 100, 1, 1) の作成
            z = torch.randn(1, 100, 1, 1)

            # 推論計算
            with torch.no_grad():
                fake_tensor = model(z)
                # 画像の値を -1~1 から 0~1 に補正
                fake_tensor = (fake_tensor + 1) / 2.0
                
                # Tensor -> PIL Image 変換
                grid = vutils.make_grid(fake_tensor, normalize=False)
                ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                result_img = Image.fromarray(ndarr)

        # 結果の描画
        col1, col2 = st.columns(2)
        with col1:
            st.image(result_img, caption=f"プロンプト '{prompt}' の生成結果", width=300)
        with col2:
            st.subheader("計算ステータス")
            st.write(f"**生成シード値:** `{seed}`")
            st.write(f"**出力解像度:** `{result_img.size[0]} x {result_img.size[1]} px`")
