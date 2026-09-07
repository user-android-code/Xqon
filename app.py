import streamlit as st
import torch
import torch.nn as nn
import torchvision.utils as vutils
from PIL import Image
import hashlib
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="GAN .pth 読み込みアプリ", layout="wide")

st.title("🎨 実在する GAN (.pth) 推論アプリ")
st.write("Hugging Face の実在するアニメGANモデル (.pth) を自動取得して画像を生成します。")

# ---------------------------------------------------------
# 1. ネットワーク構造の定義 (64x64 出力の DCGAN Generator)
# ---------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# ---------------------------------------------------------
# 2. 実在する Hugging Face モデルをダウンロード＆ロード
# ---------------------------------------------------------
@st.cache_resource
def load_gan_model():
    # 実在する公開リポジトリとファイル名
    repo_id = "AlekseyKorshuk/anime-gan"
    filename = "generator.pth"
    
    # .pth の自動ダウンロード
    weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
    
    # モデルのインスタンス化と重み適用
    model = Generator()
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model, repo_id, filename

# モデルの読み込み実行
try:
    with st.spinner("実在モデル (AlekseyKorshuk/anime-gan) を取得中..."):
        model, loaded_repo, loaded_file = load_gan_model()
    st.sidebar.success(f"読込完了:\n{loaded_repo}/{loaded_file}")
except Exception as e:
    st.error(f"ダウンロードエラー: {e}")
    st.stop()

# ---------------------------------------------------------
# 3. メイン画面：プロンプト入力からノイズ計算・推論
# ---------------------------------------------------------
prompt = st.text_input("生成プロンプト（例: 'cool character', 'pink hair'）", value="cute anime girl")

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

        # 結果表示
        col1, col2 = st.columns(2)
        with col1:
            st.image(result_img, caption=f"プロンプト '{prompt}' の生成結果", width=300)
        with col2:
            st.subheader("計算ステータス")
            st.write(f"**生成シード値:** `{seed}`")
            st.write(f"**モデル:** `{loaded_repo}/{loaded_file}`")
