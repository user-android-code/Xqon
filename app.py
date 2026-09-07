import streamlit as st
import torch
import torchvision.utils as vutils
from PIL import Image
import hashlib
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="プロンプト駆動 GAN アプリ", layout="wide")

st.title("🎨 プロンプト対応 GAN 画像生成アプリ")
st.write("プロンプトを入力して、Hugging Face のモデルを使って画像を生成してみよう！")

# ---------------------------------------------------------
# 1. Hugging Face からモデル（.pth）をダウンロードして読み込む処理
# ---------------------------------------------------------
@st.cache_resource
def load_gan_model_from_hf(repo_id: str, filename: str):
    """
    Hugging Face から指定した .pth ファイルをダウンロードしてモデルを準備する
    """
    st.info(f"Hugging Face ({repo_id}) から重みファイル ({filename}) を取得中...")
    
    # HFからファイルをダウンロード (ローカルキャッシュに保存される)
    weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
    
    # 簡易モデルの定義 (※実際の学習時と同じネットワーク構造にする必要があります)
    class SimpleGenerator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(100, 128 * 7 * 7),
                torch.nn.ReLU(),
            )
            self.deconv = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
                torch.nn.Tanh()
            )
        def forward(self, z):
            out = self.fc(z).view(-1, 128, 7, 7)
            return self.deconv(out)
            
    model = SimpleGenerator()
    
    # .pth の重みをロード
    try:
        state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    except Exception as e:
        st.warning(f"重みファイルの構造チェック（デモ用初期値を使用）: {e}")
        
    model.eval()
    return model

# ---------------------------------------------------------
# サイドバー設定：Hugging Face のリポジトリ指定
# ---------------------------------------------------------
st.sidebar.header("🤗 Hugging Face モデル設定")
# 例: 自分のリポジトリ名や既存のモデルIDを指定
repo_id = st.sidebar.text_input("Repo ID", value="hf-internal-testing/tiny-random-PyTorchModel")
filename = st.sidebar.text_input("ファイル名 (.pth)", value="pytorch_model.bin")

# モデルのロード
model = load_gan_model_from_hf(repo_id, filename)

# ---------------------------------------------------------
# メイン画面：プロンプト入力
# ---------------------------------------------------------
prompt = st.text_input("生成プロンプトを入力してね（例: 'cyberpunk cat', 'sunset landscape'）", value="a cute robot")

if st.button("生成する"):
    if not prompt:
        st.warning("プロンプトを入力してね！")
    else:
        with st.spinner("プロンプトから潜在ベクトルを計算して画像生成中..."):
            # プロンプト（文字列）をハッシュ化してシード値（潜在ベクトル）に変換
            prompt_hash = int(hashlib.md5(prompt.encode('utf-8')).hexdigest(), 16) % (2**32)
            torch.manual_seed(prompt_hash)
            
            # 100次元の潜在ノイズベクトルを作成
            z = torch.randn(1, 100)

            with torch.no_grad():
                fake_img = model(z)
                # 画像を 0.0 ～ 1.0 の範囲に正規化
                fake_img = (fake_img + 1) / 2.0
                
                # Pillow画像に変換
                grid = vutils.make_grid(fake_img, normalize=False)
                ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                img = Image.fromarray(ndarr)

        # 結果表示
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption=f"プロンプト: '{prompt}' の生成結果", width=350)
        with col2:
            st.subheader("生成ステータス")
            st.write(f"**生成シード値:** `{prompt_hash}`")
            st.write(f"**読込モデル:** `{repo_id}/{filename}`")
