import streamlit as st
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(page_title="XQon")
st.title("XQon s-1")

MODEL_ID = "Nagohachi/tiny-lm-japanese-500m-dpo-v1"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, 
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    return tokenizer, model

tokenizer, model = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input(""):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner(""):
            messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
            
            try:
                input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                input_text = prompt

            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            input_length = inputs["input_ids"].shape[1]
            response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
