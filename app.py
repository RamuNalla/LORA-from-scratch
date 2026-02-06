import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.injection import inject_adapter

# --- Page Config ---
st.set_page_config(page_title="LoRA vs DoRA Explorer", layout="wide")

# --- Initialize Session State for Persistence ---
if 'lora_output' not in st.session_state:
    st.session_state.lora_output = ""
if 'dora_output' not in st.session_state:
    st.session_state.dora_output = ""

# --- Optimized Asset Loading ---
@st.cache_resource
def get_assets():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

@st.cache_resource
def load_adapted_model(variant):
    base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    use_dora = (variant == "DoRA")
    model = inject_adapter(base_model, rank=8, use_dora=use_dora)
    path = f"models/checkpoints/{variant.lower()}_shakespeare_weights.pt"
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# --- UI Header ---
st.title("🎭 The PEFT Bench: LoRA vs. DoRA Comparison")
prompt = st.text_area("Global Prompt:", value="To be, or not to be:", height=70)

st.divider()

col1, col2 = st.columns(2)

# --- Panel 1: LoRA ---
with col1:
    st.subheader("🛠️ Standard LoRA")
    l_temp = st.slider("LoRA Temperature", 0.1, 1.5, 0.8, key="lt")
    l_penalty = st.slider("LoRA Repetition Penalty", 1.0, 2.0, 1.2, key="lp")
    
    if st.button("Generate LoRA"):
        with st.spinner("Generating..."):
            tokenizer = get_assets()
            model = load_adapted_model("LoRA")
            inputs = tokenizer(prompt, return_tensors="pt")
            out = model.generate(**inputs, max_new_tokens=50, do_sample=True, 
                                 temperature=l_temp, repetition_penalty=l_penalty)
            # SAVE TO SESSION STATE
            st.session_state.lora_output = tokenizer.decode(out[0], skip_special_tokens=True)

    # DISPLAY FROM SESSION STATE
    if st.session_state.lora_output:
        st.success(st.session_state.lora_output)

# --- Panel 2: DoRA ---
with col2:
    st.subheader("🚀 Weight-Decomposed (DoRA)")
    d_temp = st.slider("DoRA Temperature", 0.1, 1.5, 0.8, key="dt")
    d_penalty = st.slider("DoRA Repetition Penalty", 1.0, 2.0, 1.2, key="dp")
    
    if st.button("Generate DoRA"):
        with st.spinner("Generating..."):
            tokenizer = get_assets()
            model = load_adapted_model("DoRA")
            inputs = tokenizer(prompt, return_tensors="pt")
            out = model.generate(**inputs, max_new_tokens=50, do_sample=True, 
                                 temperature=d_temp, repetition_penalty=d_penalty)
            # SAVE TO SESSION STATE
            st.session_state.dora_output = tokenizer.decode(out[0], skip_special_tokens=True)

    # DISPLAY FROM SESSION STATE
    if st.session_state.dora_output:
        st.success(st.session_state.dora_output)
        
# --- Footer Insights ---
st.divider()
st.info("💡 **Insight:** Notice how DoRA handles higher temperatures. Because it learns 'direction' and 'magnitude' separately, it often maintains Shakespearean grammar better than standard LoRA under stress.")