from transformers import AutoModelForCausalLM
from src.injection import inject_adapter
from src.utils import print_trainable_parameters

# Load DistilGPT2
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

print("--- Before LoRA ---")
print_trainable_parameters(model)

# Apply Injection
model = inject_adapter(model, rank=8, use_dora=True)

print("\n--- After DoRA Injection ---")
print_trainable_parameters(model)