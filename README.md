# PEFT-from-Scratch: Shakespearean LLM Adaptation (LoRA & DoRA)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=Streamlit&logoColor=white)
![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg?style=for-the-badge)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

This project focuses on implementing **Parameter-Efficient Fine-Tuning (PEFT)** from first principles. This repository contains implementation of **Low-Rank Adaptation (LoRA)** and its 2024 evolution, **Weight-Decomposed Low-Rank Adaptation (DoRA)** from ground-up.

The objective was to transform a base `DistilGPT2` model into a Shakespearean playwright while training less than **0.3%** of the original parameters.

---

## The Interactive Playground
*The final application allows for real-time A/B testing of LoRA vs. DoRA with independent control over creativity (temperature) and stability (repetition penalty).*
![The Application](./application_screenshots/image1.png)

---

## Engineering Details
Modern LLMs are too massive to fine-tune fully on consumer hardware. This project explores the feasibility that we can achieve high-quality stylistic shifts by only updating the "low-rank" components of the LLM's weights.

### Core Technical Implementation
1. **Custom LoRA Layers**: Instead of using the `peft` library, I manually implemented `LoRALinear` wrappers that handle both standard PyTorch `nn.Linear` and GPT-2's specific `Conv1D` layers.
2. **Weight Injection**: A custom recursive script that "surges" through the Transformer architecture, freezing the pre-trained weights and injecting trainable $A$ and $B$ matrices into the Attention (not MLP) blocks.
3. **DoRA Integration**: Implementation of the 2024 DoRA paper, which decomposes the weight update into Magnitude ($m$) and Direction ($V$).

### The Repetition Penalty Discovery
During early testing, both models suffered from **Semantic Degeneration**—the tendency to loop common phrases (e.g., *"We'll go. We'll go."*). 
- **The Finding**: The default repetition penalty of **1.0** was insufficient for a model of this size ($r=8$) to maintain long-term coherence.
- **The Solution**: Implementing a **1.2 penalty** during inference scaled the logits of previously used tokens, forcing the model to explore lower-probability "creative" paths, which successfully broke the loops and restored Shakespearean syntax.
---

## Comparative Analysis: LoRA vs. DoRA

| Metric | LoRA (2021) | DoRA (2024) |
| :--- | :--- | :--- |
| **Trainable Params** | ~235K | ~237K (Adds Magnitude Vector) |
| **Learning Strategy** | Coupled updates | Decoupled Magnitude/Direction |
| **Stylistic Accuracy** | Good thematic shift | **Superior** syntactic accuracy |
| **Validation Loss** | ~4.1 | ~4.1 (Lower is better) |

## Comparison Results: Raw vs. Tuned

### Phase A: Raw Inference (No Penalty)
*Notice the "Degeneration" loops in the second and third prompts.*
![The Application](./application_screenshots/image2.png)

### Phase B: Tuned Inference (Penalty = 1.2)
*Notice the improved coherence and better use of archaic syntax*
![The Application](./application_screenshots/image3.png)

---

### Why DoRA wins here?
Standard LoRA updates both the magnitude and direction of the weight simultaneously, which can be restrictive. **DoRA** mimics full fine-tuning more closely by allowing the model to adjust *how much* a weight changes (magnitude) separately from *the direction* of that change. 

In my tests, this resulted in a much better grasp of Shakespearean pronouns (*thy, thee*) compared to standard LoRA.

---

## Results & Insights
*Loss Data can be viewed from the training notebook in the repo*

- **Parameter Reduction**: Successfully reduced trainable parameters from **82M** to **~250K** (a 99.7% reduction).
- **Training Efficiency**: Achieved convergence on the TinyShakespeare dataset in just 3 epochs using a T4 GPU on Google Colab.
- **Inference Portability**: The resulting adapters are only **~2.5MB** each, making them ultra-portable and capable of running on a standard local CPU via the included Streamlit app.

---

## Performance Tuning (Lessons Learned)

1. **Learning Rate Scaling**: Found that adapters require a much higher learning rate (**5e-4**) than full fine-tuning to overcome the "gravity" of the frozen weights.

---

## Local Setup & Usage

1. **Clone the repo:**
   ```bash
   git clone [https://github.com/RamuNalla/LORA-from-scratch.git](https://github.com/RamuNalla/LORA-from-scratch.git)
   cd LoRA-from-scratch
   pip install -r requirements.txt
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Playground:**
   ```bash
   streamlit run app.py
   ```


## Local Setup & Usage

`src/`: Custom LoRA/DoRA injection logic.

`app.py`: The Streamlit comparison application.

`notebooks/`: The training and Perplexity analysis logs.