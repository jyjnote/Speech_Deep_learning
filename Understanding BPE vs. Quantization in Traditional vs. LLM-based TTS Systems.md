# 📚 Understanding BPE vs. Quantization in Traditional vs. LLM-based TTS Systems

## 🧱 1. Traditional TTS Pipeline

- **BPE Tokenization**
  - Purpose: Converts raw text into subword tokens
  - Role: Input to the TTS model
  - Example: `"internationalization"` → `["inter", "national", "ization"]`

- **Quantization (VQ / FSQ)**
  - Purpose: Converts continuous audio features (e.g., spectrogram) into discrete tokens
  - Role: Used **only as training labels**
  - Example: waveform → mel → VQ → `[32, 78, 5, 91, ...]`

- **Separation of Modalities**
  - BPE and quantized tokens are processed **independently**
  - No shared vocabulary, no interaction in training

### ✅ Summary
- Inputs: BPE tokens
- Targets: Quantized speech tokens
- Training: `BPE → Model → Quantized Tokens (Label)`
- No context sharing between modalities

---

## 🤖 2. LLM-based TTS (e.g., Spark-TTS, VALL-E, Llasa)

- **Unified Token Stream**
  - Input and output tokens are concatenated into a **single autoregressive sequence**
  - Example input:  
    ```
    [<BOS>, BPE_1, BPE_2, ..., GENDER, PITCH=3, SPEED=2]
    ```
  - Target:  
    ```
    [global_tok_1, ..., sem_tok_1, sem_tok_2, ...]
    ```

- **Shared Vocabulary & Embedding Space**
  - BPE, attribute tokens, and quantized audio tokens are **embedded in the same space**
  - Enables smooth alignment between text and audio modalities

- **Transformer-based Self-Attention**
  - Every token (text or audio) can **attend to all previous tokens**
  - Enables **contextual learning** across modalities

### ✅ Summary
- Inputs: `[BPE + attributes]`
- Outputs: `[quantized audio tokens]`
- Training: `LM learns to map unified sequence → audio`
- Self-attention enables full contextual learning
- Efficient, controllable, and zero-shot capable

---

## 🧠 Conclusion

| Feature                     | Traditional TTS       | LLM-based TTS (Spark-TTS)      |
|----------------------------|-----------------------|--------------------------------|
| Token Stream               | Separate              | Unified (BPE + Quantized)      |
| BPE ↔ Quantization Link    | ❌ None               | ✅ Indirect via LLM training    |
| Embedding Space            | Separate              | Shared                         |
| Training Targets           | Quantized audio only  | Quantized audio + attributes   |
| Contextual Interaction     | ❌ None               | ✅ Full via self-attention      |

> ✅ LLM-based TTS unifies all modalities into a single, learnable stream using transformer-based self-attention — enabling compact, efficient, and controllable speech synthesis.
