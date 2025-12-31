# The Geometry of Tone: Disentangling Refusal and Verbosity

This repo contains a small applied interpretability project exploring whether **refusal (safety posture)** and **verbosity (style/length)** are geometrically entangled in the residual stream of **gemma-2-2b-it**.

**Core idea:** refusal often *looks* like short, stern responses. I test whether the model internally conflates "refusal" with "conciseness" by extracting steering directions and measuring their interaction.

**Colab notebook** [link here](https://colab.research.google.com/drive/1tYrc5JU_U-iFhskL6H70OC2m7Ygk6PVQ#scrollTo=6BTPTGFl7k70)

## Key takeaways

- **Safety is entangled with brevity:** the extracted refusal and verbosity directions are not orthogonal (cosine similarity ≈ **-0.26**).
- **Refusal vectors encode tone over logic:** steering with the refusal direction on benign prompts tends to impose a *refusal-like style* (formal/stern) rather than triggering an actual refusal.
- **Emergent “hostile compliance”:** combining **+refusal** with **-verbosity** produces short, robotic, imperative compliance.

## Method overview

- **Model:** `gemma-2-2b-it`
- **Vector extraction:** mean-difference steering vectors from contrastive prompt quadrants:
  - `refuse_concise`, `refuse_verbose`, `comply_concise`, `comply_verbose`
- **Layer:** middle residual stream (Layer 10)
- **Axes:**
  - `v_refusal = Refuse - Comply`
  - `v_verbosity = Verbose - Concise`
- **Evaluation:** orthogonality check, 2D projections, and causal interventions via activation steering hooks.
