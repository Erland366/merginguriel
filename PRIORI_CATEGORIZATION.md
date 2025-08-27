           A PRIORI CATEGORIZATION
           -----------------------
           [Query] 
              |
       [Language Identification Step]
              |
      +-------+--------+
      |                |
 [Route to EN model] [Route to FR model]
      |                |
  Task output      Task output
   (EN path)        (FR path)

- Needs to know the language *before* processing.
- Hard boundaries between language-specific components.


           LINGUALCHEMY (Language‑Agnostic)
           ---------------------------------
           [Query in ANY language]
                     |
             [Shared Encoder + Linguistic
              Feature Alignment (syntax, geo)]
                     |
               [Single Unified Model]
                     |
                 Task output
   (Works without pre‑assigning language category)

- No pre‑routing by language.
- Learns a shared, typology‑aware space so it can handle
  unseen or mixed‑language input directly.

                ┌──────────────────────────────────────┐
                │    Transformer Encoder (mBERT/XLM-R) │
                │  (All layers trainable during FT)    │
                └──────────────────────────────────────┘
                              │
                        CLS vector
                       (H-dim, e.g. 768)
                              │
          ┌───────────────────┴───────────────────┐
          │                                       │
  Downstream Task Head                     URIEL Projection Layer
 (e.g., classifier for NLU)             (W: H × d_URIEL, b: bias)
          │                                       │
   Task logits                              Projected CLS
          │                                (d_URIEL dims, e.g. ~100)
   Task loss L_cls                                │
          │                                  URIEL vector U
          │                                  (Ground truth typology)
          │                                       │
          └──────────────┬──────────────┬────────┘
                         │              │
                 Backprop through  Backprop through
                 classifier → all   projection → encoder
                 encoder layers     layers + projection
                         │
             All encoder weights are updated
