# RubricBasedRewardSystem_StructuredReasoning
built a reproducible, highly efficient pipeline to train Gemma-2–2B-IT to produce verified, structured reasoning traces wrapped in clean syntax (&lt;reasoning>...&lt;/reasoning>&lt;answer>...&lt;/answer>).

# WHAT IS THIS?
This notebook fine-tunes Gemma-2-2B-IT to reliably produce structured outputs in the form: <reasoning>...</reasoning><answer>...</answer> The goal is to improve format compliance, answer correctness, and overall response quality using a two-stage pipeline that is reproducible on a single Kaggle TPU v5e-8 session (with checkpointing for multi-session runs).

# Configuration 🔧
Before running the training stages, you need to configure your environment variables and access tokens for model fetching and dataset synchronization. Use the example.env file as a template:

1. Rename example.env to .env.
2. Add your specific credentials, model flags, and infrastructure targets:

## Code snippet
-> Hugging Face Access (Required for downloading Gemma-2/3 weights)
HF_TOKEN=your_huggingface_token_here

-> Dataset Configuration (Kaggle or HF Sync)
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key

-> Hardware & Mesh Configurations
TPU_NAME=local
JAX_PLATFORMS=tpu

# Overall training and evaluation strategy
## Stage 1 — Supervised Fine-Tuning (SFT)
We first teach the model the desired output structure using instruction-style data that includes reasoning traces and final answers. This establishes a strong baseline that follows the required tag format.

## Stage 2 — GRPO (Reinforcement Learning)
After SFT establishes the basic <reasoning>...<answer> structure, we apply Group Relative Policy Optimization (GRPO) to refine both reasoning quality and answer correctness. GRPO optimizes the model by comparing outputs within groups rather than individual samples, enabling stable training with our multi-faceted reward signal.

### Compute allocation strategy
We trained entirely on Kaggle TPU v5e-8. To maximize available compute:

We used QLoRA to reduce memory footprint during SFT.
SFT consumed a small fraction of compute (1 epoch) to learn output format.
GRPO consumed the majority of TPU time for reward optimization.
We split the TPU mesh during GRPO:
(1,4) devices for policy + reference
(1,4) devices for judge model
This enabled parallel rollouts and judge scoring, avoiding sequential bottlenecks.
This layout allowed us to run RL with a Gemma-2-2B policy and a Gemma-3-12B judge entirely within Kaggle resource constraints.

### Reward setup for RL
1. Format Reward
Binary reward indicating correct <reasoning>...</reasoning><answer>...</answer> structure.
2. Exact Answer Reward
Binary numeric match reward for tasks with verifiable answers.
3. G-RaR (Rubric-Guided LLM-as-Judge Reward)
A continuous reward computed via a Gemma-3-12B judge using:

-> task type
-> task-specific rubrics
-> softmax expectation over discrete {1..5} rating tokens
This produces a 0–1 normalized continuous signal. We used group-relative advantage to stabilize GRPO and reduce variance across heterogeneous tasks.

## Evaluation strategy
Evaluation strategy We tracked four evaluation axes:

1.Format accuracy
2. Exact match accuracy
3. Partial numeric match
4. Rubric-based reasoning score (via G-RaR)

## Dataset
We train on our custom TFDS dataset reasoning_from_hf, designed as a dual-purpose corpus: an SFT split to teach structured reasoning traces, and an RL split to support GRPO with rubric-based rewards.

### 1.Overview
<img width="733" height="153" alt="image" src="https://github.com/user-attachments/assets/5239e475-46e0-4a78-94f0-f990bf8b928e" />

Public access: The final dataset is published on Hugging Face Hub(https://huggingface.co/datasets/comoZ/reasoning-dataset) and mirrored as a Kaggle TFDS dataset(https://www.kaggle.com/datasets/hojuna/comoz-tfds) for evaluation.

## Technologies Used 🧰
1. Google Gemma Ecosystem: Gemma-2-2B-IT (Policy/Reference), Gemma-3-12B (Continuous Reward Judge), and Gemma-3-27B-IT (Offline Rubric Generation).
2. PyTorch & PyTorch/XLA: For deep learning compilation and model parallel execution structures on Google Cloud TPU hardware.
3. TRL (Transformer Reinforcement Learning): For native Group Relative Policy Optimization (GRPO) loop structures.
4. Hugging Face Accelerate & Transformers: Managing Parameter-Efficient Fine-Tuning (PEFT/LoRA) weight matrices.
5. TensorFlow Datasets (TFDS): For low-overhead, multi-threaded parallel data loading from comoZ/reasoning-dataset.

## Contributing🤝
Contributions are welcome! If you would like to test alternative multi-dimensional rubrics, adjust the group scaling configurations ($G$), or implement token-efficiency penalties, please follow these steps:
1. Fork the repository.
2. Create a new branch: git checkout -b feature/your-feature-name.
3. Make your changes and commit: git commit -m 'Add some feature'.
4. Push to the branch: git push origin feature/your-feature-name.
5. Submit a pull request.
### open and run the notebooks/Rubric based Reward System for-structured-reasoning.ipynb directly in your Kaggle/Colab TPU environment.

### (Note: Requirements include specific versions of torch, torch_xla, transformers, trl, and jax tailored for TPU acceleration).
