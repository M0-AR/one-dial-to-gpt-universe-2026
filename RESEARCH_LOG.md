# RESEARCH_LOG — online voting, one search at a time (2026-10-06)

Rule followed: sequential websearch (no parallel), 429 backoff, DuckDuckGo fallback, then MCP families with distinct keywords.

## 1. exa/websearch (session provider)
- `reproducible ML repo best practices 2026 Docker Compose benchmark` → computational-reproducibility-pmc-docker, Cresset MLOps, pythainer JOSS, MLReplicate22 (ICML25 → 45 manuscripts, 59% fabricated), MLS-Bench 140 tasks, Bencher RPC. Basis for Dockerfile+compose+seed policy.
- `Karpathy nanoGPT TinyStories next token loss scaling 2026` → nanoGPT deprecated→nanochat, GPT-2-124M 4d 8×A100 loss 2.85, scaling_laws.ipynb C≈6ND, MiniGPT 2026-05-17 (0.83M 1.7236, 10.77M 1.4780). Anchored exps 08/09.

## 2. searxng
- `perceptron Rosenblatt 1958 single neuron gradient descent` → network error (logged; fell back to wiki+duckduckgo).
- suggestions `transformer attention from scratch nanoGPT` → [] (empty = no signal, moved on).

## 3. openresearch (9 tools, distinct keywords)
- web_search `AlexNet ImageNet 60M GPU 2012` → 60M/650k/2×GTX580/2012-09-30 confirmed (Wikipedia+Pinecone+NeurIPS PDF).
- search_openalex `scaling laws Kaplan Hoffmann Chinchilla` → Kaplan N∝C^0.73 vs Chinchilla ∝C^0.50, embedding-count reconciliation (Pearce 2024), inference-aware laws.
- search_hacker_news `build GPT from scratch tiny Shakespeare nanoGPT` → MicroGPT-C (1 pt).
- search_news `LLM training compute scaling 2026` → Bain $6T infra gap (2026-10-01) — context for secret sizes.
- search_stackoverflow `pytorch MLP two spirals backprop vanishing 2026` → no results (logged; used docs instead).
- live: get_crypto_price BTC 85980, get_fx_rate USD→EUR 0.89254/GBP 0.75616/JPY 158.23 (2026-10-05), get_current_date 2026-10-06T10:09:41Z.
- sec_filings/bluesky/indicators skipped as off-topic after 2 relevance checks (logged to avoid noise).

## 4. paper-search (22 connectors, distinct keywords)
- search_arxiv `Attention Is All You Need` → DiNA, attention-optional ViT 74.9% (Melas-Kyriazi), EGA+MoPE TinyShakespeare +0.119, “All You Need” meme 717 titles.
- search_papers unified `word2vec king man woman 2013` (arxiv+semantic+openalex+crossref) → Mikolov N13-1090 (4,011 cites), Drozd beyond-king-queen, BioConceptVec drug-gene GPT-4 parity.
- search_semantic `backprop Rumelhart Hinton 1986` → Nature 323533a0 (32,582 cites) + PDP chapter (21,350).
- search_google_scholar `InstructGPT RLHF` → Ouyang 2022 + Lambert 2025 + RLHF-Deciphered.

## 5. duckduckgo
- search `two spirals MLP 26 params` → OpenANN 2-20-20-1 solves, 3-6-1 minimal, Lang 1988 shortcut connections. Directly set capacity probe.

## 6. agent-reach
- search web `InstructGPT RLHF` → 51KB dump (truncated, kept as provenance; key: demos+rankings→fine-tune).
- stock_quote AAPL 332.89 (-0.24%), vol 34M, live via yfinance.

## 7. gitmcp
- fetch nanoGPT docs + code search `causal self-attention` → socket closed (logged, infra down; used websearch DeepWiki instead).

## 8. kaggle
- search_everything `tiny Shakespeare` → Karpathy GPT-from-scratch notebooks.
- discussions_search `CNN ImageNet` → 2 Imagenet-summary threads.
- datasets_list `cifar-10` → sort_by error (API mismatch, logged).
- kernels_list `GPT from scratch` → Transformer-From-Scratch 294 votes etc.

## 9. wiki
- search `Perceptron Rosenblatt 1958` → Mark I Perceptron, Perceptron, Frank Rosenblatt (1928–1971), Multilayer perceptron.

## 10. gsd + superpowers
- gsd_websearch `live market LLM verification Yahoo` → empty (logged).
- semantic_search_skills `reproducible research benchmarking` → systematic-debugging, writing-skills.

**Voting outcome:** history anchors kept only if ≥2 sources agreed (table in README §2). Negative signals (empty/error) logged, not hidden. No file written before its supporting execution (see git log).
