# Stage 1 Fine-Tuning — draft for the FINAL paper (not in the submitted prelim)

Result of the SmolVLM-2B LoRA fine-tune (GCP T4, leave-one-out on the 20-example
Wixárika pilot). Numbers from `outputs/smolvlm_{base,loo}.log`.

## Raw numbers
- Off-the-shelf SmolVLM-2B: **16.72 ± 3.60** ChrF++ (Spanish pilot proxy, n=20)
- SmolVLM-2B + LoRA (LOO): **17.55 ± 6.52** ChrF++ → **Δ +0.83 (within noise)**
- Per-fold LOO range: **8.2 – 29.7** (SD dwarfs the gap)
- Qwen2.5-VL generic proxy (prelim ref.): **21.0**
- LoRA: r=16, α=32, dropout=0.05, lr=2e-4; 9.3M trainable params (0.41% of 2B); caption-only loss; SigLIP frozen.

## Drop-in LaTeX (paragraph + table)

```latex
\subsection{Fine-Tuning Stage 1 under Extreme Data Scarcity}
\label{sec:stage1-ft}

To test whether adapting the Stage 1 decoder to in-domain cultural captions
improves the Spanish intermediate, we LoRA fine-tune SmolVLM-2B
\citep{marafioti2025smolvlm}: we freeze its SigLIP vision encoder
\citep{zhai2023siglip} and train rank-16 adapters \citep{hu2022lora}
($\alpha{=}32$, dropout $0.05$, lr $2\times10^{-4}$; 9.3M trainable
parameters, $0.41\%$ of the model) on the image$\rightarrow$Spanish-caption
objective, with loss computed over the caption tokens only. The only split
carrying Spanish references is the Wix\'arika pilot ($n{=}20$), so we both
train and evaluate there via leave-one-out cross-validation (LOO): for each
example we fine-tune on the other 19 and score the held-out caption's ChrF++
against its Spanish gold. To separate the effect of fine-tuning from the choice
of backbone, we compare against the identical SmolVLM-2B with no adapter.

Table~\ref{tab:stage1-ft} reports the outcome. LoRA fine-tuning reaches
$17.55 \pm 6.52$ ChrF++ against $16.72 \pm 3.60$ for the off-the-shelf model:
a $+0.83$ difference that is indistinguishable from noise, as per-fold scores
span $8.2$ to $29.7$ and the standard deviation dwarfs the gap. Both
configurations also trail the off-the-shelf Qwen2.5-VL \citep{qwen2025qwen25vl}
generic proxy ($21.0$; Table~\ref{tab:stage1-proxy}) by roughly four points.
We read this as a direct instantiation of RQ2: at 19 training examples,
parameter-efficient fine-tuning cannot extract a stable gain, and the model
small enough to fine-tune on our budget underperforms the larger
general-purpose VLM it was meant to specialize. Fine-tuning is therefore not a
viable Stage 1 lever in this data regime; the culturally-aware improvements we
do observe come from prompting (Section~\ref{sec:cultural-vqa}) rather than
weight adaptation.

\begin{table}[t]
\centering
\begin{tabular}{lc}
\hline
\textbf{Stage 1 model} (Wix\'arika pilot, $n{=}20$) & \textbf{ChrF++} \\
\hline
SmolVLM-2B, off-the-shelf        & $16.72 \pm 3.60$ \\
SmolVLM-2B + LoRA (LOO)          & $17.55 \pm 6.52$ \\
Qwen2.5-VL, generic (proxy ref.) & $21.0$ \\
\hline
\end{tabular}
\caption{Stage 1 fine-tuning on the Spanish pilot proxy. LoRA's $+0.83$ over the
off-the-shelf model is within noise (per-fold range $8.2$–$29.7$); both SmolVLM
variants trail the larger Qwen2.5-VL.}
\label{tab:stage1-ft}
\end{table}
```

## Wiring notes
- Point `\ref{tab:stage1-proxy}` and `\ref{sec:cultural-vqa}` at the real labels in the final paper.
- Keep the `21.0` proxy figure consistent with the generic-baseline number used elsewhere.
- Citation keys (`marafioti2025smolvlm`, `zhai2023siglip`, `hu2022lora`, `qwen2025qwen25vl`) already exist in `acl2023/custom.bib`.
