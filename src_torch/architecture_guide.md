# src_torch 架构详解（含核心概念深度解析）

## 一、项目结构

```
src_torch/
├── modules/
│   ├── layers.py          # 基础层: RMSNorm, RoPE, Attention, SwiGLUFFN, FinalLayer, TimestepEmbedder, BottleneckTextProj
│   ├── model.py           # 主模型: ELFBlock, ELF transformer, 工厂函数 ELF_B/M/L
│   └── t5_encoder.py      # HuggingFace T5EncoderModel 冻结文本编码器
├── utils/
│   ├── logging_utils.py   # rank-0 日志
│   ├── encoder_utils.py   # encode_text + mask 构建
│   ├── sampling_utils.py  # 噪声 / 时间步 / flow-matching 去噪采样
│   ├── data_utils.py      # DataLoader, collate, 数据集加载
│   ├── train_utils.py     # TrainState, 优化器, LR schedule
│   ├── checkpoint_utils.py# 保存/加载 checkpoint (本地 + HF Hub)
│   ├── generation_utils.py# 生成辅助 (ODE/SDE 步, 解码)
│   └── metrics_utils.py   # PPL, BLEU, ROUGE
├── optimizers/
│   └── muon.py            # PyTorch Muon 优化器
├── configs/
│   ├── config.py          # Config / SamplingConfig 定义
│   ├── sampling_configs/  # 采样配置 YAML
│   └── training_configs/  # 训练配置 YAML
├── train_step.py          # 单步训练函数
├── train.py               # 主训练入口 (DDP)
├── generation.py          # 生成/评估运行器
├── eval.py                # 评估入口
└── requirements.txt
```

## 二、训练流程 (train.py → train_step.py)

### 启动方式

```bash
torchrun --nproc_per_node=8 train.py \
    --config configs/training_configs/train_owt_ELF-B_h800_torch.yml
```

### train.py 主循环流程

```
1. 初始化 DDP (NCCL backend)
2. 加载 Config (YAML)
3. 加载 Tokenizer (AutoTokenizer, t5-small)
4. 加载训练数据集 (HF Hub / 本地 Arrow)
5. 创建冻结 T5 Encoder (HuggingFace T5EncoderModel)
6. 创建 ELF 模型 (根据 config.model 选 ELF_B/M/L)
7. 创建 Optimizer (Muon 或 AdamW)
8. 创建 LR Scheduler (LambdaLR, cosine/constant + warmup)
9. 创建 TrainState (model + optimizer + EMA params)
10. 可选: 从 checkpoint 恢复
11. FOR each epoch:
    FOR each batch from DataLoader:
        a. prepare_batch() → 将 numpy 数据移到 GPU, 生成 label_drop_mask
        b. train_step() → 核心训练步骤 (见下文)
        c. 累积梯度 (grad_accum_steps > 1 时)
        d. optimizer.step() + scheduler.step()
        e. EMA update (ema_decay=0.9999)
        f. 日志记录 (wandb / tqdm)
        g. 可选: checkpoint 保存
        h. 可选: generation 评估
```

### train_step.py 详细流程

```
train_step(model, encoder, optimizer, batch, config, ...):
    │
    ├─ 1. 编码输入 token → x0 (latent)
    │     input_ids        (B, S)         token id 序列
    │     encoder_attn_mask (B, S) or (B, S, S)  注意力 mask
    │     → encode_text(input_ids, encoder_attn_mask, encoder, latent_mean, latent_std)
    │     → x0             (B, S, D_model=512)   T5 编码并归一化后的 latent
    │
    ├─ 2. 采样时间步 t
    │     t = sample_timesteps(B, P_mean, P_std, time_schedule)
    │     → t               (B,)                每个样本一个时间步, 范围 [0,1]
    │
    ├─ 3. 生成噪声
    │     noise = torch.randn_like(x0)
    │     → noise            (B, S, D_model)     纯随机噪声
    │
    ├─ 4. 构造 loss_mask
    │     loss_mask = attention_mask * (1 - cond_seq_mask)
    │     → loss_mask        (B, S)              只在非 cond 且有效的位置计算 loss
    │
    ├─ 5. Flow-matching 加噪: z = t*x0 + (1-t)*noise*scale
    │     cond_seq_mask      (B, S, 1)           cond token 位置 = 1
    │     z = add_noise(x0, noise, t, config, cond_seq_mask)
    │     → z               (B, S, D_model)      加噪后的输入; cond 位置保持 x0
    │
    ├─ 6. Label drop (可选): 对 cond token 位置的 z 和 x0 清零
    │     → x0_for_target   (B, S, D_model)      label drop 后的 x0
    │
    ├─ 7. 随机选择分支
    │     decoder_step_active = (torch.rand(1).item() < decoder_prob)
    │     True → decoder 分支 (CE loss)
    │     False → denoiser 分支 (L2 loss)
    │
    ├─ ─── IF decoder_step_active (Decoder CE 分支) ───
    │     │
    │     ├─ 7a. 构造 decoder 输入
    │     │     decoder_lambda_t = sigmoid(N(P_mean, P_std))
    │     │     → decoder_lambda_t  (B, S, 1)    logit-normal 采样的混合系数
    │     │     decoder_noise       (B, S, D_model)  额外噪声
    │     │     decoder_z = decoder_lambda_t * x0_for_target + (1-decoder_lambda_t) * decoder_noise
    │     │     → decoder_z         (B, S, D_model)  decoder 分支的输入 latent
    │     │
    │     ├─ 7b. 模型前向 (decoder 模式)
    │     │     decoder_t = ones(B,)              decoder 分支始终用 t=1
    │     │     decoder_input = [decoder_z, zeros] (self-cond 时) 或 decoder_z
    │     │     → decoder_input    (B, S, 2*D_model) or (B, S, D_model)
    │     │     _, decoder_logits = model(decoder_input, decoder_t,
    │     │                                  self_cond_cfg_scale, decoder_step_active=True)
    │     │     → decoder_logits   (B, S, vocab_size)   vocab 上的 logits
    │     │
    │     ├─ 7c. CE loss
    │     │     log_probs = F.log_softmax(decoder_logits, dim=-1)
    │     │     ce = -log_probs.gather(dim=-1, index=decoder_targets.unsqueeze(-1)).squeeze(-1)
    │     │     → ce               (B, S)               每个位置的交叉熵
    │     │     ce_loss = reduce_token_loss(ce, loss_mask)
    │     │     → ce_loss          scalar                masked mean CE loss
    │
    ├─ ─── IF NOT decoder_step_active (Denoiser L2 分支) ───
    │     │
    │     ├─ 7d. Self-conditioning 输入构造
    │     │     (1) 先用 [z, zeros] 前向得到初始估计 x_pred_init (no_grad)
    │     │     (2) x_pred_cond = x_pred_init * use_self_cond_mask
    │     │     (3) z_input = [z, x_pred_cond]  (self-cond 时) 或 z
    │     │     → denoiser_input  (B, S, 2*D_model) or (B, S, D_model)
    │     │
    │     ├─ 7e. 模型前向 (denoiser 模式)
    │     │     net_out, _ = model(denoiser_input, t,
    │     │                         self_cond_cfg_scale, decoder_step_active=False)
    │     │     → net_out         (B, S, D_model)      模型预测的 denoised output
    │     │     → decoder_logits  None 或 zeros (denoiser 分支不计算)
    │     │
    │     ├─ 7f. 计算 velocity
    │     │     v_pred = (net_out - z) / clamp(1-t, t_eps)
    │     │     → v_pred          (B, S, D_model)      预测的速度场
    │     │
    │     ├─ 7g. v_target (含 self-cond CFG guidance, 可选)
    │     │     v_target = (x0_for_target - z) / clamp(1-t, t_eps)
    │     │     → v_target        (B, S, D_model)      真实速度场
    │     │     (可选) v_final_target = v_target + sc_guidance.detach()
    │     │     → v_final_target  (B, S, D_model)      加了 self-cond guidance 的目标
    │     │
    │     ├─ 7h. L2 loss
    │     │     per_dim_loss = (v_pred - v_final_target)²
    │     │     → per_dim_loss    (B, S, D_model)       每维的平方误差
    │     │     l2_loss = reduce_token_loss(per_dim_loss.mean(dim=-1), loss_mask)
    │     │     → l2_loss         scalar                 masked mean L2 loss
    │
    ├─ 8. Backward + optimizer step
    │     scaled_loss = loss / grad_accum_steps
    │     scaled_loss.backward()
    │     nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    │     optimizer.step()
    │
    └─ 9. Return metrics
        → {"loss": float, "l2_loss": float, "ce_loss": float}
```

---

## 三、模型结构 (modules/model.py + modules/layers.py)

### 3.1 整体架构: ELF Transformer

```
ELF Model 结构:

输入 z (B, S, C 或 B, S, 2C)
    │
    ├─ Self-conditioning merge (如果 C == 2*D_model)
    │   self_cond_proj: Linear(2*D_model → D_model)
    │   → x (B, S, D_model=768)
    │
    ├─ Text projection (Bottleneck)
    │   BottleneckTextProj: D_model → bottleneck_dim → hidden_size
    │   → x (B, S, hidden_size=768)
    │
    ├─ Model-mode tokens (可选, 4 个)
    │   mode_tokens: Parameter(1, 4, hidden_size) * gate
    │   gate = 1.0 (decoder_step_active=True) / 0.0 (False) / 0.0 (None)
    │   → prepend mode_tokens to x: (B, 4+S, hidden_size)
    │
    ├─ Prefix time + CFG tokens
    │   t_embedder: TimestepEmbedder → time_emb (B, hidden_size)
    │   t_emb_tokens: Parameter(1, num_time_tokens=4, hidden_size) + time_emb
    │   → t_prefix (B, 4, hidden_size)
    │   (可选) self_cond_cfg_embedder → sc_prefix (B, 4, hidden_size)
    │   → prepend to x: (B, prefix_len + S, hidden_size)
    │
    ├─ RoPE (TextRotaryEmbeddingFast)
    │   dim = head_dim = hidden_size / num_heads
    │   pt_seq_len = max_length
    │   num_empty_token = prefix_len + model_mode_offset
    │   → prefix/mode tokens 不旋转, 后续 token 正常旋转
    │
    ├─ Transformer Blocks × depth (12 for ELF-B)
    │   每个块:
    │   ┌──────────────────────────────┐
    │   │ x = x + Attention(RMSNorm(x))│  ← pre-norm attention
    │   │ x = x + SwiGLUFFN(RMSNorm(x))│  ← pre-norm FFN
    │   └──────────────────────────────┘
    │   中间层 (depth//4 到 3*depth//4) 使用 attn_drop 和 proj_drop
    │
    ├─ Strip prefix/mode tokens
    │   → x (B, S, hidden_size)
    │
    ├─ Decoder unembedding (可选, vocab_size > 0 且 decoder_step_active != None)
    │   if decoder_step_active=True:
    │       hidden_proj = GELU(x @ proj_kernel + proj_bias)
    │       → hidden_proj (B, S, text_encoder_dim=512)
    │       decoder_logits = hidden_proj @ unembed_kernel + unembed_bias
    │       → decoder_logits (B, S, vocab_size=32128)
    │   elif decoder_step_active=False:
    │       decoder_logits = zeros(B, S, vocab_size)
    │   else (None): decoder_logits = None
    │
    └─ FinalLayer: RMSNorm → Linear(hidden_size → 1*1*D_model)
       → output (B, S, text_encoder_dim=512)    ← 最终的 denoised latent

返回: (output, decoder_logits)
```

### 3.2 ELF 配置 (工厂函数)

| 模型 | depth | hidden_size | num_heads | mlp_ratio |
|---|---|---|---|---|
| ELF-B | 12 | 768 | 12 | 4.0 |
| ELF-M | 24 | 1056 | 16 | 4.0 |
| ELF-L | 32 | 1280 | 16 | 4.0 |

### 3.3 关键子模块详解

#### Attention (modules/layers.py)

```
输入:
  x               (B, N, C)          hidden states
  rope_fn         TextRotaryEmbeddingFast  可选的 RoPE 函数
  attention_mask  (B, N) or (B, N, N)  1=valid, 0=padded

内部流程:
  1. QKV projection: Linear(C → 3C)
     → qkv             (B, N, 3, num_heads, head_dim)
     → permute          (3, B, num_heads, N, head_dim)
     → q, k, v          各 (B, num_heads, N, head_dim)

  2. QK-norm (可选): RMSNorm on q and k
     → q, k             各 (B, num_heads, N, head_dim)

  3. RoPE: rope_fn(q), rope_fn(k)
     → q, k             各 (B, num_heads, N, head_dim)  旋转后

  4. Mask 转换: 1/0 → 0/-1e9 additive mask
     → sdpa_mask        (B, 1, 1, N) or (B, 1, N, N)

  5. F.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask)
     使用 FlashAttention-2 (H800 sm_90 自动启用)
     → x                (B, num_heads, N, head_dim)

  6. Output projection + Dropout
     → x                (B, N, C)

初始化: xavier_uniform for qkv/proj weights, zeros for biases
```

#### SwiGLUFFN (modules/layers.py)

```
输入:
  x               (B, N, C)

内部流程:
  1. w12: Linear(C → 2*hidden_dim*2/3)
     → x12              (B, N, 2*actual_hidden)
     → split: x1, x2    各 (B, N, actual_hidden)

  2. SwiGLU: silu(x1) * x2
     → hidden            (B, N, actual_hidden)

  3. Dropout

  4. w3: Linear(actual_hidden → C)
     → output            (B, N, C)

其中: actual_hidden = int(mlp_ratio * C * 2 / 3)
例: ELF-B, mlp_ratio=4, C=768 → actual_hidden = int(3072*2/3) = 2048
```

#### TimestepEmbedder (modules/layers.py)

```
输入:
  t               (B,)               timestep 值, 范围 [0,1]

内部流程:
  1. timestep_embedding(t, dim=256)
     → sinusoidal emb   (B, 256)         正弦/余弦位置编码

  2. mlp_0: Linear(256 → hidden_size)
  3. SiLU activation
  4. mlp_2: Linear(hidden_size → hidden_size)
     → time_emb         (B, hidden_size)

初始化: normal(std=0.02) for mlp_0/mlp_2 weights
```

#### TextRotaryEmbeddingFast (modules/layers.py)

```
输入:
  t               (B, num_heads, S, head_dim)  需要旋转的 tensor

内部流程:
  1. 计算频率: freqs = 1/(θ^(arange(0,dim,2)/dim))   (dim//2,)
  2. 计算位置: pos = arange(ft_seq_len) / ft_seq_len * pt_seq_len  (ft_seq_len,)
  3. 外积: freqs_main = einsum(pos, freqs)           (ft_seq_len, dim)
  4. repeat: freqs_main = repeat(freqs_main, 'n → (n r)', r=2) (ft_seq_len, dim)
  5. 拼接: cos/sin parts + empty token 的 cos=1, sin=0
     → freqs_cos        (total_len, dim)
     → freqs_sin        (total_len, dim)
  6. 应用: t * freqs_cos + rotate_half(t) * freqs_sin

特性:
  - 无可学习参数
  - prefix/mode token 位置不旋转 (cos=1, sin=0)
  - 支持 num_empty_token 参数控制不旋转的位置数量
```

#### BottleneckTextProj (modules/layers.py)

```
输入:
  x               (B, S, text_encoder_dim=512)

流程:
  1. proj1: Linear(text_encoder_dim → bottleneck_dim=128, bias=False)
  2. proj2: Linear(bottleneck_dim → hidden_size=768, bias=True)
     → output            (B, S, hidden_size=768)

初始化: xavier_uniform for proj1/proj2 weights, zeros for proj2 bias
```

#### FinalLayer (modules/layers.py)

```
输入:
  x               (B, S, hidden_size=768)

流程:
  1. RMSNorm(hidden_size)
  2. Linear(hidden_size → patch_size * patch_size * out_channels)
     其中 patch_size=1, out_channels=text_encoder_dim=512
     → output            (B, S, text_encoder_dim=512)

初始化: 全零初始化 (weight=0, bias=0) ← 关键: 训练开始时 output ≈ 0
```

### 3.4 T5 Encoder (modules/t5_encoder.py)

```
使用 HuggingFace transformers.T5EncoderModel.from_pretrained("t5-small")
冻结所有参数 (requires_grad_=False, eval 模式)

输入:
  input_ids       (B, S)             token id 序列
  attention_mask  (B, S) or (B, S, S) 1=valid, 0=pad

输出:
  last_hidden_state (B, S, d_model=512)

配置:
  t5-small: d_model=512, num_layers=6, num_heads=8, d_kv=64
  t5-base:  d_model=768, num_layers=12, num_heads=12
  t5-large: d_model=1024, num_layers=24, num_heads=16
```

### 3.5 ELF 模型完整输入输出

```python
def forward(self, x, t, attention_mask, self_cond_cfg_scale, decoder_step_active):
    """
    输入:
      x                   (B, S, text_encoder_dim)   编码后的 latent
                          或 (B, S, 2*text_encoder_dim)  self-cond 时 concat
      t                   (B,)                       时间步 [0, 1]
      attention_mask      (B, S)                     1=valid, 0=pad
      self_cond_cfg_scale (B,) or None               self-cond CFG 的 scale 值
      decoder_step_active bool or None               True=计算 decoder logits
                                                    False=生成零 logits
                                                    None=不计算 decoder

    输出:
      output              (B, S, text_encoder_dim)   denoised latent
      decoder_logits      (B, S, vocab_size) or None 解码 logits
    """
```

---

## 四、去噪过程 (sampling_utils.py + generation_utils.py)

### 4.1 Flow-matching 基本原理

ELF 使用 **Flow-matching** (连续归一化流) 而不是传统的扩散模型:

- **前向过程 (加噪)**: `z = t * x0 + (1-t) * noise * scale`
  - t=0 时 z = noise (纯噪声)
  - t=1 时 z = x0 (干净数据)
  - 中间 t 是线性插值

- **速度场**: `v = dx/dt = (x0 - z) / (1 - t)`
  - v 描述了从噪声到数据的方向和速度

- **ODE 采样**: `z_next = z + (t_next - t) * v_pred`
  - 从 t=0 到 t=1, 逐步沿速度场积分

### 4.2 add_noise — 训练时的加噪

```python
def add_noise(x0, noise, t, config, cond_seq_mask=None):
    """
    输入:
      x0            (B, S, D_model)     T5 编码的干净 latent
      noise         (B, S, D_model)     随机噪声
      t             (B,)                时间步 [0,1]
      config                           含 denoiser_noise_scale
      cond_seq_mask (B, S, 1)           1=conditioning token

    流程:
      t_exp = t.reshape(-1, 1, 1)       → (B, 1, 1)
      z = t_exp * x0 + (1 - t_exp) * noise * scale  → (B, S, D_model)
      # cond 位置保持干净: z = cond_mask * x0 + (1-cond_mask) * z

    输出:
      z             (B, S, D_model)     加噪后的 latent
    """
```

### 4.3 时间步采样

```python
def sample_timesteps(batch_size, P_mean, P_std, time_schedule, device):
    """
    logit_normal: z = Normal(P_mean, P_std); t = sigmoid(z)
      → 偏向中间时间步 (P_mean=-0.8 时, t 倾向 ~0.3)
    uniform: t = Uniform(0, 1)
      → 均匀分布

    输出:
      t             (B,)                时间步 [0,1]
    """
```

### 4.4 net_out_to_v_x — 模型输出 → velocity + denoised

```python
def net_out_to_v_x(net_out, z, t, t_eps=5e-2):
    """
    输入:
      net_out       (B, S, D_model) 或 tuple(output, decoder_logits)
      z             (B, S, D_model)     加噪输入
      t             (B,)                时间步

    流程:
      如果 net_out 是 tuple, 取 net_out[0] (丢弃 decoder_logits)
      x = net_out                          → (B, S, D_model) 模型预测的 denoised 结果
      v = (x - z) / clamp(1 - t, t_eps)   → (B, S, D_model) 从模型输出推导的速度场
      # t_eps=5e-2 防止 t→1 时分母为零

    输出:
      v             (B, S, D_model)     预测速度
      x             (B, S, D_model)     预测的 denoised latent
    """
```

### 4.5 ODE 步 — 确定性采样

```python
def _ode_step(model, z, t, t_next, x_pred_prev, config,
              cfg_scale, self_cond_cfg_scale, cond_seq, cond_seq_mask):
    """
    输入:
      z             (B, S, D_model)     当前 noisy latent
      t             float               当前时间步 (如 0.0)
      t_next        float               下一个时间步 (如 0.03)
      x_pred_prev   (B, S, D_model) or None  上一步的 denoised 预测 (用于 self-cond)
      cfg_scale     float               CFG 放大系数 (1.0 = 不用 CFG)
      self_cond_cfg_scale float         self-cond CFG scale

    流程:
      1. 构造 t_batch = full(B, t)    → (B,)
      2. _forward_sample(z, t_batch, x_pred_prev, ...)
         → v_pred, x_pred             各 (B, S, D_model)
      3. z_next = z + (t_next - t) * v_pred  → (B, S, D_model)

    输出:
      z_next        (B, S, D_model)     一步后的 noisy latent
      x_pred        (B, S, D_model)     本步的 denoised 预测
    """
```

### 4.6 SDE 步 — 随机采样 (比 ODE 多一步噪声扰动)

```python
def _sde_step(model, z, t, t_next, x_pred_prev, config,
              cfg_scale, self_cond_cfg_scale, cond_seq, cond_seq_mask, gamma):
    """
    输入:
      gamma         float               SDE churn 参数
                                        gamma=0 → 退化成 ODE
                                        gamma>0 → 加随机噪声扰动

    流程:
      1. h = t_next - t
      2. alpha = clamp(1 - gamma * h, 0, 1)    信号保留比例
      3. t_back = alpha * t                     退回的时间步
      4. eps = randn_like(z) * noise_scale      新随机噪声
      5. z_back = alpha * z + (1-alpha) * eps   加噪退回
         (cond 位置仍保持干净)
      6. _forward_sample(z_back, t_batch=t_back, ...)
         → v_pred, x_pred
      7. z_next = z_back + (t_next - t_back) * v_pred

    直觉:
      - 先向后退一小步 (加噪声), 再向前积分一大步
      - gamma 越大, 退步越多, 随机性越大
      - gamma=0 时 alpha=1, t_back=t, 就是没有退步, 纯 ODE

    输出:
      z_next        (B, S, D_model)
      x_pred        (B, S, D_model)
    """
```

### 4.7 _forward_sample — 含 Self-cond + CFG 的前向

```python
def _forward_sample(model, z, t_batch, x_pred_prev, config,
                    cfg_scale, self_cond_cfg_scale, cond_seq, cond_seq_mask):
    """
    流程:
      1. _forward_sample_self_cond(z, t, x_pred_prev, ...)
         → v_cond, x_cond           条件下的 velocity/denoised

      2. IF cfg_scale == 1.0:
           直接返回 v_cond, x_cond   (不需要 CFG)

      3. ELSE: 计算无条件分支
         z_uncond = restore_cond(z, zeros, cond_seq_mask)
         # cond token 位置清零 → 无条件输入
         x_pred_prev_uncond = 同样清零

         _forward_sample_self_cond(z_uncond, t, x_pred_prev_uncond, ...)
         → v_uncond, x_uncond       无条件下的 velocity/denoised

      4. CFG 组合:
         v_out = v_uncond + cfg_scale * (v_cond - v_uncond)
         x_out = x_uncond + cfg_scale * (x_cond - x_uncond)
         # cfg_scale > 1 → 放大条件与无条件的差异

      5. restore_vx: cond 位置恢复为干净值
         → v_out, x_out

    输出:
      v_out         (B, S, D_model)     CFG 后的速度
      x_out         (B, S, D_model)     CFG 后的 denoised
    """
```

### 4.8 _forward_sample_self_cond — Self-conditioning 前向

```python
def _forward_sample_self_cond(model, z, t, x_pred_prev, config,
                               self_cond_cfg_scale, cond_seq, cond_seq_mask):
    """
    三种模式:

    ── 模式 1: num_self_cond_cfg_tokens > 0 (模型内部有 self-cond CFG token)
       x_pred_prev = None → 初始估计为零
       z_input = cat([z, x_pred_prev], dim=-1)  → (B, S, 2*D_model)
       sc_scale_batch = full(B, self_cond_cfg_scale) → (B,)
       net_out = model(z_input, t, self_cond_cfg_scale=sc_scale_batch)
       → v, x

    ── 模式 2: self_cond_prob == 0 (不做 self-cond)
       net_out = model(z, t)  ← 最简单, 直接前向
       → v, x

    ── 模式 3: self_cond_prob > 0 且 num_self_cond_cfg_tokens == 0
       (经典 self-cond CFG, 两次前向)
       ① 无条件估计: model(cat([z, zeros], dim=-1), t)
          → v_uncond, x_uncond
       ② 条件估计: model(cat([z, x_uncond], dim=-1), t)
          → v_cond, x_cond
       ③ CFG 混合:
          v_out = v_uncond + self_cond_cfg_scale * (v_cond - v_uncond)
          x_out = x_uncond + self_cond_cfg_scale * (x_cond - x_uncond)

    所有模式最后: restore_vx → cond 位置恢复干净
    """
```

### 4.9 生成循环 (generation_utils.py → generate_samples)

```python
def generate_samples(model, z, t_steps, config, sampling_config,
                     cfg_scale, self_cond_cfg_scale, cond_seq, cond_seq_mask):
    """
    输入:
      z             (B, S, D_model)     初始噪声 (纯随机 * noise_scale)
      t_steps       (n_steps+1,)        时间步序列 [0, ..., 1]

    流程:
      z = restore_cond(z, cond_seq, cond_mask)  # cond 位置替换为干净值
      x_pred = restore_cond(zeros, ...)          # 初始估计为零

      FOR i in range(n_pairs):  # n_pairs = len(t_steps) - 2
          t_cur = t_steps[i]
          t_nxt = t_steps[i+1]
          IF method == "sde":
              z, x_pred = _sde_step(z, t_cur, t_nxt, x_pred, ..., gamma)
          ELSE:
              z, x_pred = _ode_step(z, t_cur, t_nxt, x_pred, ...)

      # 最后一步始终用 ODE (保证确定性收敛)
      z, _ = _ode_step(z, t_steps[-2], t_steps[-1], x_pred, ...)

    输出:
      z             (B, S, D_model)     最终 latent (≈ x0 at t=1)
    """
```

### 4.10 解码 latent → token ids (generation_utils.py → decode_latent_to_ids)

```python
def decode_latent_to_ids(z, model, t_final_val, config, self_cond_cfg_scale):
    """
    输入:
      z             (B, S, D_model)     生成得到的 latent
      t_final_val   float               ≈ 1.0

    流程:
      1. t_final = full(B, t_final_val)
      2. z_input = cat([z, zeros_like(z)]) if self_cond_prob > 0 else z
      3. _, decoder_logits = model(z_input, t_final,
                                    self_cond_cfg_scale=sc_scale_batch,
                                    decoder_step_active=True)
      4. predicted_ids = decoder_logits.argmax(dim=-1)

    输出:
      predicted_ids  (B, S)             解码得到的 token id 序列
    """
```

---

## 五、Loss 详细解析 (train_step.py)

### 5.1 变量定义总表

| 变量名 | Shape | 含义 |
|---|---|---|
| `input_ids` | (B, S) | 原始 token id 序列 |
| `encoder_attention_mask` | (B, S, S) | 自注意力 mask: cond 看 cond, x 看 cond+x |
| `attention_mask` | (B, S) | 1=有效 token, 0=padding |
| `cond_seq_mask` | (B, S) | 1=conditioning 前缀 token, 0=target token |
| `label_drop_mask` | (B,) | True=该样本丢弃 condition label |
| `x0` | (B, S, 512) | T5 编码 + 归一化后的干净 latent |
| `x0_for_target` | (B, S, 512) | label drop 处理后的 x0 (cond 位置可能清零) |
| `t` | (B,) | 采样的时间步, logit-normal 或 uniform 分布, [0,1] |
| `noise` | (B, S, 512) | 纯随机噪声 (randn) |
| `cond_seq_mask` (unsqueeze) | (B, S, 1) | 用于 z 中 cond 位置保持干净的 mask |
| `loss_mask` | (B, S) | 只在 "有效且非 cond" 的位置计算 loss |
| `z` (denoiser_z) | (B, S, 512) | flow-matching 加噪后的 latent: z = t*x0 + (1-t)*noise*scale |
| `decoder_targets` | (B, S) | CE loss 的目标 token id (= input_ids) |
| `decoder_step_active` | bool | True=CE 分支, False=L2 分支 |
| `decoder_lambda_t` | (B, S, 1) | logit-normal 采样的 decoder 混合系数 |
| `decoder_noise` | (B, S, 512) | decoder 分支的额外噪声 |
| `decoder_z` | (B, S, 512) | decoder 输入: lambda*x0 + (1-lambda)*noise |
| `t_expanded` | (B, 1, 1) | t 的 broadcast 形式 |
| `v_target` | (B, S, 512) | 真实速度: (x0 - z) / max(1-t, t_eps) |
| `use_self_cond_mask` | (B, 1, 1) or None | self-cond 使用概率 mask |
| `self_cond_cfg_scale` | (B,) or None | self-cond CFG 的 scale 值 |
| `net_out` | (B, S, 512) | 模型输出的 denoised prediction |
| `v_pred` | (B, S, 512) | 预测速度: (net_out - z) / max(1-t, t_eps) |
| `decoder_logits` | (B, S, vocab_size) | decoder 分支输出的 logits |
| `per_dim_loss` | (B, S, 512) | 每维的 (v_pred - v_target)² |
| `l2_loss` | scalar | masked mean L2 loss |
| `ce` | (B, S) | 每位置的交叉熵 |
| `ce_loss` | scalar | masked mean CE loss |

### 5.2 Denoiser 分支 L2 Loss — 逐步详解

```
① velocity target:
   v_target = (x0_for_target - z) / clamp(1 - t, t_eps)
   ┌────────────────────────────────────────────────────┐
   │ 含义: 从当前噪声 z 到干净数据 x0 的方向和速度    │
   │ t 接近 0: v ≈ (x0 - noise*scale), 大速度          │
   │ t 接近 1: v ≈ (x0 - x0)/(t_eps) ≈ 0, 小速度      │
   │ t_eps=5e-2 防止 t→1 时分母爆炸                     │
   │ x0_for_target: label drop 时 cond 位置被清零       │
   └────────────────────────────────────────────────────┘

② Self-conditioning input:
   z_input = get_z_input(z, t, self_cond_cfg_scale)
   ┌────────────────────────────────────────────────────┐
   │ self_cond_prob > 0 时:                             │
   │   先用 [z, zeros_like(z)] 前向得到初始估计         │
   │   x_pred_init = model(cat([z, zeros], -1), t)      │
   │   只对 use_self_cond_mask=1 的样本使用估计          │
   │   z_input = cat([z, x_pred_cond], -1)              │
   │   → (B, S, 2*D_model)                              │
   │ self_cond_prob == 0 时:                             │
   │   z_input = z                                      │
   │   → (B, S, D_model)                                │
   └────────────────────────────────────────────────────┘

③ 模型前向:
   net_out, _ = model(z_input, t, self_cond_cfg_scale,
                      decoder_step_active=False)
   ┌────────────────────────────────────────────────────┐
   │ decoder_step_active=False → decoder_logits=zeros   │
   │ 模型只计算 denoised output, 不计算 vocab logits    │
   │ mode_tokens 的 gate=0 (decoder_step_active=False)   │
   │ → mode_tokens 被清零, 不影响 attention              │
   └────────────────────────────────────────────────────┘

④ 预测 velocity:
   v_pred, _ = net_out_to_v_x(net_out, z, t, t_eps)
   ┌────────────────────────────────────────────────────┐
   │ v_pred = (net_out - z) / clamp(1 - t, t_eps)      │
   │ net_out 是模型对 x0 的估计                          │
   │ v_pred 是从模型输出推导的速度                        │
   └────────────────────────────────────────────────────┘

⑤ Self-cond CFG guidance (可选):
   v_final_target = v_target + sc_guidance.detach()
   ┌────────────────────────────────────────────────────┐
   │ sc_guidance = (1 - 1/sc_w) * (v_cond - v_uncond)  │
   │ 含义: 放大条件和无条件速度的差异                     │
   │ sc_w = self_cond_cfg_scale (从 log-uniform 采样)    │
   │ .detach() → 不让 guidance target 产生梯度          │
   │ 只对 use_self_cond_mask=1 的样本生效               │
   └────────────────────────────────────────────────────┘

⑥ L2 loss:
   per_dim_loss = (v_pred - v_final_target)²
   ┌────────────────────────────────────────────────────┐
   │ per_dim_loss  → (B, S, 512) 每维的平方误差        │
   │ .mean(dim=-1) → (B, S)     每个 token 的平均误差   │
   │ loss_mask     → (B, S)     只在有效非 cond 位置=1  │
   │ l2_loss = Σ(masked_loss) / Σ(loss_mask)           │
   │ → scalar                                           │
   └────────────────────────────────────────────────────┘
```

### 5.3 Decoder 分支 CE Loss — 逐步详解

```
① Decoder 输入构造:
   decoder_z_vals = Normal(decoder_p_mean, decoder_p_std)  → (B*S,)
   decoder_lambda_t = sigmoid(decoder_z_vals)               → (B, S, 1)
   decoder_noise = randn_like(x0) * decoder_noise_scale     → (B, S, 512)
   decoder_z = lambda * x0_for_target + (1-lambda) * noise
   ┌────────────────────────────────────────────────────┐
   │ 含义: 用 logit-normal 采样的混合系数              │
   │ lambda 倾向 0.7~0.8 (P_mean=0.8)                 │
   │ decoder_z ≈ 0.7*x0 + 0.3*noise                   │
   │ 这是 x0 的轻度加噪版本, 用于教模型解码           │
   │ decoder_noise_scale=1.0 (torch 版)               │
   │   或 5.0 (jax h800 版)                            │
   └────────────────────────────────────────────────────┘

② 模型前向 (decoder 模式):
   decoder_t = ones(B,)               → t=1, 表示"已经去噪到干净数据"
   decoder_input = cat([decoder_z, zeros_like(decoder_z)]) (self-cond 时)
                 或 decoder_z
   _, decoder_logits = model(decoder_input, decoder_t,
                              self_cond_cfg_scale,
                              decoder_step_active=True)
   ┌────────────────────────────────────────────────────┐
   │ decoder_step_active=True →                        │
   │   mode_tokens 的 gate=1 (激活, 告知模型"解码模式") │
   │   decoder_logits = GELU(x @ proj_kernel + proj_bias) @ unembed + bias│
   │   → (B, S, vocab_size=32128)                      │
   └────────────────────────────────────────────────────┘

③ CE loss:
   log_probs = F.log_softmax(decoder_logits.float(), dim=-1)
   ┌────────────────────────────────────────────────────┐
   │ decoder_logits → (B, S, 32128) vocab 上的 logits │
   │ log_softmax     → (B, S, 32128) 概率的对数        │
   └────────────────────────────────────────────────────┘

   ce = -log_probs.gather(dim=-1, index=decoder_targets.unsqueeze(-1)).squeeze(-1)
   ┌────────────────────────────────────────────────────┐
   │ decoder_targets = input_ids → (B, S)              │
   │ .unsqueeze(-1) → (B, S, 1)                       │
   │ gather → 取出目标 token 的 log_prob → (B, S, 1)  │
   │ .squeeze(-1) → (B, S)                            │
   │ ce = -log_prob[target_token] → (B, S)             │
   │ 含义: 每个位置的交叉熵损失                         │
   └────────────────────────────────────────────────────┘

   ce_loss = reduce_token_loss(ce, loss_mask)
   ┌────────────────────────────────────────────────────┐
   │ 只在 loss_mask=1 的位置累加 CE                    │
   │ loss_mask = attention_mask * (1 - cond_seq_mask)  │
   │ → 只在"有效且非 cond"的位置计算                   │
   │ ce_loss = Σ(masked_ce) / Σ(loss_mask) → scalar   │
   └────────────────────────────────────────────────────┘
```

### 5.4 reduce_token_loss 函数

```python
def reduce_token_loss(per_token_loss, mask):
    """
    输入:
      per_token_loss  (B, S) 或 (B, S, D)   每位置/每维的 loss
      mask            (B, S)                 1=计算 loss, 0=忽略

    流程:
      safe = where(mask > 0, per_token_loss, zeros)   # 忽略 mask=0 的位置
      return (safe * mask).sum() / mask.sum().clamp(min=1.0)

    含义: 对 mask=1 的位置求加权平均 loss
          padding 和 cond token 位置不贡献 loss
    """
```

### 5.5 Metrics 返回值

```python
metrics = {
    "loss":    loss.item(),           # 总 loss (= l2_loss 或 ce_loss)
    "l2_loss": l2_loss / (1-decoder_prob),  # denoiser 分支 loss (按概率归一化)
    "ce_loss": ce_loss / decoder_prob,       # decoder 分支 loss (按概率归一化)
}
```

归一化含义: `decoder_prob=0.5` 时, 只有 50% 的步骤进入 CE 分支,
所以 `ce_loss / 0.5 = 2 * ce_loss` 反映了"如果每步都做 CE 分支的期望 loss"。

---

## 六、数据流全景图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        训练数据流                                    │
│                                                                     │
│  Dataset (HF)                                                       │
│      │                                                              │
│      ├─ input_ids (B, S) ────────────────────┐                     │
│      ├─ condition_input_ids (B, S_cond) ───┐ │                     │
│      │                                     │ │                     │
│      │  DataLoader.collate_fn               │ │                     │
│      │  ├─ concat cond + input_ids          │ │                     │
│      │  ├─ pad/truncate to max_length       │ │                     │
│      │  ├─ build_self_attn_cond_masks       │ │                     │
│      │  │  ├─ encoder_attention_mask (B,S,S)│ │                     │
│      │  │  ├─ attention_mask (B, S)         │ │                     │
│      │  │  └─ cond_seq_mask (B, S)          │ │                     │
│      │  └───────────────────────────────────┘ │                     │
│      │                                        │                     │
│      │  prepare_batch()                       │                     │
│      │  ├─ numpy → torch tensors             │                     │
│      │  └─ label_drop_mask (B,)              │                     │
│      │                                        │                     │
│      │  encode_text()                         │                     │
│      │  ├─ T5Encoder(input_ids, attn_mask)    │                     │
│      │  ├─ normalize: (latents - mean) / std  │                     │
│      │  └─ → x0 (B, S, 512)                  │                     │
│      │                                        │                     │
│      │  add_noise()                           │                     │
│      │  └─ z = t*x0 + (1-t)*noise*scale       │                     │
│      │  └─ → z (B, S, 512)                    │                     │
│      │                                        │                     │
│      │  ELF Model Forward                     │                     │
│      │  ├─ self_cond_proj (如果 2*512)         │                     │
│      │  ├─ BottleneckTextProj (512→128→768)    │                     │
│      │  ├─ prepend mode_tokens × gate          │                     │
│      │  ├─ prepend t_prefix + sc_prefix        │                     │
│      │  ├─ RoPE                               │                     │
│      │  ├─ 12 ELFBlocks (Attention + SwiGLU)   │                     │
│      │  ├─ strip prefix                        │                     │
│      │  ├─ decoder_logits (B,S,vocab) [可选]   │                     │
│      │  └─ FinalLayer → output (B, S, 512)     │                     │
│      │                                        │                     │
│      │  Loss Calculation                       │                     │
│      │  ├─ CE branch: -log_softmax[target]     │                     │
│      │  └─ L2 branch: (v_pred - v_target)²     │                     │
│      │  └─ masked mean → scalar loss            │                     │
│      │                                        │                     │
│      │  Backward + Optimizer Step              │                     │
│      │  ├─ loss / grad_accum_steps.backward()  │                     │
│      │  ├─ clip_grad_norm(1.0)                  │                     │
│      │  ├─ optimizer.step()                     │                     │
│      │  └─ EMA update (decay=0.9999)            │                     │
│      └──────────────────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        采样/生成数据流                               │
│                                                                     │
│  初始噪声 z = randn(B, S, 512) * noise_scale                        │
│      │                                                              │
│      │  t_steps = get_sampling_steps(n_steps, ...)                  │
│      │  → [0, t1, t2, ..., tN, 1]  (n_steps+1 个点)               │
│      │                                                              │
│      │  FOR each (t_cur, t_nxt) pair:                               │
│      │      │                                                        │
│      │      ├─ _forward_sample_self_cond()                          │
│      │      │  ├─ 条件前向: model(cat([z, x_prev], -1), t, sc_cfg) │
│      │      │  └─ 可选: 无条件前向 + CFG 混合                       │
│      │      │  └─ → v_pred, x_pred (B, S, 512)                     │
│      │      │                                                        │
│      │      ├─ z_next = z + (t_nxt - t_cur) * v_pred  [ODE]        │
│      │      │  或 z_next = z_back + (t_nxt - t_back) * v_pred [SDE] │
│      │      │                                                        │
│      │  最后一步 ODE: z → final_z                                   │
│      │                                                              │
│      │  decode_latent_to_ids()                                      │
│      │  ├─ model(final_z, t=1.0, decoder_step_active=True)          │
│      │  ├─ decoder_logits.argmax(-1) → predicted_ids (B, S)        │
│      │  └─ mask_after_eos → 截断 EOS 后的内容                      │
│      │  └─ tokenizer.decode → 文本                                 │
│      └──────────────────────────────────────────────────────────────│
└─────────────────────────────────────────────────────────────────────┘
```

---

## 七、核心概念深度解析

### 7.1 两种加噪方式的异同

ELF 有两种加噪公式:

**Denoiser 分支 (flow-matching 加噪):**
```
z = t * x0 + (1-t) * noise * denoiser_noise_scale
```

**Decoder 分支 (logit-normal 加噪):**
```
decoder_z = decoder_lambda_t * x0_for_target + (1-decoder_lambda_t) * decoder_noise * decoder_noise_scale
```

#### 表面相似，实质不同

| | Denoiser z | Decoder z |
|---|---|---|
| 混合系数来源 | `t` 采样自 logit-normal(P_mean, P_std) | `decoder_lambda_t = sigmoid(N(decoder_p_mean, decoder_p_std))` |
| 混合系数分布 | P_mean=-1.5→t≈0.18 (偏向小t, 即更多噪声) | decoder_p_mean=0.8→λ≈0.69 (偏向大λ, 即更多信号) |
| 噪声来源 | `randn_like(x0)` (每个样本一个噪声) | `randn_like(x0) * decoder_noise_scale` (独立噪声) |
| 条件位置处理 | cond token 位置保持干净 x0 (不加噪) | 使用 x0_for_target (label drop 后可能已清零) |
| 时间步 | t 是**随机**的 (训练时覆盖 [0,1] 全范围) | **固定 t=1** (decoder 分支始终在"干净端") |
| 作用 | 教模型学习 velocity field v = dx/dt | 教 decoder head 从"稍微有噪声的 latent"识别出 token |

#### 为什么 decoder 还要单独加噪？

关键理解：flow-matching 的 z 是在**任意 t 值**下的插值，而 decoder 分支的工作是在 **t=1 附近**（即接近干净数据端）做 token 分类。

如果 decoder 直接用 t=1 时的 z (= x0, 无噪声)，那 decoder head 只能处理完美干净的 latent。但在实际生成时，ODE/SDE 采样得到的最终 z **不可能是完美的 x0**——它有数值误差、离散化误差、以及采样路径的累积偏差。decoder 必须学会容忍这种"近似干净但不完美"的输入。

所以 decoder_z 的加噪是一种 **数据增强**：
- `decoder_lambda_t` 偏向高值（~0.69），意味着 decoder_z ≈ 69%干净 + 31%噪声
- 这模拟了"采样出来的 latent 不完美"的情况
- decoder_noise_scale 可以更大（h800.yml 中 5.0 vs denoiser_noise_scale=2.0），给 decoder 更强的鲁棒性训练

**一句话总结：** flow-matching 加噪是为了训练速度场（覆盖全范围 t），decoder 加噪是为了训练 token 分类头（只在 t≈1 附近，容忍不完美的 latent）。

### 7.2 decoder_input 的 shape 对不对？

```python
# train_step.py line 193-196
decoder_input = (
    torch.cat([decoder_z, torch.zeros_like(decoder_z)], dim=-1)
    if self_cond_prob > 0 else decoder_z
)
```

**shape 分析:**

当 `self_cond_prob > 0`:
- `decoder_z` shape: `(B, S, D_model)` 其中 D_model = text_encoder_dim = 512
- `torch.zeros_like(decoder_z)` shape: `(B, S, 512)`
- `torch.cat(..., dim=-1)` → `(B, S, 2*512) = (B, S, 1024)`
- 进入模型 `ELF.forward` 后: `if x.shape[-1] == 2 * self.text_encoder_dim: x = self.self_cond_proj(x)`
- `self_cond_proj`: Linear(1024 → 512) → `(B, S, 512)` ← **正确**

当 `self_cond_prob == 0`:
- `decoder_z` shape: `(B, S, 512)`
- 不拼接，直接传入模型
- 不触发 `self_cond_proj` → `(B, S, 512)` ← **正确**

**为什么 decoder 分支第二个半是 zeros 而不是 x_pred_prev？**

因为 decoder 分支在 **t=1**（干净端），没有"前一步的 denoised 预测"。在采样循环中，self-conditioning 的 x_pred_prev 来自上一个 ODE/SDE 步的输出，但 decoder 分支是直接从 decoder_z 解码，不需要迭代估计。所以第二半填零，告诉模型"我没有之前的估计"。

**Shape 是对的。**

### 7.3 数据集详解 — 不同任务，不同输入输出

ELF 支持三类任务:

#### 任务 1: 无条件文本生成 (OpenWebText)

配置: `train_owt_ELF-B.yml`

```
数据集: OpenWebText (随机互联网文本, tokenized by T5)
数据结构: 每条样本只有 input_ids (没有 condition_input_ids)

collate_fn 处理:
  input_ids_list = [item["input_ids"] for item in batch]   # 只有目标文本
  seq_list = input_ids_list                                  # 直接用，不拼接
  cond_lens = zeros(B)                                       # 没有 conditioning 前缀

  → input_ids           (B, S)     整个文本的 token id
  → cond_seq_mask       (B, S)     全零 (没有 cond token)
  → attention_mask      (B, S)     1=有效, 0=padding
  → encoder_attn_mask   (B, S, S)  全 1 (所有 token 相互可见, 因为没有 cond/x 区分)

训练目标: 给定噪声, 去噪还原出完整的文本 latent, 然后解码回 token
生成时: 从纯噪声开始采样, 无条件地生成文本
```

#### 任务 2: 条件翻译 (德→英 WMT14)

配置: `train_de-en_ELF-B.yml`

```
数据集: WMT14 de-en (德英翻译, tokenized by T5)
数据结构: 每条样本有 condition_input_ids (德语) + input_ids (英语)

collate_fn 处理:
  condition_input_ids = item["condition_input_ids"][:max_input_length=64]  # 德语源, 截断到64
  input_ids = item["input_ids"]                                           # 英语目标

  seq = concatenate([condition_input_ids, input_ids])  # 德语+英语拼在一起
  cond_lens = [len(condition_input_ids)]               # 德语部分的长度
  total_lens = [len(seq)]                              # 整体长度

  → input_ids           (B, S=128)  [德语token, 英语token, pad, pad, ...]
  → cond_seq_mask       (B, S=128)  [1, 1, ..., 1, 0, 0, ..., 0, 0, 0]  ← 德语位置=1
  → attention_mask      (B, S=128)  [1, 1, ..., 1, 1, ..., 1, 0, 0, 0]  ← 有效位置=1
  → encoder_attn_mask   (B, S, S)   复杂的 2D mask:
                                     德语行可以看所有 (德+英)
                                     英语行可以看英语, 但看不到德语 (label_drop 时)

训练目标: 给定 [德语latent, 英语噪声], 去噪还原出英语部分
           cond_seq_mask 确保德语位置始终保持干净 (不被噪声覆盖)
生成时:   先用 T5 编码德语源 → cond_seq
           从噪声开始采样, 用 CFG 引导生成英语翻译
```

#### 任务 3: 条件摘要 (XSum)

配置: `train_xsum_ELF-B.yml`

```
数据集: XSum (新闻摘要, tokenized by T5)
数据结构: 每条样本有 condition_input_ids (原文) + input_ids (摘要)

max_length = 1088, max_input_length = 1024

  seq = concatenate([原文tokens(最长1024), 摘要tokens])  # 拼在一起
  cond_lens = [len(原文)]

  → cond_seq_mask       (B, S=1088)  原文位置=1, 摘要位置=0
  → attention_mask      (B, S=1088)  有效位置=1, padding=0

训练目标: 给定 [原文latent, 摘要噪声], 去噪还原出摘要部分
生成时:   编码原文 → cond_seq, 从噪声采样出摘要
```

#### 关键: 无条件 vs 条件的数据差异

| | 无条件 (OWT) | 条件 (de-en/XSum) |
|---|---|---|
| `condition_input_ids` | ❌ 不存在 | ✅ 存在 (源文本) |
| `cond_seq_mask` | 全零 | 前缀部分=1, 后续=0 |
| `encoder_attn_mask` | 简单 (全1或pad) | 复杂 2D mask (cond/x 区分) |
| `loss_mask` | `attention_mask` (忽略pad) | `attention_mask × (1-cond_mask)` (忽略pad和cond) |
| 采样时 cond_seq | None (无条件) | T5编码的源文本 latent |

### 7.4 CE loss 的 label 和输入是同一个吗？

**是的，但意义不同。**

```python
# 同一个 input_ids 的两种用途:

# 用途1: 送入 T5 编码器, 产出每个词对应位置的上下文 embedding
x0 = encode_text(input_ids=batch["input_ids"], ...)
# → x0 (B, S, 512): 每个 token 位置的 512 维上下文向量

# 用途2: 作为 CE loss 的目标 label
decoder_targets = batch["input_ids"]
# → (B, S): 每个 token 位置的 token id (整数)

# CE loss 计算:
log_probs = F.log_softmax(decoder_logits, dim=-1)  # (B, S, vocab_size)
ce = -log_probs.gather(dim=-1, index=decoder_targets.unsqueeze(-1)).squeeze(-1)
# → 对每个位置, 取出"该位置原始 token id 对应的 log概率", 然后取负
```

**逻辑链:**

```
input_ids "The cat sat on the mat"
    │
    ├→ T5 Encoder → x0: [The的上下文向量, cat的上下文向量, sat的上下文向量, ...]
    │                每个向量 512维, 包含双向注意力的信息
    │
    ├→ 加噪 → z / decoder_z: x0 的加噪版本
    │
    ├→ ELF Model → net_out: 去噪后的 latent
    │                → decoder_logits: (B, S, vocab_size=32128)
    │
    └→ CE Loss target: input_ids = [196, 5574, 382, 29, 8, 119]
                       对 decoder_logits 的每个位置, 取出目标 token 对应的概率
                       希望这个概率尽可能高
```

**所以同一个 `input_ids` 被用了两次:**
1. 做编码器的输入 → 产出 latent 空间中的 x0
2. 做 CE loss 的目标 → 衡量 decoder head 是否能从 latent 恢复出原始 token

这就像一个"自编码器"：先把文本编码成 latent，再从 latent 解码回文本。flow-matching 在 latent 空间中做加噪/去噪，而 decoder head 则是从 latent 回到 token 空间的桥梁。

**对于条件任务，CE loss 的 label 也只是 `input_ids`（目标部分），不包括 `condition_input_ids`（源部分）。** 因为 `loss_mask = attention_mask × (1-cond_seq_mask)` 会忽略 cond 位置的 loss，所以 decoder 只被要求还原目标文本的 token，不被要求还原源文本的 token。

### 7.5 decoder_logits 的 shape 和句子长度

#### Shape

```
decoder_logits: (B, S, vocab_size)

其中:
  B = batch_size (如 64)
  S = max_length (如 128 或 1024, 固定值, 来自 config)
  vocab_size = tokenizer 的词表大小 (t5-small: 32128)
```

#### 模型不知道句子长度

模型**不预测句子长度**。它始终输出固定长度 S 的 logits, 包括 padding 位置。

实际句子长度通过后处理确定:

```python
# generation_utils.py → mask_after_eos
predicted_ids = mask_after_eos(predicted_ids, eos_token_id, pad_token_id)

def mask_after_eos(predicted_ids, eos_token_id, pad_token_id):
    eos_mask = (predicted_ids == eos_token_id)        # 找到 EOS token
    keep_mask = torch.cumsum(eos_mask.long(), dim=1) == 0  # EOS 之前的位置
    return torch.where(keep_mask, predicted_ids, pad_token_id)
    # EOS 之后的所有位置替换为 pad_token_id
```

**示例:**

```
max_length = 128, 实际句子 "The cat sat on the mat" 只有 6 个词 + 1 个 EOS

模型输出 logits shape: (1, 128, 32128)
argmax → predicted_ids: [196, 5574, 382, 29, 8, 119, 1, 0, 0, ..., 0]
                                  ^词    ^词  ^词  ^词 ^词  ^词  ^EOS ^pad...

mask_after_eos →:      [196, 5574, 382, 29, 8, 119, 1, pad, pad, ..., pad]
                                                                   ↑ EOS后全变pad

tokenizer.decode(skip_special_tokens=True) → "The cat sat on the mat"
```

### 7.6 模型如何将隐向量恢复成 logits (Factored Decoder Unembedding)

模型不是一步直接从 hidden_size (768) 映射到 vocab_size (32128)，而是**两步分解**:

```python
# modules/model.py line 268-276

# 第一步: hidden_size → text_encoder_dim (768 → 512)
hidden_proj = F.gelu(x @ self.proj_kernel + self.proj_bias)
# x:               (B, S, hidden_size=768)
# proj_kernel:      (hidden_size=768, text_encoder_dim=512)  ← Parameter, xavier_uniform 初始化
# proj_bias:        (text_encoder_dim=512)                    ← Parameter, zeros 初始化
# x @ proj_kernel:  (B, S, 512)                              ← 矩阵乘法
# + proj_bias:      (B, S, 512)
# F.gelu:           (B, S, 512)                              ← GELU 非线性激活

# 第二步: text_encoder_dim → vocab_size (512 → 32128)
decoder_logits = hidden_proj @ self.unembed_kernel + self.unembed_bias
# hidden_proj:      (B, S, 512)
# unembed_kernel:   (text_encoder_dim=512, vocab_size=32128) ← Parameter, xavier_uniform 初始化
# unembed_bias:     (vocab_size=32128)                        ← Parameter, zeros 初始化
# @ unembed_kernel: (B, S, 32128)
# + unembed_bias:   (B, S, 32128)
# → decoder_logits: (B, S, vocab_size=32128)
```

#### 为什么是两步而不是一步？

直接映射: Linear(768 → 32128) = 768 × 32128 = **24.6M 参数**

分解映射: Linear(768 → 512) + Linear(512 → 32128) = 768×512 + 512×32128 = 393K + 16.5M = **16.9M 参数**

**节省 7.7M 参数 (31%)**, 且中间维度 512 与 T5 encoder 的 d_model 匹配。

#### 中间维度为什么选 512 (= text_encoder_dim)？

这不是巧合。中间投影 `proj_kernel` 把 ELF 的 hidden space (768) 映射到 T5 的 latent space (512)，然后 `unembed_kernel` 从 T5 latent space 映射到 vocab。这形成一个语义上的分解:

```
ELF hidden (768维, 模型内部表示)
    → proj_kernel + GELU → T5 latent (512维, 与 T5 编码器输出同一空间)
    → unembed_kernel → vocab logits (32128维, token 空间)
```

GELU 在中间提供了非线性, 使得第一步不只是线性压缩, 而是一个有表达力的变换。

#### 另一个输出路径: FinalLayer

同时, 模型的**另一个输出** `output` (用于 denoiser 分支) 经过不同的路径:

```python
# modules/model.py line 278-279
output = self.final_layer(x)  # (B, S, text_encoder_dim=512)
# FinalLayer = RMSNorm(768) → Linear(768 → 1*1*512 = 512)
# 全零初始化: 训练刚开始时 output ≈ 0
```

**两个输出, 两个用途:**
- `output` (FinalLayer): 回到 T5 latent 空间, 用于 denoiser 分支的 L2 loss / 下一步采样
- `decoder_logits` (Factored Unembedding): 回到 vocab 空间, 用于 decoder 分支的 CE loss / 解码 token

### 7.7 Self-conditioning merge — 为什么输入有时是 2*D_model

#### Self-conditioning 是什么？

Self-conditioning (自条件化) 是一种**迭代精化**技术:

**没有 self-cond 时:**
```
模型输入: z (加噪的 latent)
模型输出: denoised 预测
```

**有 self-cond 时:**
```
第1次前向: 输入 [z, zeros]           → 得到初始估计 x_pred_init
第2次前向: 输入 [z, x_pred_init]     → 得到更好的估计 x_pred_final
```

把上一步的去噪估计作为当前步的额外输入, 让模型可以"看到自己之前的猜测"并修正它。

#### 为什么特征维度翻倍？

因为要把 z 和 x_pred_prev **拼接 (concatenate)** 在特征维度上:

```
z:              (B, S, D_model=512)   加噪 latent
x_pred_prev:    (B, S, D_model=512)   上一步的去噪估计
concat:         (B, S, 2*D_model=1024) ← 特征维度翻倍!
```

然后在模型内部用 `self_cond_proj` 压缩回去:

```python
# modules/model.py line 220-221
if x.shape[-1] == 2 * self.text_encoder_dim:
    x = self.self_cond_proj(x)
# self_cond_proj: Linear(2*512=1024 → 512)  ← 把两半合并回原始维度
```

#### 为什么不把 x_pred_prev 加到 z 上 (add) 而是拼接 (concat)?

拼接给模型更多信息:
- 加法: z + x_pred → 模型只看到 "合并后的信号", 无法区分"哪部分是噪声, 哪部分是估计"
- 拼接: [z, x_pred] → 模型可以同时看到"原始噪声"和"我的估计", 然后通过 Linear 层学习如何最优地融合两者

`self_cond_proj` 的 Linear 层可以学习任意线性组合, 包括但不限于简单的加法。所以拼接更灵活。

#### 训练时的 self-cond

```python
# train_step.py line 173-188 get_z_input()
# 1. 先用 [z, zeros] 前向 (no_grad)
z_with_zeros = torch.cat([z, restore_cond(torch.zeros_like(z), x0, cond_mask)], dim=-1)
net_out_init = model(z_with_zeros, t, self_cond_cfg_scale=sc_cfg)  # no_grad!
_, x_pred_init = net_out_to_v_x(net_out_init, z, t, t_eps)

# 2. 只对 self_cond_prob 概率的样本使用估计
x_pred_cond = x_pred_init * use_self_cond_mask   # (B, 1, 1) mask
x_pred_cond = restore_cond(x_pred_cond, x0, cond_mask)

# 3. 拼接成最终输入
z_input = torch.cat([z, x_pred_cond], dim=-1)   # (B, S, 1024)
```

`use_self_cond_mask` 是 `(B, 1, 1)` 的 0/1 mask, 以 `self_cond_prob=0.5` 的概率为 1。当 mask=0 时, x_pred_cond=0, 相当于输入 [z, zeros], 即"没有自条件化"。

#### 采样时的 self-cond

在 `generate_samples` 中, `x_pred_prev` 是上一步的输出:

```
步1: z_0, x_pred=None → [z_0, zeros] → model → v_1, x_pred_1
步2: z_1, x_pred=x_pred_1 → [z_1, x_pred_1] → model → v_2, x_pred_2
步3: z_2, x_pred=x_pred_2 → [z_2, x_pred_2] → model → v_3, x_pred_3
...
```

每一步都用上一步的 denoised 预测作为 self-cond 输入, 逐步精化。

### 7.8 T5 在 ELF 中的角色

**T5 产出的是每个词对应位置的上下文 embedding, 不是单独的 vocab embedding, 也不是句子级向量。**

#### 三种可能的用法对比

| 用法 | T5 输出 | Shape | 特点 |
|---|---|---|---|
| ❌ 只取 vocab embedding | 查表得到词向量 | (B, S, 512) | 每个 token 固定向量, 不考虑上下文 |
| ✅ **ELF 实际用法** | T5 encoder 整体输出 | (B, S, 512) | 每个 token 的向量融合了双向注意力信息 |
| ❌ 句子级向量 | 池化/平均 | (B, 1, 512) | 丢失了 per-token 信息 |

#### ELF 的实际流程

```python
# modules/t5_encoder.py
class T5Encoder:
    def forward(self, input_ids, attention_mask):
        # input_ids:  (B, S)        例: "The cat sat on the mat" → [196, 5574, 382, ...]
        # attention_mask: (B, S)     例: [1, 1, 1, 1, 1, 1, 1, 0, 0, ...]

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # T5EncoderModel 内部:
        #   1. Token embedding: input_ids → embedding_lookup → (B, S, 512)
        #   2. 6 层 T5 encoder blocks (每层含 bidirectional self-attention + FFN)
        #   3. 每个 token 可以看到序列中所有其他有效 token

        return outputs.last_hidden_state   # (B, S, 512)
```

#### 关键理解: 双向上下文 embedding

T5 encoder 是**双向**的 (不像 GPT 是单向的)。这意味着:

- "cat" 在 "The **cat** sat" 中的 embedding 包含了 "The" 和 "sat" 的信息
- 同一个 token "cat" 在不同句子中会有不同的 embedding
- 这是 bidirectional self-attention 的结果: 每层每个 token 都与序列中所有其他 token 交互

**示例:**

```
句子A: "The cat sat on the mat"
句子B: "My cat is named Felix"

T5 输出 (句子A): [
    The_A的向量,   ← 包含 "cat, sat, mat" 的上下文信息
    cat_A的向量,   ← 包含 "The, sat, mat" 的上下文信息 (知道这是"坐在垫子上的猫")
    sat_A的向量,   ← 包含 "The, cat, mat" 的上下文信息
    ...
]

T5 输出 (句子B): [
    My_B的向量,
    cat_B的向量,   ← 包含 "named, Felix" 的上下文信息 (知道这是"名叫Felix的猫")
    ...
]

cat_A ≠ cat_B  ← 虽然 token id 相同, 但上下文 embedding 不同!
```

#### T5 在 ELF 中的完整作用

T5 定义了 ELF 的**数据空间**:

```
文本空间:    "The cat sat on the mat"       (人类可读的字符串)
    ↓ tokenizer (T5 的 tokenizer)
Token 空间:  [196, 5574, 382, 29, 8, 119]  (整数序列)
    ↓ T5 encoder (bidirectional attention)
Latent 空间: [(B,S,512) 的上下文向量]        (ELF 的工作空间)
    ↓ flow-matching 加噪/去噪                ← ELF 的核心能力在这里
    ↓ decoder unembedding (factored)
Token 空间:  [196, 5574, 382, ...]           (预测的 token id)
    ↓ tokenizer.decode
文本空间:    "The cat sat on the mat"       (生成的文本)
```

ELF 不在 token 空间中做扩散 (那样会很困难, 因为 token 是离散的),
而是在 T5 的**连续 latent 空间**中做 flow-matching (连续空间的扩散容易得多)。

T5 是 **文本 ↔ latent** 的桥梁:
- 编码方向: 文本 → latent (训练时用, 固定不变)
- 解码方向: latent → token (通过 decoder head, 训练时学习)

#### 为什么 T5 是冻结的?

```python
# modules/t5_encoder.py line 53-55
self.encoder = T5EncoderModel.from_pretrained(model_name)
self.encoder.requires_grad_(False)   # ← 冻结! 不更新参数
self.encoder.eval()
```

T5 不参与训练, 因为:
1. T5 已经在大规模数据上预训练好了, 产出的 embedding 质量很高
2. 如果 T5 也训练, latent 空间会不断变化, flow-matching 模型需要追踪这个变化的空间, 非常不稳定
3. 固定 T5 = 固定 latent 空间 = flow-matching 有一个稳定的目标分布
4. 训练 `encode_text` 时用 `@torch.no_grad()`, 完全不计算 T5 的梯度

#### 归一化的意义

```python
# utils/encoder_utils.py line 29-30
latents = encoder(input_ids=input_ids, attention_mask=attention_mask)
return (latents - latent_mean) / latent_std
```

T5 输出的原始 latent 的均值和方差可能不稳定。归一化到 `mean=0, std=1` (或 `std=0.2`) 确保:
- flow-matching 的噪声尺度与数据尺度匹配
- 不同 T5 模型 (small/base/large) 产出不同尺度的 embedding, 归一化统一了尺度

`latent_mean=0.0, latent_std=0.2` (h800.yml) 意味着归一化后 x0 的 std≈0.2, 比标准正态分布更紧凑。这让噪声和数据的比例更合适。

#### attention_mask 的特殊处理

T5 的 `forward` 还处理了 2D mask:

```python
# modules/t5_encoder.py line 72-78
if attention_mask.ndim == 3:  # (B, S, S) 2D mask
    attention_mask_1d = (attention_mask.sum(dim=-1) > 0).long()
    # 2D mask → 1D mask: 只要某行的 mask 在任何列有非零值, 该行就是 "有效"
```

这是因为 HuggingFace T5EncoderModel 只接受 1D mask (B, S), 但 ELF 的 label drop 机制产出了 2D mask (B, S, S)。2D → 1D 转换保留了 "哪些 token 是有效的" 信息, 但丢失了 "哪些 token 可以看到哪些 token" 的精细控制。这是 PyTorch 版本的一个**简化**（JAX 版本使用了手写的 T5, 可以直接接受 2D mask）。

### 7.9 全景总结: 数据→编码→加噪→去噪→解码→文本

```
┌─────────────────────────────────────────────────────────────┐
│  训练时 (每个 batch):                                        │
│                                                             │
│  文本: "The cat sat on the mat"                             │
│    ↓ tokenizer                                              │
│  input_ids: [196, 5574, 382, 29, 8, 119, 1, 0, ...]       │
│    ↓ T5 encoder (冻结, 双向注意力, 6层)                     │
│  x0: 每个位置一个512维上下文向量 → (B, S, 512)              │
│    ↓ 归一化: (x0 - mean) / std                              │
│  x0_normalized: (B, S, 512) ← ELF 的 "干净数据"            │
│                                                             │
│  ┌── Denoiser 分支 (L2 loss) ──────────────────────┐       │
│  │  x0 → z = t*x0 + (1-t)*noise*scale               │       │
│  │  z → [z, x_pred_prev] (self-cond) → ELF model    │       │
│  │  → net_out → v_pred                               │       │
│  │  v_target = (x0 - z)/(1-t)                        │       │
│  │  loss = (v_pred - v_target)² 的 masked mean       │       │
│  └──────────────────────────────────────────────────┘       │
│                                                             │
│  ┌── Decoder 分支 (CE loss) ───────────────────────┐       │
│  │  x0 → decoder_z = λ*x0 + (1-λ)*noise             │       │
│  │  decoder_z → ELF model (decoder_step_active=True) │       │
│  │  → decoder_logits (B, S, 32128)                   │       │
│  │  loss = -log_softmax[target_token] 的 masked mean │       │
│  └──────────────────────────────────────────────────┘       │
│                                                             │
│  两个分支随机切换: decoder_prob=0.2 → 80% L2, 20% CE       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  生成时:                                                     │
│                                                             │
│  初始: z = randn(B, S, 512) * noise_scale                  │
│  (条件任务时: cond_seq = T5编码(源文本), prepend 到 z)     │
│                                                             │
│  FOR step 1..N:                                             │
│    [z, x_pred_prev] → ELF model → v_pred                   │
│    z_next = z + Δt × v_pred                                 │
│    (可选 CFG: v = v_uncond + scale*(v_cond-v_uncond))      │
│    (可选 SDE: 先加噪退一步, 再向前积分)                    │
│                                                             │
│  最后: final_z ≈ x0 (去噪到接近干净数据)                   │
│    final_z → ELF model (decoder_step_active=True)           │
│    → decoder_logits (B, S, 32128)                           │
│    → argmax → predicted_ids                                 │
│    → mask_after_eos → 截断EOS后                             │
│    → tokenizer.decode → "The cat sat on the mat"            │
└─────────────────────────────────────────────────────────────┘
```