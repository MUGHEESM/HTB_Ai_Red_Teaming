# Diffusion Models

## Overview

**Diffusion models** are a class of generative models that have gained significant attention for their ability to **generate high-quality images**. They represent one of the most exciting recent developments in generative AI.

### What are Diffusion Models?

**Definition:** Generative models that learn to create data by reversing a gradual noising process.

**Key innovation:** Use a series of noise addition and removal steps to learn the data distribution.

---

## Diffusion Models vs Other Generative Models

| Feature | Diffusion Models | GANs | VAEs |
|---------|-----------------|------|------|
| **Training** | Stable, no adversarial dynamics | Unstable, adversarial | Stable, reconstruction-based |
| **Quality** | State-of-the-art | High quality | Good (sometimes blurry) |
| **Diversity** | Excellent | Prone to mode collapse | Good diversity |
| **Speed** | Slow (many steps) | Fast (single forward pass) | Fast (single forward pass) |
| **Control** | Fine-grained | Moderate | Good (latent space) |
| **Architecture** | Denoising network | Generator + Discriminator | Encoder + Decoder |

---

## Why Diffusion Models?

**Advantages over previous approaches:**

✅ **Superior image quality:**
- State-of-the-art results
- Highly realistic outputs
- Fine details preserved

✅ **Stable training:**
- No adversarial dynamics (unlike GANs)
- No mode collapse
- Predictable convergence

✅ **Excellent diversity:**
- Generate wide variety of outputs
- Captures full data distribution
- No missing modes

✅ **Flexible conditioning:**
- Easy to condition on text, class labels, etc.
- Fine-grained control over generation
- Multiple conditioning methods

✅ **Theoretical foundation:**
- Well-understood probabilistic framework
- Clear mathematical formulation
- Principled approach

**Trade-offs:**
❌ **Slow generation**: Requires many denoising steps (50-1000)
❌ **Computational cost**: Expensive training and inference
❌ **Large models**: Require significant resources

---

## Applications of Diffusion Models

**Diffusion models excel at:**

### Image Generation

**Text-to-Image:**
- ✅ **DALL-E 2**: OpenAI's text-to-image model
- ✅ **Stable Diffusion**: Open-source, high-quality generation
- ✅ **Midjourney**: Artistic image generation
- ✅ **Imagen**: Google's text-to-image model

**Image Editing:**
- ✅ **Inpainting**: Filling in missing regions
- ✅ **Outpainting**: Extending images beyond borders
- ✅ **Image-to-image translation**: Style transfer, variations
- ✅ **Super-resolution**: Upscaling images

---

### Audio and Video

**Audio Generation:**
- ✅ Speech synthesis
- ✅ Music generation
- ✅ Sound effect creation

**Video Generation:**
- ✅ Text-to-video synthesis
- ✅ Video editing and enhancement
- ✅ Animation generation

---

### Other Domains

**3D Generation:**
- 3D object synthesis
- Scene generation
- Texture synthesis

**Scientific Applications:**
- Molecular design (drug discovery)
- Protein structure prediction
- Materials science

---

## How Diffusion Models Work

![Cat in a Hat Example](images/a_cat_in_a_hat.png)
*Cat wearing a colorful hat with a feather, sitting in a room with books and teacups.*

Diffusion models function by **gradually adding noise** to an input and then **learning to reverse this process** to generate new data.

---

## Two-Phase Process

### Phase 1: Forward Process (Training Time)

**Add noise progressively:**

```
Clean Image → Slightly Noisy → Noisier → Very Noisy → Pure Noise

x₀ → x₁ → x₂ → ... → x_T

Step 0: Original cat image
Step 1: Add small amount of noise
Step 2: Add more noise
...
Step T: Pure random noise (cat completely destroyed)
```

**Purpose:** Understand how noise corrupts data

---

### Phase 2: Reverse Process (Generation Time)

**Remove noise progressively:**

```
Pure Noise → Very Noisy → Noisier → Slightly Noisy → Clean Image

x_T → ... → x₂ → x₁ → x₀

Step T: Start with pure random noise
Step T-1: Remove a bit of noise
Step T-2: Remove more noise
...
Step 0: Generated cat image
```

**Purpose:** Generate new data by reversing the noising process

---

## Text-to-Image Generation

**When generating images from text prompts** (e.g., "a cat in a hat"), additional steps are required to incorporate the text.

### Complete Text-to-Image Pipeline

```
Text Prompt: "A cat in a hat"
    ↓
[Text Encoder (CLIP/T5)] → Text Embedding
    ↓
Pure Noise + Text Embedding
    ↓
[Conditional Denoising Process]
    ↓
Generated Image
```

---

## Step 1: Text Encoding

**The first step is to encode the textual prompt** using a pre-trained text encoder.

**Process:**

**Input:**
```
Text prompt: "a cat in a hat"
```

**Text Encoder (e.g., CLIP, T5):**
- Converts text to high-dimensional vector
- Captures semantic meaning
- Understands relationships between concepts

**Output:**
```
Text embedding: [0.34, -0.12, 0.89, ..., 0.45]
(typically 512-1024 dimensions)
```

**This vector captures:**
- "cat" (animal concept)
- "hat" (clothing item)
- "wearing" (relationship between cat and hat)
- Visual attributes
- Compositional understanding

---

### Popular Text Encoders

**CLIP (Contrastive Language-Image Pre-training):**
- Trained on 400M image-text pairs
- Understands visual-semantic relationships
- Used in: DALL-E 2, Stable Diffusion

**T5 (Text-to-Text Transfer Transformer):**
- Large language model
- Rich text understanding
- Used in: Imagen

**BERT variants:**
- Deep language understanding
- Contextual embeddings
- Various diffusion model implementations

---

## Step 2: Conditioning the Denoising Process

**The latent representation of the text is used to condition the denoising network.**

### How Conditioning Works

**During the reverse process:**

**Standard (unconditional) denoising:**
```python
predicted_noise = denoising_network(noisy_image, timestep)
```

**Conditional (text-guided) denoising:**
```python
predicted_noise = denoising_network(
    noisy_image, 
    timestep, 
    text_embedding  # ← Conditioning!
)
```

---

### Conditioning Mechanisms

**Cross-attention:**
- Denoising network attends to text embedding
- Different parts of image attend to different text features
- Most common in modern diffusion models

**Architecture:**
```
Noisy Image Features → [Self-Attention]
                              ↓
                     [Cross-Attention with Text]
                              ↓
                       [Feed-Forward]
                              ↓
                    Denoised Image Features
```

**Concatenation:**
- Append text embedding to noisy image
- Simple but less flexible

**Adaptive normalization:**
- Text embedding modulates normalization parameters
- Used in StyleGAN-style architectures

---

### Loss Function Modification

**The loss function includes a term measuring alignment with text:**

**Standard diffusion loss:**
```
L = E[||ε - ε_pred||²]
```

**Text-conditioned diffusion loss:**
```
L = E[||ε - ε_pred(x_t, t, text_emb)||²]
```

**With classifier-free guidance:**
```
ε_pred = ε_uncond + w · (ε_cond - ε_uncond)
```

Where:
- **ε_uncond**: Noise prediction without text
- **ε_cond**: Noise prediction with text
- **w**: Guidance scale (controls text adherence)

**Higher w → Stronger text adherence, less diversity**
**Lower w → Weaker text adherence, more diversity**

---

## Step 3: Sampling Process

**The sampling process begins with pure noise**, as in unconditional diffusion models.

### Text-Guided Sampling

**Initialize:**
```python
x_T = random_noise()  # Pure Gaussian noise
text_emb = encode_text("a cat in a hat")
```

**Iterative denoising:**
```python
for t in reversed(range(T)):
    # Predict noise using both image and text
    predicted_noise = denoising_network(
        noisy_image=x_t,
        timestep=t,
        text_embedding=text_emb
    )
    
    # Remove predicted noise
    x_t_minus_1 = denoise_step(x_t, predicted_noise, t)
    
    # Add small random noise (except last step)
    if t > 0:
        x_t_minus_1 += small_random_noise()
    
    x_t = x_t_minus_1
```

**Result:** x₀ = generated image of "a cat in a hat"

---

### Guidance Techniques

**Classifier Guidance:**
- Use pre-trained classifier to guide generation
- Compute gradient toward desired class
- Push generation toward target concept

**Classifier-Free Guidance:**
- Train model with and without conditioning
- Interpolate between conditional and unconditional predictions
- More stable, widely used (Stable Diffusion, DALL-E 2)

**CLIP Guidance:**
- Use CLIP score to guide generation
- Optimize for high text-image similarity
- Can be combined with other methods

---

## Step 4: Final Image Generation

**After a sufficient number of denoising steps**, the model produces a final image consistent with the given prompt.

### What Makes It Work?

**The iterative process of adding and removing noise, guided by the text embedding:**

✅ **Gradual refinement:**
- Start from pure noise
- Slowly add structure
- Refine details over time

✅ **Text alignment:**
- Each step considers text prompt
- Ensures consistency with description
- Balances text adherence and image quality

✅ **Stochasticity:**
- Random noise in intermediate steps
- Enables diversity
- Different results each generation

✅ **Multi-scale understanding:**
- Early steps: Rough composition, layout
- Middle steps: Object shapes, colors
- Late steps: Fine details, textures

---

### Generation Quality Factors

**What affects output quality:**

**Model quality:**
- Size of denoising network
- Training dataset quality and size
- Training duration

**Sampling parameters:**
- Number of steps (more = better quality, slower)
- Guidance scale (higher = stronger text adherence)
- Noise schedule

**Prompt quality:**
- Clear, descriptive prompts
- Specific details
- Style specifications

---

## By Integrating These Steps

**Diffusion models can effectively generate images from textual prompts**, making them powerful tools for:

✅ **Text-to-image synthesis:**
- Generate images from descriptions
- Creative content creation
- Concept visualization

✅ **Creative content generation:**
- Art generation
- Design prototyping
- Marketing materials

✅ **Practical applications:**
- Product visualization
- Architectural rendering
- Game asset creation
- Film pre-production

**The ability to condition the generation process on text** allows diffusion models to produce **diverse and contextually relevant images**, opening up a wide range of applications in fields like **art, design, and content creation**.

---

## Forward Process: Adding Noise

![Forward Process](images/diffusion_first_five_forward_process_adding_Noise.png)
*Five grayscale noise pattern images labeled Step 0 to Step 4.*

The forward process in diffusion models involves **gradually adding noise to the data until it becomes pure noise**.

This process is often called the **"forward diffusion"** or **"noising"** process.

---

## Forward Process: Mathematical Formulation

**Final state:**
```python
x_T ~ q(x_T | x_0)
```

**Where:**
- **x₀**: Original data (e.g., clean image)
- **x_T**: Pure noise (Gaussian noise)
- **q(x_T | x₀)**: Distribution of noisy data given original data

---

### Sequential Forward Process

**A sequence of intermediate steps** typically defines the forward process:

```python
x_t ~ q(x_t | x_{t-1})
```

**Where:**
- **t**: Time step, ranging from 0 to T
- **q(x_t | x_{t-1})**: Transition probability from step t-1 to step t

**At each step:**
```
x_t = √(1 - β_t) · x_{t-1} + √β_t · ε_t
```

Where:
- **β_t**: Noise variance at step t (noise schedule)
- **ε_t**: Gaussian noise ~ N(0, I)

---

## Forward Process Visualization

**Step-by-step transformation:**

```
Step 0: x₀ (Original image - clear cat)
        100% signal, 0% noise

Step 1: x₁ = √(1-β₁)·x₀ + √β₁·ε₁
        98% signal, 2% noise
        (barely noticeable noise)

Step 2: x₂ = √(1-β₂)·x₁ + √β₂·ε₂
        96% signal, 4% noise
        (slight graininess)

Step 10: x₁₀
        80% signal, 20% noise
        (visible noise, still recognizable)

Step 50: x₅₀
        30% signal, 70% noise
        (heavily noisy, vague shapes)

Step 100: x₁₀₀ ≈ pure noise
        ~0% signal, ~100% noise
        (no discernible image, pure Gaussian noise)
```

---

## Key Properties of Forward Process

**1. Markov property:**
- Each step depends only on previous step
- Not on entire history
```
q(x_t | x_{t-1}, x_{t-2}, ..., x_0) = q(x_t | x_{t-1})
```

**2. Deterministic given noise:**
- If we know x₀ and all noise ε_t
- We can compute any x_t directly

**3. Closed-form solution:**
```python
x_t = √(ᾱ_t) · x_0 + √(1 - ᾱ_t) · ε

where ᾱ_t = ∏(1 - β_i) for i=1 to t
```

**This is powerful:** Can sample any timestep directly without computing all intermediate steps!

---

## Forward Process Algorithm

```python
def forward_process(x_0, t, beta_schedule):
    """
    Add noise to clean image x_0 at timestep t.
    
    Args:
        x_0: Clean image
        t: Timestep (0 to T)
        beta_schedule: Noise schedule β_1, ..., β_T
    
    Returns:
        x_t: Noisy image at timestep t
        epsilon: The noise added
    """
    # Compute cumulative products
    alpha_t = 1 - beta_schedule[t]
    alpha_bar_t = torch.prod(1 - beta_schedule[:t+1])
    
    # Sample noise
    epsilon = torch.randn_like(x_0)
    
    # Add noise to clean image
    sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t)
    
    x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * epsilon
    
    return x_t, epsilon
```

---

## Why Forward Process?

**Purpose of the forward process:**

✅ **Training data generation:**
- Create noisy versions of clean images
- Train denoising network on these pairs
- (x_t, t) → predict ε

✅ **Well-defined target:**
- Know exact noise added
- Clear training objective
- Supervised learning

✅ **Gradual corruption:**
- Model learns denoising at all noise levels
- From slightly noisy to pure noise
- Multi-scale understanding

---

## Reverse Process: Removing Noise

![Reverse Process](images/diffusion_final_fiveReverse_process_removing_noise.png)
*Five grayscale images showing noise patterns labeled Step 96 to Step 100, with emerging shapes.*

The reverse process, known as the **"denoising"** process, involves **learning to remove the noise** added during the forward process.

**The goal:** Map the noisy data back to the original data distribution.

---

## Reverse Process: Mathematical Formulation

**Learned reverse transition:**
```python
x_{t-1} ~ p_θ(x_{t-1} | x_t)
```

**Where:**
- **p_θ(x_{t-1} | x_t)**: Learned distribution parameterized by model's parameters θ
- **θ**: Neural network parameters (learned during training)

---

### Parameterization

**The reverse process is typically parameterized to predict the noise:**

```python
ε_pred = ε_θ(x_t, t)
```

**Then compute x_{t-1}:**
```python
x_{t-1} = (1/√α_t) · (x_t - ((1-α_t)/√(1-ᾱ_t)) · ε_pred) + σ_t · z
```

**Where:**
- **ε_pred**: Predicted noise
- **α_t = 1 - β_t**: Signal retention factor
- **ᾱ_t**: Cumulative product of α
- **σ_t**: Variance of noise to add (for stochasticity)
- **z**: Random Gaussian noise (for diversity)

---

## Training the Reverse Process

**The reverse process is trained to minimize the difference between predicted and actual noise.**

### Training Objective

**Loss function (Mean Squared Error):**

```python
L = E[||ε - ε_pred||²]
```

**Expanded:**
```python
L = E_{x_0, t, ε} [||ε - ε_θ(x_t, t)||²]
```

**Where:**
- **ε**: Actual noise added in forward process
- **ε_pred = ε_θ(x_t, t)**: Predicted noise by network
- **x_t**: Noisy image at timestep t
- **t**: Random timestep sampled uniformly

---

### Training Algorithm

```python
def train_step(x_0, denoising_network, beta_schedule):
    """
    Single training step for diffusion model.
    
    Args:
        x_0: Clean image from dataset
        denoising_network: Neural network to train
        beta_schedule: Noise schedule
    
    Returns:
        loss: Training loss
    """
    # Sample random timestep
    t = random.randint(0, T)
    
    # Add noise to clean image
    x_t, epsilon = forward_process(x_0, t, beta_schedule)
    
    # Predict the noise
    epsilon_pred = denoising_network(x_t, t)
    
    # Compute loss
    loss = mse_loss(epsilon, epsilon_pred)
    
    # Backpropagation and parameter update
    loss.backward()
    optimizer.step()
    
    return loss
```

---

## Reverse Process Visualization

**Step-by-step denoising:**

```
Step T=100: x₁₀₀ (Pure noise)
            100% noise, 0% signal
            
            ↓ [Predict and remove noise]

Step 99: x₉₉
         95% noise, 5% signal
         (very faint structure emerging)

Step 90: x₉₀
         80% noise, 20% signal
         (vague shapes appearing)

Step 70: x₇₀
         60% noise, 40% signal
         (rough object outlines)

Step 50: x₅₀
         40% noise, 60% signal
         (recognizable objects, colors)

Step 20: x₂₀
         10% noise, 90% signal
         (clear image, fine details forming)

Step 1: x₁
        1% noise, 99% signal
        (nearly perfect image)

Step 0: x₀ (Generated image)
        0% noise, 100% signal
        (final clean image)
```

---

## Why Predict Noise Instead of Image?

**Alternative parameterizations:**

**1. Predict clean image directly:**
```python
x_0_pred = f_θ(x_t, t)
```
- Difficult: Large jump from noise to clean image
- Unstable training

**2. Predict noise (standard approach):**
```python
ε_pred = ε_θ(x_t, t)
```
- Easier: Small corrections at each step
- Stable training
- Better empirical results

**3. Predict velocity:**
```python
v_pred = v_θ(x_t, t)
```
- Interpolation between noise and image prediction
- Used in some recent models

---

## Noise Schedule

![Noise Schedule](images/diffusion_mid_five_noise_schedule.png)
*Five grayscale images showing noise patterns labeled Step 80 to Step 84, with emerging shapes.*

The **noise schedule** determines how much noise is added at each step of the forward process.

**Critical hyperparameter:** Significantly impacts diffusion model performance.

---

## What is a Noise Schedule?

**Definition:** Sequence of noise variances β₁, β₂, ..., β_T that controls noise addition.

**At each step t:**
- **β_t**: Variance of noise to add
- Controls how quickly image becomes noise

---

## Common Noise Schedules

### 1. Linear Schedule

**Formula:**
```python
β_t = β_min + (t / T) · (β_max - β_min)
```

**Where:**
- **β_t**: Variance of noise at step t
- **β_min**: Minimum variance (e.g., 0.0001)
- **β_max**: Maximum variance (e.g., 0.02)
- **T**: Total number of steps

**Example:**
```python
T = 1000
β_min = 0.0001
β_max = 0.02

β_1 = 0.0001 + (1/1000) · (0.02 - 0.0001) ≈ 0.00012
β_500 = 0.0001 + (500/1000) · (0.02 - 0.0001) ≈ 0.01
β_1000 = 0.0001 + (1000/1000) · (0.02 - 0.0001) = 0.02
```

**Characteristics:**
- Simple, intuitive
- Uniform noise addition rate
- Used in original DDPM paper

---

### 2. Cosine Schedule

**Formula:**
```python
ᾱ_t = cos²((t/T + s) / (1 + s) · π/2)
β_t = 1 - (ᾱ_t / ᾱ_{t-1})
```

**Where:**
- **s**: Small offset (e.g., 0.008)
- **ᾱ_t**: Cumulative signal retention

**Characteristics:**
- Slower noise addition early
- Faster in middle
- Slower near end
- Better performance on images

**Advantages:**
✅ More stable training
✅ Better sample quality
✅ Improved generation

---

### 3. Learned Schedule

**Approach:** Learn noise schedule during training

**Methods:**
- Optimize β_t as trainable parameters
- Use neural network to predict schedule
- Adapt schedule to dataset

**Advantages:**
✅ Dataset-specific optimization
✅ Potentially better performance

**Disadvantages:**
❌ More complex
❌ Harder to train
❌ Less interpretable

---

## Noise Schedule Visualization

**Impact on forward process:**

```
Linear Schedule:
Step 0:   ████████████████ (100% signal)
Step 250: ████████░░░░░░░░ (50% signal)
Step 500: ████░░░░░░░░░░░░ (25% signal)
Step 750: ██░░░░░░░░░░░░░░ (12% signal)
Step 1000: ░░░░░░░░░░░░░░░░ (0% signal)

Cosine Schedule:
Step 0:   ████████████████ (100% signal)
Step 250: ██████████████░░ (88% signal) ← Slower early
Step 500: ██████░░░░░░░░░░ (38% signal) ← Faster middle
Step 750: ██░░░░░░░░░░░░░░ (12% signal)
Step 1000: ░░░░░░░░░░░░░░░░ (0% signal)
```

---

## Choosing Noise Schedule

**Factors to consider:**

**1. Data type:**
- Images: Cosine often better
- Audio: Linear may work
- 3D data: Depends on structure

**2. Number of steps:**
- Few steps (50-100): Cosine recommended
- Many steps (1000+): Linear acceptable

**3. Empirical testing:**
- Try different schedules
- Evaluate sample quality
- Measure FID scores

**Common defaults:**
- **DDPM**: Linear schedule, 1000 steps
- **Stable Diffusion**: Cosine schedule, 50-100 steps
- **DALL-E 2**: Custom schedule

---

## Impact on Training and Sampling

**Well-designed schedule ensures:**

✅ **Effective learning:**
- Model learns denoising across all noise levels
- Not too easy, not too hard
- Balanced difficulty

✅ **Sample quality:**
- Smooth denoising process
- No abrupt transitions
- High-quality outputs

✅ **Training stability:**
- Stable gradient flow
- No vanishing/exploding gradients
- Convergence

---

## Denoising Network

The **denoising network** is a neural network that learns to **predict the noise at each time step**.

**Core component:** The "brain" of the diffusion model.

---

## Denoising Network Architecture

**Input:**
- **x_t**: Noisy image at timestep t
- **t**: Timestep (often embedded as additional input)
- **c** (optional): Conditioning information (text, class label)

**Output:**
- **ε_pred**: Predicted noise

**Function:**
```python
ε_pred = ε_θ(x_t, t, c)
```

---

## Common Architectures

### 1. U-Net (Most Popular)

**Structure:**
```
                    [Bottleneck]
                         ↑
    [Encoder]    → → → → ↓ → → →    [Decoder]
                   Skip Connections
```

**Components:**

**Encoder (Downsampling):**
- Convolutional layers
- Progressively reduce spatial dimensions
- Increase channel dimensions
- Extract features

**Bottleneck:**
- Lowest spatial resolution
- Highest-level features
- Often includes attention layers

**Decoder (Upsampling):**
- Transposed convolutions or upsampling
- Progressively increase spatial dimensions
- Reconstruct image
- Skip connections from encoder

---

**Why U-Net?**

✅ **Skip connections:**
- Preserve fine details
- Better gradient flow
- Combines low and high-level features

✅ **Multi-scale processing:**
- Different resolutions capture different features
- Coarse to fine generation

✅ **Proven effectiveness:**
- Originally for medical image segmentation
- Adapted successfully for diffusion models
- State-of-the-art results

---

### 2. Transformer-Based (DiT)

**Diffusion Transformer (DiT):**
- Replace U-Net with Vision Transformer
- Patch-based processing
- Self-attention across patches

**Architecture:**
```
Noisy Image Patches
    ↓
[Patch Embedding]
    ↓
[Transformer Blocks]
    ↓
[Unpatch]
    ↓
Predicted Noise
```

**Advantages:**
✅ Scalability
✅ Long-range dependencies
✅ Flexibility

**Disadvantages:**
❌ Higher computational cost
❌ Requires more data

---

### 3. Hybrid Architectures

**Combining best of both:**
- U-Net backbone
- Transformer layers in bottleneck
- Attention at multiple scales

**Example: Stable Diffusion**
- U-Net architecture
- Cross-attention layers for text conditioning
- Self-attention at low resolutions

---

## Time Embedding

**Timestep t must be incorporated into the network:**

**Methods:**

**1. Sinusoidal embedding:**
```python
def timestep_embedding(t, dim):
    """
    Create sinusoidal embedding for timestep.
    """
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim) * -emb)
    emb = t[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb
```

**2. Adaptive normalization (AdaGN):**
```python
# Time embedding modulates normalization
gamma, beta = time_mlp(time_embedding)
normalized = group_norm(features)
output = gamma * normalized + beta
```

**3. Conditional layer normalization:**
- Time embedding conditions each layer
- Similar to AdaGN

---

## Conditioning Mechanisms

**For text-to-image or class-conditional generation:**

### Cross-Attention

**Most effective for text conditioning:**

```
Image Features                Text Embedding
      ↓                             ↓
[Self-Attention]              [Text Encoder]
      ↓                             ↓
      └──→ [Cross-Attention] ←──────┘
                  ↓
          Combined Features
```

**Allows:**
- Different image regions attend to different text features
- Fine-grained control
- Compositionality

---

### Classifier-Free Guidance

**Train single model for both conditional and unconditional generation:**

**During training:**
- Randomly drop conditioning (e.g., set text = "" with probability 10%)
- Model learns both conditional and unconditional distributions

**During inference:**
```python
ε_pred = ε_uncond + w · (ε_cond - ε_uncond)
```

Where **w** is guidance scale (typically 7-15).

**Benefits:**
✅ No separate classifier needed
✅ Better sample quality
✅ Controllable text adherence

---

## Architecture Considerations

**The architecture must be:**

✅ **Powerful enough:**
- Capture complex patterns in data
- Handle various noise levels
- Learn rich representations

✅ **Efficient enough:**
- Handle large datasets
- Process high-resolution images
- Reasonable training time

✅ **Scalable:**
- Work with different image sizes
- Handle different conditioning types
- Adaptable to various domains

---

## Training

**Training a diffusion model involves minimizing the loss function over multiple time steps** using gradient descent and backpropagation.

**Computational intensity:** Especially for high-resolution images, but results in a model that can generate high-quality samples.

---

## Training Process Summary

### Step 1: Initialize the Model

**Start with initial parameters θ:**

```python
# Define architecture
denoising_network = UNet(
    in_channels=3,
    out_channels=3,
    hidden_dims=[128, 256, 512, 1024],
    attention_resolutions=[16, 8],
    num_heads=8
)

# Initialize parameters randomly
denoising_network.apply(init_weights)

# Setup optimizer
optimizer = AdamW(
    denoising_network.parameters(),
    lr=1e-4,
    weight_decay=0.01
)
```

---

### Step 2: Forward Process

**Add noise to original data using noise schedule:**

```python
# Sample batch from dataset
batch = next(dataloader)  # Clean images x_0

# Sample random timesteps for each image
t = torch.randint(0, T, (batch_size,))

# Add noise according to forward process
x_t, epsilon = forward_process(batch, t, beta_schedule)
```

This creates training pairs: (x_t, t) → ε

---

### Step 3: Reverse Process (Training)

**Train denoising network to predict noise:**

```python
# Predict noise
epsilon_pred = denoising_network(x_t, t)

# Could also condition on text/class
# epsilon_pred = denoising_network(x_t, t, text_embedding)
```

---

### Step 4: Loss Calculation

**Compute loss between predicted and actual noise:**

```python
# Mean Squared Error loss
loss = F.mse_loss(epsilon_pred, epsilon)

# Optional: Weighted loss based on timestep
# weights = get_loss_weights(t)
# loss = weights * F.mse_loss(epsilon_pred, epsilon, reduction='none')
# loss = loss.mean()
```

---

### Step 5: Parameter Update

**Update model parameters to minimize loss using gradient descent:**

```python
# Zero gradients
optimizer.zero_grad()

# Backpropagation
loss.backward()

# Gradient clipping (optional, for stability)
torch.nn.utils.clip_grad_norm_(
    denoising_network.parameters(), 
    max_norm=1.0
)

# Update parameters
optimizer.step()
```

---

### Step 6: Iterate

**Repeat process for multiple epochs until model converges:**

```python
num_epochs = 1000
num_iterations_per_epoch = len(dataloader)

for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(dataloader):
        # Steps 2-5
        loss = train_step(batch, denoising_network, 
                         beta_schedule, optimizer)
        
        # Logging
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Iter {batch_idx}, Loss: {loss:.4f}")
    
    # Save checkpoint
    if epoch % 10 == 0:
        save_checkpoint(denoising_network, optimizer, epoch)
    
    # Sample and evaluate
    if epoch % 50 == 0:
        samples = sample(denoising_network, num_samples=16)
        save_images(samples, f"samples_epoch_{epoch}.png")
        fid_score = compute_fid(samples, real_images)
        print(f"FID Score: {fid_score:.2f}")
```

---

## Complete Training Algorithm

```python
def train_diffusion_model(
    dataset,
    denoising_network,
    beta_schedule,
    num_epochs=1000,
    batch_size=128,
    learning_rate=1e-4
):
    """
    Complete training algorithm for diffusion model.
    """
    # 1. Initialize
    optimizer = AdamW(denoising_network.parameters(), lr=learning_rate)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for batch in dataloader:
            # 2. Forward process: Add noise
            t = torch.randint(0, T, (batch_size,))
            x_t, epsilon = forward_process(batch, t, beta_schedule)
            
            # 3. Reverse process: Predict noise
            epsilon_pred = denoising_network(x_t, t)
            
            # 4. Compute loss
            loss = F.mse_loss(epsilon_pred, epsilon)
            
            # 5. Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # 6. Logging
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
        
        # Checkpointing and evaluation
        if epoch % 50 == 0:
            save_checkpoint(denoising_network, optimizer, epoch)
            evaluate_model(denoising_network, beta_schedule)
    
    return denoising_network
```

---

## Training Optimizations

**Techniques to speed up training:**

### 1. Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    epsilon_pred = denoising_network(x_t, t)
    loss = F.mse_loss(epsilon_pred, epsilon)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Benefits:**
- 2-3x faster training
- Lower memory usage
- Minimal quality loss

---

### 2. Gradient Accumulation

```python
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    loss = train_step(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Benefits:**
- Effective larger batch size
- Reduces memory requirements

---

### 3. EMA (Exponential Moving Average)

```python
ema_model = copy.deepcopy(denoising_network)
ema_decay = 0.9999

# After each update
for ema_param, param in zip(ema_model.parameters(), 
                            denoising_network.parameters()):
    ema_param.data = ema_decay * ema_param.data + \
                     (1 - ema_decay) * param.data
```

**Benefits:**
- Smoother parameter updates
- Better sample quality
- More stable generation

---

### 4. Multi-GPU Training

```python
denoising_network = torch.nn.DataParallel(denoising_network)
# or
denoising_network = torch.nn.parallel.DistributedDataParallel(
    denoising_network
)
```

**Benefits:**
- Faster training
- Handle larger batches
- Scale to massive datasets

---

## Training Resources

**Typical requirements for high-quality image generation:**

**Small model (CIFAR-10, 32×32):**
- 1-4 GPUs
- ~100,000 iterations
- ~1-2 days

**Medium model (CelebA, 256×256):**
- 4-8 GPUs
- ~500,000 iterations
- ~1-2 weeks

**Large model (ImageNet, 256×256):**
- 16-64 GPUs
- ~1,000,000 iterations
- ~2-4 weeks

**Very large model (Stable Diffusion scale):**
- 256-512 GPUs
- ~1,000,000+ iterations
- ~1-2 months
- $1-10 million in compute costs

---

## Sampling

![Full Sampling Process](images/diffusion_full_sampling.png)
*Image showing a grid of 100 steps, from 0 to 99, illustrating the gradual transformation of random noise into a clear image of a smiling face.*

Once the model is trained, you can **generate new images by sampling from the learned distribution**.

This involves **starting with pure noise** and **iteratively applying the reverse process** to remove the noise.

---

## Sampling Process

**Mathematical formulation:**
```python
x_0 ~ p_θ(x_0 | x_T)
```

**Where:**
- **x_T**: Initial pure noise (random Gaussian)
- **p_θ(x_0 | x_T)**: Learned distribution
- **x_0**: Final generated image

---

## Sampling Algorithm

### Step 1: Start with Noise

**Initialize with pure random noise:**

```python
# Sample from standard Gaussian
x_T = torch.randn(batch_size, channels, height, width)

# Example: Generate 4 images of size 256×256×3
x_T = torch.randn(4, 3, 256, 256)
```

**This is:** Pure random noise, no structure, no image content.

---

### Step 2: Iterative Denoising

**For each timestep from T to 1, use denoising network:**

```python
def sample(denoising_network, shape, T, beta_schedule):
    """
    Generate samples using trained diffusion model.
    
    Args:
        denoising_network: Trained model
        shape: Shape of images to generate
        T: Number of timesteps
        beta_schedule: Noise schedule used during training
    
    Returns:
        x_0: Generated images
    """
    # Start with pure noise
    x_t = torch.randn(shape)
    
    # Iteratively denoise
    for t in reversed(range(T)):
        # Current timestep
        t_tensor = torch.full((shape[0],), t, dtype=torch.long)
        
        # Predict noise
        epsilon_pred = denoising_network(x_t, t_tensor)
        
        # Compute denoising parameters
        alpha_t = 1 - beta_schedule[t]
        alpha_bar_t = compute_alpha_bar(beta_schedule, t)
        
        # Remove predicted noise
        x_t = (1 / torch.sqrt(alpha_t)) * (
            x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * epsilon_pred
        )
        
        # Add noise (except for last step)
        if t > 0:
            noise = torch.randn_like(x_t)
            sigma_t = compute_sigma(beta_schedule, t)
            x_t = x_t + sigma_t * noise
    
    return x_t  # This is x_0, the generated image
```

---

### Step 3: Final Sample

**After T steps, x_0 is the generated image:**

```python
# Generate 16 samples
samples = sample(
    denoising_network=trained_model,
    shape=(16, 3, 256, 256),
    T=1000,
    beta_schedule=cosine_schedule
)

# samples now contains 16 generated images!
```

---

## Sampling Variations

### 1. DDPM (Denoising Diffusion Probabilistic Models)

**Original algorithm:**
- Uses full reverse process
- T steps (typically 1000)
- High quality, but slow

```python
# Add stochastic noise at each step
if t > 0:
    z = torch.randn_like(x_t)
    x_t = x_t + sigma_t * z
```

---

### 2. DDIM (Denoising Diffusion Implicit Models)

**Deterministic sampling:**
- Skip steps for faster generation
- 50-100 steps instead of 1000
- Slight quality trade-off

```python
# Deterministic update (no noise added)
x_t_minus_1 = sqrt(alpha_bar_{t-1}) * x_0_pred + \
              sqrt(1 - alpha_bar_{t-1}) * epsilon_pred
```

**Advantages:**
✅ 10-20x faster
✅ Deterministic (same noise → same image)
✅ Interpolation in latent space

---

### 3. Ancestral Sampling

**Add noise proportional to predicted uncertainty:**

```python
sigma_t = eta * sqrt((1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)) * \
          sqrt(1 - alpha_t)
```

Where **eta** controls stochasticity:
- eta = 0: Deterministic (DDIM)
- eta = 1: Fully stochastic (DDPM)

---

## Conditional Sampling

**Generate images conditioned on class, text, or other information:**

### Class-Conditional

```python
# During sampling, provide class label
class_label = 207  # "golden retriever"
samples = sample_conditional(
    denoising_network,
    shape=(16, 3, 256, 256),
    condition=class_label
)
```

---

### Text-Conditional (Text-to-Image)

```python
# Encode text prompt
text_prompt = "a cat in a hat sitting on a mat"
text_embedding = text_encoder(text_prompt)

# Sample with text guidance
samples = sample_text_conditional(
    denoising_network,
    text_embedding=text_embedding,
    guidance_scale=7.5,
    num_steps=50
)
```

---

### Image-Conditional (Image-to-Image)

```python
# Start from noisy version of source image
source_image = load_image("photo.jpg")
x_t = forward_process(source_image, t=500)  # Partially noise

# Denoise with different prompt
samples = sample_conditional(
    denoising_network,
    initial_noise=x_t,
    text_prompt="make it look like a painting",
    start_step=500
)
```

---

## Sampling Speed vs Quality Trade-off

| Method | Steps | Time | Quality |
|--------|-------|------|---------|
| **DDPM** | 1000 | ~5 min | Highest |
| **DDIM** | 100 | ~30 sec | Very high |
| **DDIM** | 50 | ~15 sec | High |
| **Fast samplers** | 20-30 | ~5 sec | Good |

**Note:** Times approximate for 512×512 image on GPU.

---

## Sampling Tips

**For best results:**

✅ **More steps = better quality:**
- Use 50-100 steps for good balance
- Use 1000 steps for maximum quality

✅ **Adjust guidance scale:**
- Low (1-5): More creative, diverse
- Medium (7-10): Balanced
- High (15-20): Strong text adherence, less diverse

✅ **Use EMA model:**
- Use exponential moving average of parameters
- Better sample quality

✅ **Proper noise initialization:**
- Use random seed for reproducibility
- Sample from N(0, 1) distribution

---

## Data Assumptions

Diffusion models make **certain assumptions about the data**:

---

## 1. Markov Property

**The diffusion process exhibits the Markov property.**

### What is the Markov Property?

**Definition:** Each step depends only on the **immediately preceding step**, not the entire history.

**Mathematical formulation:**

**Forward process:**
```
q(x_t | x_{t-1}, x_{t-2}, ..., x_0) = q(x_t | x_{t-1})
```

Only depends on x_{t-1}, not on x_{t-2}, ..., x_0.

**Reverse process:**
```
p(x_{t-1} | x_t, x_{t+1}, ..., x_T) = p(x_{t-1} | x_t)
```

Only depends on x_t, not on future states.

---

### Why This Matters

**Benefits:**

✅ **Computational efficiency:**
- Don't need to store entire history
- Can compute each step independently
- Enables parallel training

✅ **Mathematical tractability:**
- Simplifies analysis
- Clear probabilistic framework
- Provable properties

✅ **Practical implementation:**
- Straightforward algorithms
- Scalable to long sequences
- Memory efficient

**If Markov property didn't hold:**
- Would need to condition on entire history
- Much more complex
- Computationally prohibitive

---

## 2. Static Data Distribution

**Diffusion models are trained on a fixed dataset**, and they learn to represent the **underlying distribution of this data**.

### What This Means

**Assumption:** The data distribution is **static** (doesn't change) during training.

**In practice:**
```
Training data: {x_1, x_2, ..., x_N} ~ p_data(x)

Model learns: p_θ(x) ≈ p_data(x)
```

The distribution p_data(x) is assumed constant.

---

### Implications

**Strengths:**

✅ **Clear training objective:**
- Learn fixed distribution
- Measurable progress
- Convergence guarantees

✅ **Generalization:**
- Generate new samples from learned distribution
- Interpolate between training examples
- Capture data characteristics

**Limitations:**

❌ **No adaptation to new data:**
- Can't learn from new distributions without retraining
- No online learning
- Fixed knowledge

❌ **Distribution shift:**
- If test distribution differs from training, quality degrades
- Model doesn't know what it doesn't know
- May hallucinate when extrapolating

---

### Handling Distribution Shifts

**Solutions:**

**Fine-tuning:**
```python
# Fine-tune on new data distribution
fine_tuned_model = continue_training(
    pretrained_model,
    new_dataset,
    num_epochs=100
)
```

**Domain adaptation:**
- Techniques to adapt to new domains
- Transfer learning approaches

**Continual learning:**
- Methods to learn from streaming data
- Avoid catastrophic forgetting

---

## 3. Smoothness Assumption

**While not a strict requirement**, diffusion models often perform well when the **data distribution is smooth**.

### What is Smoothness?

**Definition:** Small changes in input result in small changes in output.

**Mathematically:**
```
If ||x_1 - x_2|| is small, then ||f(x_1) - f(x_2)|| is also small
```

---

### Why Smoothness Helps

**Benefits for diffusion models:**

✅ **Gradual denoising:**
- Small noise → small change in image
- Denoising network learns smooth transitions
- Easier optimization

✅ **Interpolation:**
- Points between training examples are meaningful
- Can generate novel but realistic samples
- Smooth latent space

✅ **Stability:**
- Training more stable
- Less prone to artifacts
- Consistent outputs

---

### Examples

**Smooth distributions:**

**Natural images:**
```
Neighboring pixels correlated
Smooth color transitions
Gradual texture changes
→ Smooth distribution
```

**Audio:**
```
Continuous waveforms
Temporal correlations
Smooth frequency changes
→ Smooth distribution
```

---

**Non-smooth distributions:**

**Discrete data:**
```
Binary images (black/white only)
Categorical data
Sharp boundaries
→ Less smooth
```

**Solutions for non-smooth data:**
- Embed in continuous space
- Use specialized architectures
- Adjust noise schedule

---

## Consequences of Violating Assumptions

**What happens if assumptions don't hold:**

### Violating Markov Property

❌ **Longer-range dependencies needed:**
- Standard diffusion may struggle
- Need modified architectures
- Possible solutions: Attention mechanisms, longer context

---

### Violating Static Distribution

❌ **Distribution shift:**
- Model generates from training distribution
- Poor results on out-of-distribution data
- Need retraining or adaptation

**Example:**
```
Training: Photos from 1990s
Testing: Modern high-res photos
→ Quality degradation
```

---

### Violating Smoothness

❌ **Harder training:**
- Less stable optimization
- Potential artifacts
- May need more capacity

**Solutions:**
- Larger models
- More training data
- Custom noise schedules

---

## Summary: Diffusion Models

### Core Concepts

**Two-phase process:**
1. **Forward:** Gradually add noise (x₀ → x_T)
2. **Reverse:** Learn to remove noise (x_T → x₀)

**Key components:**
- **Noise schedule**: Controls noise addition rate
- **Denoising network**: Predicts noise to remove (typically U-Net)
- **Training**: Learn to predict noise via MSE loss
- **Sampling**: Iteratively denoise from random noise

---

### Text-to-Image Generation

**Steps:**
1. **Text encoding**: Convert prompt to embedding (CLIP, T5)
2. **Conditional denoising**: Guide denoising with text
3. **Iterative sampling**: Denoise noise → image
4. **Final output**: Generated image matching text

---

### Advantages

✅ **High-quality generation**: State-of-the-art image quality
✅ **Stable training**: No adversarial dynamics, no mode collapse
✅ **Flexibility**: Easy to condition on various inputs
✅ **Theoretical foundation**: Well-understood probabilistic framework
✅ **Diversity**: Excellent sample diversity

---

### Challenges

❌ **Slow generation**: Requires many denoising steps (50-1000)
❌ **Computational cost**: Expensive training and inference
❌ **Large models**: Significant resource requirements
❌ **Sampling time**: Minutes per image (improving with research)

---

### Assumptions

**1. Markov property**: Each step depends only on previous step
**2. Static distribution**: Training data distribution is fixed
**3. Smoothness**: Data distribution is relatively smooth

---

### Applications

**Current uses:**
- Text-to-image generation (DALL-E 2, Stable Diffusion, Midjourney)
- Image editing (inpainting, outpainting)
- Super-resolution and enhancement
- Audio and video generation
- 3D generation
- Scientific applications (drug discovery, materials science)

**Diffusion models represent the current state-of-the-art** in generative modeling, powering most modern text-to-image systems and continuing to evolve rapidly!
