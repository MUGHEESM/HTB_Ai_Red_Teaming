# Recurrent Neural Networks

Recurrent Neural Networks (RNNs) are a class of artificial neural networks specifically designed to handle sequential data, where the order of the data points matters. Unlike traditional feedforward neural networks, which process data in a single pass, RNNs have a unique structure that allows them to maintain a "memory" of past inputs. This memory enables them to capture temporal dependencies and patterns within sequences, making them well-suited for tasks like natural language processing, speech recognition, and time series analysis.

## Handling Sequential Data

![RNN Sequence Processing](images/05%20-%20Recurrent%20Neural%20Networks_0.png)

*Sequence processing diagram: RNN cells process inputs 'The cat sat on the mat' with hidden states and predictions at each step, ending the sequence.*

The key to understanding how RNNs handle sequential data lies in their recurrent connections. These connections create loops within the network, allowing information to persist and be passed from one step to the next. Imagine an RNN processing a sentence word by word. As it encounters each word, it considers the current input and incorporates information from the previous words, effectively "remembering" the context.

This process can be visualized as a chain of repeating modules, each representing a time step in the sequence. At each step, the module takes two inputs:

- The current input in the sequence (e.g., a word in a sentence)
- The hidden state from the previous time step encapsulates the information learned from past inputs.

The module then performs calculations and produces two outputs:

- Output for the current time step (e.g., a prediction of the next word)
- An updated hidden state is passed to the next time step in the sequence.

This cyclical flow of information allows the RNN to learn patterns and dependencies across the entire sequence, enabling it to understand context and make informed predictions.

For example, consider the sentence, "The cat sat on the mat." An RNN processing this sentence would:

1. Start with an initial hidden state (usually set to 0).
2. Process the word "The," and update its hidden state based on this input.
3. Process the word "cat," considering both the word itself and the hidden state now containing information about "The."
4. Continue processing each word this way, accumulating context in the hidden state at each step.

By the time the RNN reaches the word "mat," its hidden state would contain information about the entire preceding sentence, allowing it to make a more accurate prediction about what might come next.

## The Vanishing Gradient Problem

While RNNs excel at processing sequential data, they can suffer from a significant challenge known as the vanishing gradient problem. This problem arises during training, specifically when using backpropagation through time (BPTT) to update the network's weights.

In BPTT, the gradients of the loss function are calculated and propagated back through the network to adjust the weights and improve the model's performance. However, as the gradients travel back through the recurrent connections, they can become increasingly smaller, eventually vanishing to near zero. This vanishing gradient hinders the network's ability to learn long-term dependencies, as the weights associated with earlier inputs receive minimal updates.

The vanishing gradient problem is particularly pronounced in RNNs due to the repeated multiplication of gradients across time steps. If the gradients are small (less than 1), their product diminishes exponentially as they propagate back through the network. This means that the influence of earlier inputs on the final output becomes negligible, limiting the RNN's ability to capture long-range dependencies.

## LSTMs and GRUs

To address the vanishing gradient problem, researchers have developed specialized RNN architectures, namely Long-Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) networks. These architectures introduce gating mechanisms that control the flow of information through the network, allowing them to better capture long-term dependencies.

![LSTM Cell](images/05%20-%20Recurrent%20Neural%20Networks_1.png)

*LSTM cell diagram: Input and previous hidden state pass through input, forget, and output gates to update cell and hidden states.*

LSTMs incorporate memory cells that can store information over extended periods. These cells are equipped with three gates:

**Input gate:** Regulates the flow of new information into the memory cell.

**Forget gate:** Controls how much of the existing information in the memory cell is retained or discarded.

**Output gate:** Determines what information from the memory cell is output to the next time step.

These gates enable LSTMs to selectively remember or forget information, mitigating the vanishing gradient problem and allowing them to learn long-term dependencies.

![GRU Cell](images/05%20-%20Recurrent%20Neural%20Networks_2.png)

*GRU cell diagram: Input and previous hidden state pass through update and reset gates to candidate activation, updating the hidden state.*

GRUs offer a simpler alternative to LSTMs, with only two gates:

**Update gate:** Controls how much of the previous hidden state is retained.

**Reset gate:** Determines how much of the previous hidden state is combined with the current input.

GRUs achieve comparable performance to LSTMs in many tasks while being computationally more efficient due to their reduced complexity.

LSTMs and GRUs have proven highly effective in overcoming the vanishing gradient problem, leading to significant advancements in sequence modeling tasks, including machine translation, speech recognition, and sentiment analysis.

## Bidirectional RNNs

In addition to the standard RNNs that process sequences in a forward direction, there are also bidirectional RNNs. These networks process the sequence in both forward and backward directions simultaneously. This allows them to capture information from past and future contexts, which can be beneficial in tasks where the entire sequence is available, such as natural language processing.

A bidirectional RNN consists of two RNNs, one processing the sequence from left to right and the other from right to left. The hidden states of both RNNs are combined at each time step to produce the final output. This approach enables the network to consider the entire context surrounding each element in the sequence, leading to improved performance in many tasks.

**Recurrent Neural Networks (RNNs)** are a class of artificial neural networks specifically designed to handle **sequential data**, where **the order of the data points matters**.

### What Makes RNNs Special?

**Unlike traditional feedforward neural networks:**

**Feedforward Networks:**
- Process data in a **single pass**
- No memory of previous inputs
- Each input processed independently
- **Good for**: Static data (images, fixed-size inputs)

**Recurrent Networks:**
- Have **loops** that allow information to persist
- Maintain a **"memory"** of past inputs
- Process sequences step-by-step
- **Good for**: Sequential data (text, speech, time series)

---

## Why RNNs?

**Sequential data is everywhere:**

**Natural Language:**
- Sentences: Word order matters
- "Dog bites man" ≠ "Man bites dog"
- Context from previous words

**Speech:**
- Audio signals over time
- Phonemes depend on previous sounds
- Temporal patterns

**Time Series:**
- Stock prices over time
- Weather data
- Sensor readings

**Video:**
- Sequence of frames
- Motion over time
- Temporal events

**Traditional neural networks fail at these tasks because:**
- ❌ No memory of previous inputs
- ❌ Fixed input size required
- ❌ Cannot handle variable-length sequences
- ❌ No temporal modeling

**RNNs solve these problems by:**
- ✅ Maintaining hidden state (memory)
- ✅ Processing variable-length sequences
- ✅ Capturing temporal dependencies
- ✅ Sharing weights across time steps

---

## Applications of RNNs

RNNs excel at tasks involving sequential data:

**Natural Language Processing:**
- ✅ **Language modeling**: Predicting next word
- ✅ **Machine translation**: English → French
- ✅ **Sentiment analysis**: Positive/negative classification
- ✅ **Text generation**: Creating new text
- ✅ **Named entity recognition**: Finding names, places

**Speech Processing:**
- ✅ **Speech recognition**: Audio → Text
- ✅ **Speaker identification**: Who is speaking?
- ✅ **Speech synthesis**: Text → Audio

**Time Series Analysis:**
- ✅ **Stock price prediction**: Future values
- ✅ **Weather forecasting**: Temperature, rainfall
- ✅ **Anomaly detection**: Unusual patterns
- ✅ **Energy consumption**: Predicting demand

**Video Analysis:**
- ✅ **Action recognition**: What's happening?
- ✅ **Video captioning**: Describing video content
- ✅ **Video generation**: Creating new frames

---

## RNN Structure

### Basic RNN Architecture

**Key components:**

```
Input sequence: x₁, x₂, x₃, ..., xₜ
Hidden states: h₁, h₂, h₃, ..., hₜ
Outputs: y₁, y₂, y₃, ..., yₜ
```

**At each time step t:**
- **Input**: xₜ (current element in sequence)
- **Previous hidden state**: hₜ₋₁ (memory from past)
- **Current hidden state**: hₜ (updated memory)
- **Output**: yₜ (prediction at this step)

---

## Recurrent Connections

**The key to RNNs: Recurrent connections create loops within the network.**

### Unrolled RNN Visualization

**Folded view (compact):**
```
    ┌─────────┐
xₜ→ │   RNN   │ →yₜ
    │    ↻    │
    └─────────┘
```

The loop (↻) represents recurrence: output feeds back as input.

---

**Unrolled view (over time):**
```
Time:     t=1        t=2        t=3        t=4
          
Input:    x₁         x₂         x₃         x₄
          ↓          ↓          ↓          ↓
        ┌───┐      ┌───┐      ┌───┐      ┌───┐
        │RNN│─────→│RNN│─────→│RNN│─────→│RNN│
        └───┘      └───┘      └───┘      └───┘
          ↓          ↓          ↓          ↓
Output:   y₁         y₂         y₃         y₄

Hidden:   h₁         h₂         h₃         h₄
        (passed to next step)
```

**Each box is the same RNN cell with shared weights!**

---

## How RNNs Work: Step by Step

### Mathematical Formulation

At each time step **t**, the RNN performs these computations:

**Hidden state update:**
```
hₜ = tanh(Wₓₕ · xₜ + Wₕₕ · hₜ₋₁ + bₕ)
```

**Output computation:**
```
yₜ = Wₕy · hₜ + by
```

**Where:**
- **xₜ**: Input at time t
- **hₜ**: Hidden state at time t (memory)
- **hₜ₋₁**: Previous hidden state
- **yₜ**: Output at time t
- **Wₓₕ**: Input-to-hidden weights
- **Wₕₕ**: Hidden-to-hidden weights (recurrent)
- **Wₕy**: Hidden-to-output weights
- **bₕ, by**: Bias terms
- **tanh**: Activation function

---

## RNN Computation Process

**Step-by-step for sequence [x₁, x₂, x₃]:**

**Initialization:**
```
h₀ = 0  (or small random values)
```

**Time step 1:**
```
h₁ = tanh(Wₓₕ·x₁ + Wₕₕ·h₀ + bₕ)
y₁ = Wₕy·h₁ + by
```

**Time step 2:**
```
h₂ = tanh(Wₓₕ·x₂ + Wₕₕ·h₁ + bₕ)  ← Uses h₁ (memory!)
y₂ = Wₕy·h₂ + by
```

**Time step 3:**
```
h₃ = tanh(Wₓₕ·x₃ + Wₕₕ·h₂ + bₕ)  ← Uses h₂ (memory!)
y₃ = Wₕy·h₃ + by
```

**Key insight:** Each step uses hidden state from previous step, creating **memory** of sequence.

---

## Handling Sequential Data

![RNN Sequence Processing](images/05%20-%20Recurrent%20Neural%20Networks_0.png)
*Sequence processing diagram: RNN cells process inputs 'The cat sat on the mat' with hidden states and predictions at each step, ending the sequence.*

**The key to understanding how RNNs handle sequential data lies in their recurrent connections.**

These connections create loops within the network, allowing **information to persist and be passed from one step to the next**.

---

## Example: Processing a Sentence

**Sentence:** "The cat sat on the mat"

**How RNN processes it word by word:**

### Initial State
```
h₀ = [0, 0, 0, ..., 0]  (initial hidden state)
```

---

### Step 1: Process "The"
```
Input: x₁ = embedding("The")
h₁ = tanh(Wₓₕ·x₁ + Wₕₕ·h₀ + bₕ)
y₁ = predict("cat")

Hidden state h₁ now contains information about "The"
```

---

### Step 2: Process "cat"
```
Input: x₂ = embedding("cat")
h₂ = tanh(Wₓₕ·x₂ + Wₕₕ·h₁ + bₕ)  ← Uses h₁ (knows about "The")
y₂ = predict("sat")

Hidden state h₂ now contains information about "The cat"
```

---

### Step 3: Process "sat"
```
Input: x₃ = embedding("sat")
h₃ = tanh(Wₓₕ·x₃ + Wₕₕ·h₂ + bₕ)  ← Knows about "The cat"
y₃ = predict("on")

Hidden state h₃ now contains information about "The cat sat"
```

---

### Step 4: Process "on"
```
Input: x₄ = embedding("on")
h₄ = tanh(Wₓₕ·x₄ + Wₕₕ·h₃ + bₕ)  ← Knows about "The cat sat"
y₄ = predict("the")

Hidden state h₄ now contains information about "The cat sat on"
```

---

### Step 5: Process "the"
```
Input: x₅ = embedding("the")
h₅ = tanh(Wₓₕ·x₅ + Wₕₕ·h₄ + bₕ)  ← Full context
y₅ = predict("mat")

Hidden state h₅ now contains information about "The cat sat on the"
```

---

### Step 6: Process "mat"
```
Input: x₆ = embedding("mat")
h₆ = tanh(Wₓₕ·x₆ + Wₕₕ·h₅ + bₕ)  ← Complete sentence context
y₆ = predict("<end>")

Hidden state h₆ contains information about entire sentence!
```

---

## Information Flow Through Time

**This process can be visualized as a chain of repeating modules:**

```
"The"    "cat"    "sat"    "on"     "the"    "mat"
  ↓        ↓        ↓        ↓        ↓        ↓
[RNN] → [RNN] → [RNN] → [RNN] → [RNN] → [RNN]
  ↓        ↓        ↓        ↓        ↓        ↓
 h₁   →   h₂   →   h₃   →   h₄   →   h₅   →   h₆
  ↓        ↓        ↓        ↓        ↓        ↓
"cat"    "sat"    "on"    "the"    "mat"   "<end>"
```

**Each module:**
- Takes **two inputs**: current word (xₜ) + previous hidden state (hₜ₋₁)
- Produces **two outputs**: prediction (yₜ) + updated hidden state (hₜ)

---

## The Power of Hidden States

**By the time the RNN reaches "mat", its hidden state contains information about:**
- ✅ "The" (article introducing subject)
- ✅ "cat" (subject of sentence)
- ✅ "sat" (action/verb)
- ✅ "on" (preposition indicating location)
- ✅ "the" (article introducing object)

**This accumulated context allows it to:**
- Make accurate predictions about what comes next
- Understand sentence structure
- Capture dependencies between words
- Generate coherent continuations

---

## Weight Sharing Across Time

**Important: The same RNN cell (with same weights) is used at each time step!**

**Weights:**
- **Wₓₕ**: Same for all time steps
- **Wₕₕ**: Same for all time steps (recurrent)
- **Wₕy**: Same for all time steps

**Benefits:**
✅ **Parameter efficiency**: Don't need separate weights for each position
✅ **Generalization**: Works on sequences of any length
✅ **Translation invariance**: Pattern detected anywhere in sequence

---

## Types of RNN Architectures

**Different input/output configurations:**

### 1. One-to-Many
```
Input: Single value
Output: Sequence
Example: Image captioning (image → sentence)

    x
    ↓
  [RNN] → [RNN] → [RNN] → [RNN]
    ↓       ↓       ↓       ↓
    y₁      y₂      y₃      y₄
```

---

### 2. Many-to-One
```
Input: Sequence
Output: Single value
Example: Sentiment analysis (sentence → positive/negative)

  x₁      x₂      x₃      x₄
  ↓       ↓       ↓       ↓
[RNN] → [RNN] → [RNN] → [RNN]
                          ↓
                          y
```

---

### 3. Many-to-Many (Same Length)
```
Input: Sequence
Output: Sequence (same length)
Example: Part-of-speech tagging (word → tag)

  x₁      x₂      x₃      x₄
  ↓       ↓       ↓       ↓
[RNN] → [RNN] → [RNN] → [RNN]
  ↓       ↓       ↓       ↓
  y₁      y₂      y₃      y₄
```

---

### 4. Many-to-Many (Different Length)
```
Input: Sequence
Output: Different-length sequence
Example: Machine translation (English → French)

  x₁      x₂      x₃            [RNN] → [RNN] → [RNN]
  ↓       ↓       ↓               ↓       ↓       ↓
[RNN] → [RNN] → [RNN] → [context] y₁      y₂      y₃
         Encoder                    Decoder
```

---

## Training RNNs: Backpropagation Through Time

**RNNs are trained using Backpropagation Through Time (BPTT):**

**Forward pass:**
1. Process sequence step by step
2. Compute hidden states h₁, h₂, ..., hₜ
3. Compute outputs y₁, y₂, ..., yₜ
4. Compute loss

**Backward pass:**
1. Compute gradients at final time step
2. **Propagate gradients backward through time**
3. Update weights using accumulated gradients

**Key difference from standard backpropagation:**
- Gradients flow backward through time steps
- Same weights updated from multiple time steps
- More computationally expensive

---

## The Vanishing Gradient Problem

While RNNs excel at processing sequential data, they suffer from a **significant challenge**: the **vanishing gradient problem**.

### What is the Vanishing Gradient Problem?

**Problem arises during training**, specifically when using **Backpropagation Through Time (BPTT)** to update the network's weights.

**In BPTT:**
1. Gradients of loss function are calculated
2. Propagated **backward through the network**
3. Used to adjust weights and improve performance

**However:**
- As gradients travel back through recurrent connections
- They can become **increasingly smaller**
- Eventually **vanishing to near zero**

---

## Why Gradients Vanish

**Mathematical explanation:**

Hidden state update:
```
hₜ = tanh(Wₓₕ·xₜ + Wₕₕ·hₜ₋₁ + bₕ)
```

**Gradient computation (chain rule):**
```
∂L/∂hₜ₋ₖ = ∂L/∂hₜ · ∂hₜ/∂hₜ₋₁ · ∂hₜ₋₁/∂hₜ₋₂ · ... · ∂hₜ₋ₖ₊₁/∂hₜ₋ₖ
```

**Each term involves:**
```
∂hₜ/∂hₜ₋₁ = tanh'(...) · Wₕₕ
```

**Problem:**
- **tanh' ≤ 1** (derivative of tanh is at most 1)
- If **|Wₕₕ| < 1**, then each term **< 1**
- **Product of many terms < 1**: Exponentially decreasing
- **Result**: Gradient → 0 as we go back in time

---

## Consequences of Vanishing Gradients

**Impact on learning:**

❌ **Cannot learn long-term dependencies:**
- Early inputs have negligible gradient
- Weights for early inputs barely updated
- Network "forgets" distant past

❌ **Limited memory:**
- Effective memory span: ~5-10 time steps
- Cannot capture dependencies beyond this range

❌ **Poor performance on long sequences:**
- Fails on tasks requiring long-term memory
- Cannot connect cause and effect far apart

---

## Example: Long Sentence

**Sentence:** "The cat, which we saw yesterday in the park, **was** very cute."

**Task:** Predict verb form ("was" vs "were")

**Challenge:**
- Subject is "cat" (singular)
- But "cat" is 10 words before "was"
- Standard RNN: Gradient vanishes before reaching "cat"
- **Result**: Cannot learn subject-verb agreement

---

## The Exploding Gradient Problem

**Opposite problem can also occur:**

**If |Wₕₕ| > 1:**
- Gradients **grow exponentially**
- Can lead to **numerical instability**
- Weights receive huge updates
- Training becomes unstable

**Solutions:**
- **Gradient clipping**: Cap gradient magnitude
- **Weight regularization**: Penalize large weights
- **Careful initialization**: Set weights appropriately

---

## Vanishing vs Exploding Gradients

| Problem | Cause | Effect | Solution |
|---------|-------|--------|----------|
| **Vanishing** | \|Wₕₕ\| < 1, many time steps | Gradients → 0, can't learn long-term | LSTMs, GRUs |
| **Exploding** | \|Wₕₕ\| > 1, many time steps | Gradients → ∞, unstable training | Gradient clipping |

---

## LSTMs and GRUs

To address the vanishing gradient problem, researchers developed specialized RNN architectures:

### 1. Long Short-Term Memory (LSTM)
### 2. Gated Recurrent Unit (GRU)

**Key innovation:** **Gating mechanisms** that control the flow of information.

---

## Long Short-Term Memory (LSTM)

![LSTM Cell](images/05%20-%20Recurrent%20Neural%20Networks_1.png)
*LSTM cell diagram: Input and previous hidden state pass through input, forget, and output gates to update cell and hidden states.*

**LSTMs incorporate memory cells** that can store information over **extended periods**.

### LSTM Architecture

**Key components:**

**1. Cell state (Cₜ):**
- **Long-term memory**
- Information highway running through time
- Modified by gates

**2. Hidden state (hₜ):**
- **Short-term memory**
- Filtered version of cell state
- Output to next layer

**3. Three gates:**
- **Forget gate**: What to discard from memory
- **Input gate**: What new information to store
- **Output gate**: What to output from memory

---

## LSTM: Three Gates

### 1. Forget Gate

**Purpose:** Decide what information to **discard** from cell state.

**Computation:**
```
fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)
```

**Where:**
- **σ**: Sigmoid function (outputs 0 to 1)
- **fₜ**: Forget gate activation
- **hₜ₋₁**: Previous hidden state
- **xₜ**: Current input

**Output:**
- **0**: Forget completely
- **1**: Remember completely
- **0.5**: Partially remember

**Example:** "The cat was cute. **The dog** was playful."
- Forget gate: Forget "cat" when processing "dog"

---

### 2. Input Gate

**Purpose:** Decide what **new information** to store in cell state.

**Two steps:**

**a) Input gate:** What to update
```
iₜ = σ(Wi · [hₜ₋₁, xₜ] + bi)
```

**b) Candidate values:** What values to add
```
C̃ₜ = tanh(Wc · [hₜ₋₁, xₜ] + bc)
```

**Combined:** New information to add
```
New info = iₜ * C̃ₜ
```

**Example:** "The cat was cute. **The dog** was playful."
- Input gate: Store information about "dog"

---

### 3. Output Gate

**Purpose:** Decide what to **output** from cell state.

**Computation:**
```
oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)
```

**Hidden state update:**
```
hₜ = oₜ * tanh(Cₜ)
```

**Output gate filters cell state to produce hidden state.**

---

## LSTM: Cell State Update

**Cell state update combines forget and input:**

```
Cₜ = fₜ * Cₜ₋₁ + iₜ * C̃ₜ
      ↑            ↑
   Forget old   Add new
```

**This equation is key to LSTM success:**
- **fₜ * Cₜ₋₁**: Selectively forget old information
- **iₜ * C̃ₜ**: Selectively add new information
- **Additive**: Gradients flow more easily (no repeated multiplication)

---

## LSTM: Complete Computation

**At each time step t:**

**1. Forget gate:**
```
fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)
```

**2. Input gate:**
```
iₜ = σ(Wi · [hₜ₋₁, xₜ] + bi)
C̃ₜ = tanh(Wc · [hₜ₋₁, xₜ] + bc)
```

**3. Cell state update:**
```
Cₜ = fₜ * Cₜ₋₁ + iₜ * C̃ₜ
```

**4. Output gate:**
```
oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)
hₜ = oₜ * tanh(Cₜ)
```

---

## Why LSTMs Solve Vanishing Gradients

**Key insight:** **Additive cell state update**

**Standard RNN:**
```
hₜ = tanh(W·hₜ₋₁ + ...)
```
- Gradient: **Repeated multiplication** by W
- If |W| < 1: Vanishes exponentially

**LSTM:**
```
Cₜ = fₜ * Cₜ₋₁ + iₜ * C̃ₜ
```
- Gradient: **Additive path** through cell state
- Can flow backward without vanishing
- Forget gate can = 1 (perfect gradient flow)

**Result:**
✅ Can learn dependencies **hundreds** of steps back
✅ Gradients remain **stable**
✅ Effective memory over **long sequences**

---

## Gated Recurrent Unit (GRU)

![GRU Cell](images/05%20-%20Recurrent%20Neural%20Networks_2.png)
*GRU cell diagram: Input and previous hidden state pass through update and reset gates to candidate activation, updating the hidden state.*

**GRUs offer a simpler alternative to LSTMs**, with only **two gates** instead of three.

### GRU Architecture

**Key simplifications:**
- **No separate cell state** (only hidden state)
- **Two gates** instead of three:
  - **Update gate** (combines forget + input gates)
  - **Reset gate** (controls use of previous hidden state)
- **Fewer parameters** → Faster training

---

## GRU: Two Gates

### 1. Update Gate

**Purpose:** Controls how much of **previous hidden state** to retain and how much **new information** to add.

**Computation:**
```
zₜ = σ(Wz · [hₜ₋₁, xₜ] + bz)
```

**Where:**
- **zₜ**: Update gate activation (0 to 1)
- **1**: Keep old hidden state (ignore new input)
- **0**: Replace with new hidden state (forget old)

**Combines LSTM's forget and input gates!**

---

### 2. Reset Gate

**Purpose:** Determines how much of the **previous hidden state** to use when computing **candidate hidden state**.

**Computation:**
```
rₜ = σ(Wr · [hₜ₋₁, xₜ] + br)
```

**Where:**
- **rₜ**: Reset gate activation (0 to 1)
- **1**: Use full previous hidden state
- **0**: Ignore previous hidden state (reset)

**Allows model to drop irrelevant information.**

---

## GRU: Hidden State Update

**Candidate hidden state:**
```
h̃ₜ = tanh(Wh · [rₜ * hₜ₋₁, xₜ] + bh)
```
- **rₜ * hₜ₋₁**: Reset gate controls previous state usage
- **h̃ₜ**: New candidate hidden state

**Final hidden state:**
```
hₜ = (1 - zₜ) * hₜ₋₁ + zₜ * h̃ₜ
      ↑                 ↑
   Keep old         Use new
```
- **Update gate** interpolates between old and new
- **Linear interpolation** allows gradient flow

---

## GRU: Complete Computation

**At each time step t:**

**1. Update gate:**
```
zₜ = σ(Wz · [hₜ₋₁, xₜ] + bz)
```

**2. Reset gate:**
```
rₜ = σ(Wr · [hₜ₋₁, xₜ] + br)
```

**3. Candidate hidden state:**
```
h̃ₜ = tanh(Wh · [rₜ * hₜ₋₁, xₜ] + bh)
```

**4. Hidden state update:**
```
hₜ = (1 - zₜ) * hₜ₋₁ + zₜ * h̃ₜ
```

---

## LSTM vs GRU Comparison

| Feature | LSTM | GRU |
|---------|------|-----|
| **Gates** | 3 (forget, input, output) | 2 (update, reset) |
| **Cell State** | Separate (Cₜ) | Combined with hidden state |
| **Parameters** | More | Fewer |
| **Computation** | Slower | Faster |
| **Memory** | Explicit long-term memory | Implicit in hidden state |
| **Performance** | Slightly better on some tasks | Comparable on many tasks |
| **Training** | More data needed | More efficient |
| **Use Cases** | Very long sequences | Moderate-length sequences |

---

## When to Use Which?

**Use LSTM when:**
- ✅ Very long sequences (100+ steps)
- ✅ Complex long-term dependencies
- ✅ Sufficient training data
- ✅ Computational resources available
- ✅ Need explicit memory control

**Use GRU when:**
- ✅ Moderate-length sequences
- ✅ Limited training data
- ✅ Need faster training
- ✅ Resource-constrained environments
- ✅ Simpler model preferred

**Use Standard RNN when:**
- ✅ Very short sequences (<10 steps)
- ✅ Simple patterns
- ✅ Baseline comparison

---

## Performance Comparison

**LSTMs and GRUs have proven highly effective** in overcoming the vanishing gradient problem.

**Achievements:**

✅ **Machine translation**: Google Translate, DeepL
✅ **Speech recognition**: Siri, Alexa, Google Assistant
✅ **Sentiment analysis**: Opinion mining, review classification
✅ **Text generation**: GPT predecessors, language models
✅ **Time series forecasting**: Stock prices, weather
✅ **Video analysis**: Action recognition, captioning

**Performance characteristics:**

**LSTMs:**
- Best for very long dependencies
- More stable training
- Industry standard for many years

**GRUs:**
- Competitive performance
- Faster training (30% less time)
- Fewer parameters (simpler model)

**Modern Note:** Transformers (attention-based models) have largely replaced RNNs for many NLP tasks, but RNNs still valuable for time series and online learning.

---

## Bidirectional RNNs

**Standard RNNs process sequences in one direction** (left to right). **Bidirectional RNNs (BiRNNs)** process sequences in **both directions** simultaneously.

### Why Bidirectional?

**In many tasks, future context is also important:**

**Example 1: Sentence completion**
- "The **___** was very cute."
- Need to look ahead to "cute" to predict "cat" or "puppy"

**Example 2: Named entity recognition**
- "Paris is the capital of **France**."
- Need context before and after "France" to classify as country

**Example 3: Machine translation**
- Entire sentence available before translation
- Can use full context for each word

---

## Bidirectional RNN Architecture

**A bidirectional RNN consists of two RNNs:**

**Forward RNN:**
- Processes sequence **left to right**
- Captures **past context**
- Hidden states: h₁→, h₂→, h₃→, ...

**Backward RNN:**
- Processes sequence **right to left**
- Captures **future context**
- Hidden states: h₁←, h₂←, h₃←, ...

**Combined at each time step:**
```
hₜ = [hₜ→; hₜ←]  (concatenation)
```

---

## Bidirectional RNN Visualization

```
Forward pass (→):
x₁      x₂      x₃      x₄
↓       ↓       ↓       ↓
[RNN]→ [RNN]→ [RNN]→ [RNN]
h₁→     h₂→     h₃→     h₄→

Backward pass (←):
x₁      x₂      x₃      x₄
↓       ↓       ↓       ↓
[RNN]← [RNN]← [RNN]← [RNN]
h₁←     h₂←     h₃←     h₄←

Combined:
h₁      h₂      h₃      h₄
[h₁→]   [h₂→]   [h₃→]   [h₄→]
[h₁←]   [h₂←]   [h₃←]   [h₄←]
↓       ↓       ↓       ↓
y₁      y₂      y₃      y₄
```

---

## Bidirectional RNN Computation

**At each time step t:**

**Forward RNN:**
```
hₜ→ = tanh(Wₓₕ→·xₜ + Wₕₕ→·hₜ₋₁→ + bₕ→)
```

**Backward RNN:**
```
hₜ← = tanh(Wₓₕ←·xₜ + Wₕₕ←·hₜ₊₁← + bₕ←)
```

**Combined hidden state:**
```
hₜ = [hₜ→; hₜ←]
```

**Output:**
```
yₜ = Wₕy·hₜ + by
```

**Note:** Forward and backward RNNs have **separate weights**.

---

## Benefits of Bidirectional RNNs

✅ **Full context:**
- Access to past **and** future
- More informed predictions
- Better understanding of each element

✅ **Improved accuracy:**
- Often significantly better than unidirectional
- Especially for tasks with full sequence available

✅ **Applications:**
- Natural language processing (most tasks)
- Speech recognition (offline)
- Protein structure prediction
- Time series analysis (when future data available)

---

## Limitations of Bidirectional RNNs

❌ **Requires full sequence:**
- Cannot use for online/streaming tasks
- Need to see entire sequence before processing

❌ **Higher computational cost:**
- Two RNNs instead of one
- Double memory requirement
- Slower training and inference

❌ **Not suitable for:**
- Real-time applications
- Text generation (can't see future)
- Online predictions
- Streaming data

---

## Bidirectional LSTM/GRU

**Bidirectional concept extends to LSTM and GRU:**

**Bidirectional LSTM:**
- Forward LSTM + Backward LSTM
- Combines advantages of both
- Very powerful for sequence labeling

**Bidirectional GRU:**
- Forward GRU + Backward GRU
- Faster than BiLSTM
- Good trade-off between performance and speed

---

## RNN Summary

**Standard RNNs:**
- ✅ Handle sequential data
- ✅ Maintain memory through hidden states
- ❌ Suffer from vanishing gradients
- ❌ Limited to short-term dependencies

**LSTMs:**
- ✅ Solve vanishing gradient problem
- ✅ Learn long-term dependencies
- ✅ Explicit memory control (cell state)
- ❌ More parameters, slower

**GRUs:**
- ✅ Simpler than LSTMs
- ✅ Faster training
- ✅ Comparable performance
- ❌ Slightly less capacity

**Bidirectional RNNs:**
- ✅ Access to full context
- ✅ Better accuracy
- ❌ Requires full sequence
- ❌ Cannot use for online tasks

---

## Key Takeaways

**1. RNNs are designed for sequential data:**
- Maintain memory through hidden states
- Process sequences step by step
- Share weights across time

**2. Vanishing gradients limit standard RNNs:**
- Cannot learn long-term dependencies
- Gradients diminish exponentially backward in time

**3. LSTMs and GRUs solve this problem:**
- Gating mechanisms control information flow
- Enable learning of long-range dependencies
- Industry standard for sequence modeling

**4. Bidirectional RNNs use full context:**
- Process forward and backward
- Improved performance when full sequence available
- Not suitable for online/streaming tasks

**5. Modern context:**
- Transformers have largely replaced RNNs for NLP
- RNNs still valuable for time series, online learning, resource-constrained settings
- Understanding RNNs essential for deep learning fundamentals

