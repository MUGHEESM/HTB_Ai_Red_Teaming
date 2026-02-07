# Reinforcement Learning Algorithms

## Overview

**Reinforcement learning (RL)** introduces a unique paradigm in machine learning where an agent learns by interacting with an environment. Unlike supervised learning, which relies on labeled data, or unsupervised learning, which explores unlabeled data, RL focuses on **learning through trial and error**, guided by feedback in the form of rewards or penalties.

### Human Learning Analogy

This approach mimics how humans learn through experience, making RL particularly suitable for tasks that involve **sequential decision-making in dynamic environments**.

---

## Training a Dog Analogy

Think of it like **training a dog**:

1. **No Explicit Instructions**: You don't give the dog explicit instructions on sitting, staying, or fetching

2. **Positive Reinforcement**: You reward it with treats and praise when it performs the desired actions

3. **Correction**: You correct it when it doesn't perform well

4. **Learning Through Feedback**: The dog learns to associate specific actions with positive outcomes through trial, error, and feedback

This is exactly how RL works—learning optimal behavior through interaction and feedback.

---

## How Reinforcement Learning Works

In RL, an **agent** interacts with an **environment** by taking actions and observing the consequences.

### The Learning Process

1. **Agent Takes Action**: The agent performs an action in the current state

2. **Environment Responds**: The environment transitions to a new state

3. **Reward/Penalty**: The environment provides feedback through rewards or penalties

4. **Policy Update**: The agent updates its strategy based on this feedback

5. **Repeat**: This cycle continues, guiding the agent toward learning an **optimal policy**

### Policy

A **policy** is a strategy for selecting actions that maximize cumulative rewards over time. It's the core of what the agent learns.

---

## Categories of Reinforcement Learning

Reinforcement learning algorithms can be broadly categorized into:

### 1. Model-Based RL

**Approach:** The agent learns a model of the environment, which it uses to predict future states and plan its actions.

**Analogy:** This is like **having a map of a maze** before navigating it:
- The agent can use this map to plan the most efficient path to the goal
- Reduces the need for trial and error
- Can simulate outcomes before taking real actions

**Advantages:**
- More sample efficient (requires fewer interactions)
- Can plan ahead
- Enables reasoning about consequences

**Challenges:**
- Model accuracy is critical
- Computationally expensive to learn and maintain model
- Model errors can compound

**Examples:**
- Dyna-Q
- Model Predictive Control
- AlphaZero (uses learned model for planning)

---

### 2. Model-Free RL

**Approach:** The agent learns directly from experience without explicitly modeling the environment.

**Analogy:** This is like **navigating a maze without a map**:
- The agent relies solely on trial and error
- Learns from the feedback from the environment
- Gradually improves its policy by exploring different paths
- Learns from the rewards or penalties it receives

**Advantages:**
- Simpler to implement
- No need to learn environment dynamics
- Can work in complex environments where modeling is difficult

**Challenges:**
- Less sample efficient (requires more interactions)
- Cannot plan ahead
- Must learn from direct experience

**Examples:**
- Q-Learning
- SARSA
- Policy Gradient methods
- Actor-Critic algorithms

---

## Core Concepts in Reinforcement Learning

Understanding Reinforcement Learning (RL) requires a grasp of its core concepts. These concepts provide the foundation for understanding how agents learn and interact with their environment to achieve their goals.

---

## Agent

The **agent** is the learner and decision-maker in an RL system.

### Responsibilities

- Interacts with the environment
- Takes actions
- Observes consequences
- Learns from experiences
- Updates its policy

### Goal

The agent aims to learn an **optimal policy** that maximizes cumulative rewards over time.

### Examples

- **Robot navigating a maze**: Makes movement decisions
- **Program playing a game**: Decides on moves
- **Self-driving car**: Navigates through traffic
- **Trading algorithm**: Makes buy/sell decisions
- **Chatbot**: Selects responses

In each case, the agent makes decisions and learns from its experiences.

---

## Environment

The **environment** is the external system or context in which the agent operates.

### Definition

It encompasses everything outside the agent, including:
- The physical world
- A simulated world
- A game board
- Market conditions
- Social interactions

### Role

The environment:
- Responds to the agent's actions
- Provides feedback through rewards or penalties
- Transitions between states
- Defines the rules of interaction

### Examples

- **Maze navigation**: The maze itself, with walls, paths, and goal location
- **Game play**: The game with its rules and opponent moves
- **Stock trading**: The market with price movements
- **Robot control**: The physical world with physics constraints

---

## State

The **state** represents the current situation or condition of the environment.

### Definition

It provides a snapshot of the relevant information that the agent needs to make informed decisions.

### Components

The state can include various aspects of the environment:
- The agent's position
- The positions of other objects
- Velocity and acceleration
- Inventory levels
- Game scores
- Any other relevant variables

### Examples

**Robot in maze:**
- Current location (x, y coordinates)
- Surrounding walls
- Distance to goal

**Chess game:**
- Current configuration of the chessboard
- All piece positions
- Whose turn it is
- Castling rights

**Self-driving car:**
- Car position and velocity
- Positions of other vehicles
- Traffic light states
- Road conditions

### State Representation

**Fully Observable (Complete State):**
- Agent can see all relevant information
- Example: Chess (can see entire board)

**Partially Observable (Incomplete State):**
- Agent has limited information
- Example: Poker (can't see opponents' cards)

---

## Action

An **action** is a move or decision that the agent makes that affects the environment.

### Characteristics

- Selected based on the current state
- Selected based on the agent's policy
- Causes the environment to transition to a new state
- May be deterministic or stochastic

### Action Spaces

**Discrete Actions:**
- Finite number of possible actions
- Example: Up, Down, Left, Right

**Continuous Actions:**
- Infinite number of possible actions within a range
- Example: Steering angle (-30° to +30°)

### Examples

**Maze navigation:**
- Move forward
- Turn left
- Turn right
- Stay in place

**Game playing:**
- Move a piece
- Make a specific play
- Pass

**Robot control:**
- Apply torque to joints
- Set motor speeds
- Grasp object

---

## Reward

The **reward** is feedback from the environment indicating the desirability of the agent's action.

### Characteristics

- A scalar value (single number)
- Can be positive, negative, or zero
- Positive rewards encourage the agent to repeat the action
- Negative rewards (penalties) discourage it

### Purpose

The agent's goal is to **maximize cumulative rewards over time**, not just immediate rewards.

### Examples

**Maze navigation:**
- **Positive reward**: Getting closer to the goal (+1)
- **Negative reward**: Hitting a wall (-1)
- **Large positive reward**: Reaching the goal (+100)

**Game playing:**
- **Positive**: Winning (+1)
- **Negative**: Losing (-1)
- **Neutral**: Draw (0)

**Self-driving car:**
- **Positive**: Staying in lane (+0.1)
- **Negative**: Collision (-100)
- **Negative**: Traffic violation (-10)

### Reward Shaping

Designing good reward functions is crucial:
- **Too sparse**: Agent may never learn (e.g., only reward at goal)
- **Too dense**: May guide agent to local optima
- **Well-shaped**: Guides agent while allowing exploration

---

## Policy

A **policy** is a strategy or mapping from states to actions that the agent follows.

### Definition

It determines which action the agent should take in a given state.

### Goal

The agent aims to learn an **optimal policy** that maximizes cumulative rewards.

### Types of Policies

#### 1. Deterministic Policy

**Notation:** π(s) = a

**Characteristic:** Always selects the same action in a given state.

**Example:** In state "traffic light is red," always take action "stop."

---

#### 2. Stochastic Policy

**Notation:** π(a|s) = probability of taking action a in state s

**Characteristic:** Selects actions with certain probabilities.

**Example:** In state "at junction," 70% turn left, 30% turn right.

**Advantage:** Allows for exploration and handles uncertainty.

---

## Value Function

The **value function** estimates the long-term value of being in a particular state or taking a specific action.

### Purpose

It predicts the **expected cumulative reward** that the agent can obtain from that state or action onwards.

### Importance

The value function is a crucial component in many RL algorithms, as it guides the agent towards choosing actions that lead to higher long-term rewards.

---

## Types of Value Functions

### 1. State-Value Function

**Notation:** V^π(s)

**Definition:** Estimates the expected cumulative reward from starting in a given state and following a particular policy.

**Formula:**
```python
V^π(s) = E[Rt + γRt+1 + γ²Rt+2 + ... | St = s, π]
```

**Interpretation:** "How good is it to be in this state?"

**Example:** In maze, a state close to the goal has high value.

---

### 2. Action-Value Function (Q-Function)

**Notation:** Q^π(s, a)

**Definition:** Estimates the expected cumulative reward from taking a specific action in a given state and then following a particular policy.

**Formula:**
```python
Q^π(s, a) = E[Rt + γRt+1 + γ²Rt+2 + ... | St = s, At = a, π]
```

**Interpretation:** "How good is it to take this action in this state?"

**Example:** In maze, moving towards the goal has high Q-value.

---

## Discount Factor

The **discount factor (γ)** is a parameter in RL that determines the present value of future rewards.

### Range

γ ranges between **0 and 1**:

| γ Value | Interpretation | Behavior |
|---------|----------------|----------|
| **γ = 0** | Only immediate rewards matter | Myopic, short-sighted |
| **γ = 0.9** | Balanced approach | Reasonable foresight |
| **γ = 0.99** | Long-term focus | Far-sighted planning |
| **γ = 1** | All future rewards equally important | Infinite horizon |

### Mathematical Effect

**Cumulative reward formula:**
```python
G_t = R_t + γR_{t+1} + γ²R_{t+2} + γ³R_{t+3} + ...
```

Where:
- **G_t**: Total return from time t
- **R_t**: Immediate reward at time t
- **γ**: Discount factor

### Why Discount?

**Reasons for discounting:**

1. **Mathematical Convenience**: Ensures infinite sums converge
2. **Uncertainty**: Future is less certain than present
3. **Preference**: Immediate rewards are often more valuable
4. **Finite Horizon Approximation**: Models limited planning horizon

### Practical Implications

**High γ (e.g., 0.99):**
- Agent plans far ahead
- Good for strategic tasks
- May be slower to learn

**Low γ (e.g., 0.5):**
- Agent focuses on immediate rewards
- Good for quick reactions
- May miss long-term benefits

---

## Episodic vs. Continuous Tasks

### Episodic Tasks

**Definition:** The agent interacts with the environment in **episodes**, each ending at a terminal state.

**Characteristics:**
- Clear beginning and end
- Episode terminates at terminal state
- Agent resets for new episode
- Learning can happen between episodes

**Examples:**
- **Maze navigation**: Episode ends when goal is reached or time runs out
- **Game playing**: Episode ends when game is won/lost
- **Robot pick-and-place**: Episode ends when object is placed or dropped

**Notation:**
- Episode: τ = (s₀, a₀, r₁, s₁, a₁, r₂, ..., s_T)
- T: Terminal time step

---

### Continuous Tasks

**Definition:** Have **no explicit end** and continue indefinitely.

**Characteristics:**
- No terminal states
- Runs continuously
- Learning happens online
- Must balance exploration and exploitation constantly

**Examples:**
- **Controlling a robot arm**: Continuous control with no end
- **Stock trading**: Market operates continuously
- **Temperature control**: Thermostat runs continuously
- **Process control**: Manufacturing processes

**Challenge:** Discount factor (γ < 1) is essential to ensure rewards don't sum to infinity.

---

## The Exploration-Exploitation Tradeoff

A fundamental challenge in RL is balancing:

### Exploration

**Definition:** Trying new actions to discover potentially better strategies.

**Why Important:** Prevents getting stuck in local optima.

**Risk:** May waste time on suboptimal actions.

---

### Exploitation

**Definition:** Using current knowledge to maximize rewards.

**Why Important:** Leverages what has been learned.

**Risk:** May miss better alternatives.

---

### Strategies

**ε-greedy:**
- Exploit with probability 1-ε
- Explore with probability ε
- Simple but effective

**Softmax/Boltzmann:**
- Choose actions proportionally to Q-values
- More sophisticated than ε-greedy

**Upper Confidence Bound (UCB):**
- Favor actions with high uncertainty
- Optimistic exploration

---

## Summary

Reinforcement Learning is a powerful paradigm for learning through interaction:

**Key Characteristics:**
- **Trial and Error**: Learn from experience, not labeled data
- **Sequential Decision Making**: Actions affect future states
- **Delayed Rewards**: Actions may have long-term consequences
- **Agent-Environment Interaction**: Continuous feedback loop

**Core Components:**
- **Agent**: The learner and decision-maker
- **Environment**: The world the agent interacts with
- **State**: Current situation
- **Action**: Agent's decision
- **Reward**: Feedback signal
- **Policy**: Strategy for action selection
- **Value Function**: Estimate of long-term returns

**Types of RL:**
- **Model-Based**: Learn environment model, plan ahead
- **Model-Free**: Learn directly from experience

**Key Concepts:**
- **Policy (π)**: Maps states to actions
- **Value Function (V/Q)**: Estimates long-term value
- **Discount Factor (γ)**: Balances present vs. future
- **Episodes**: Task structure (episodic vs. continuous)

**Fundamental Challenge:**
- **Exploration vs. Exploitation**: Balance learning new things with using current knowledge

**Applications:**
- Game playing (Chess, Go, video games)
- Robotics (manipulation, navigation)
- Autonomous vehicles
- Resource management
- Recommendation systems
- Financial trading

RL provides a framework for agents to learn optimal behavior in complex, dynamic environments through interaction and feedback, making it one of the most exciting and promising areas of machine learning.
