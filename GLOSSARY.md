# 📖 Complete AI Research Glossary (`GLOSSARY.md`)

An alphabetical index of core machine learning, deep learning, and frontier AI research terms. Every term is defined in plain English with a link back to the primary module where it is introduced and implemented.

---

## A
- **Ablation Study**: The experimental practice of systematically removing or disabling individual components of a machine learning architecture or pipeline to determine which elements actually account for performance improvements. ([Module 19](19_Reading_and_Deconstructing_AI_Papers/README.md))
- **Action**: A choice or move executed by an agent in an environment that influences the subsequent state and expected reward. ([Module 13](13_Foundations_of_Reinforcement_Learning/README.md))
- **Actor-Critic**: A reinforcement learning architecture featuring two networks (or heads): an Actor that outputs action probabilities ($\pi_\theta$) and a Critic that estimates state baseline values ($V_\phi$). ([Module 14](14_Deep_RL_Policy_Gradients_and_PPO/README.md))
- **Adam (Adaptive Moment Estimation)**: An optimization algorithm that maintains exponentially decaying moving averages of past gradients (momentum) and past squared gradients (uncentered variance) to scale step sizes adaptively per parameter. ([Module 04](04_Intuitive_Calculus_and_Optimization/README.md))
- **Advantage Function**: A mathematical metric $A(s, a) = Q(s, a) - V(s)$ quantifying how much better or worse a specific chosen action performs compared to the average expected return in that state. ([Module 14](14_Deep_RL_Policy_Gradients_and_PPO/README.md))
- **Attention**: A dynamic neural routing mechanism that allows representation vectors at each token position to compute pairwise alignment scores with all other tokens, retrieving relevant context via weighted summation. ([Module 11](11_Attention_and_The_Transformer/README.md))
- **Autograd**: Automatic differentiation software engine that records operations performed on tensors during the forward pass in a directed acyclic graph (DAG) to compute gradients automatically during backpropagation. ([Module 07](07_Neural_Networks_and_Autograd_From_Scratch/README.md))
- **Autoregressive Generation**: The token generation protocol where a model iteratively predicts the single next token, appends it to the context sequence, and feeds the updated sequence back as input for subsequent predictions. ([Module 12](12_Building_GPT_From_Scratch/README.md))

## B
- **Backpropagation**: Reverse-mode automatic differentiation algorithm that applies the chain rule of calculus starting from the scalar loss output to calculate gradients with respect to every trainable weight in the network. ([Module 04](04_Intuitive_Calculus_and_Optimization/README.md), [Module 07](07_Neural_Networks_and_Autograd_From_Scratch/README.md))
- **Baseline**: A reference value subtracted from returns in policy gradient estimators to reduce variance without introducing mathematical bias. ([Module 14](14_Deep_RL_Policy_Gradients_and_PPO/README.md))
- **Batch Normalization**: A layer technique that normalizes feature activations across the batch dimension for each channel, popular in computer vision but largely replaced by LayerNorm in NLP. ([Module 09](09_Deep_Learning_Dynamics_and_Architectures/README.md))
- **Bellman Equation**: A recursive mathematical relationship stating that the value of a current state equals the immediate reward plus the discounted expected value of the next state. ([Module 13](13_Foundations_of_Reinforcement_Learning/README.md))
- **Bias-Variance Tradeoff**: The fundamental machine learning balance between underfitting (high bias, overly rigid assumptions) and overfitting (high variance, memorizing noise). ([Module 06](06_Classical_Machine_Learning/README.md))
- **Broadcasting**: The computational protocol allowing element-wise arithmetic operations between arrays of different shapes by conceptually expanding dimensions of size 1 without allocating duplicate memory. ([Module 02](02_NumPy_Vectorization_and_Data/README.md))
- **Byte-Pair Encoding (BPE)**: A subword tokenization algorithm that begins with individual raw UTF-8 byte tokens and iteratively merges the most frequently occurring adjacent pairs into new vocabulary tokens. ([Module 10](10_Tokenization_and_Byte_Pair_Encoding/README.md))

## C
- **Causal Mask**: A lower-triangular matrix of binary or infinite values applied to attention score matrices to prevent tokens at position $i$ from attending to future tokens $j > i$. ([Module 11](11_Attention_and_The_Transformer/README.md), [Module 12](12_Building_GPT_From_Scratch/README.md))
- **Chain Rule**: The fundamental rule of calculus stating that the derivative of a composite function $f(g(x))$ equals the product of their local derivatives: $\frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$. ([Module 04](04_Intuitive_Calculus_and_Optimization/README.md))
- **Chain-of-Thought (CoT)**: An inference prompting or generation technique where language models output sequential, intermediate reasoning steps before arriving at a final answer. ([Module 17](17_Reasoning_Models_and_Test_Time_Compute/README.md))
- **ChatML**: A standard XML-style token markup format used to distinguish system, user, and assistant conversational turns during instruction tuning. ([Module 15](15_Post_Training_and_Alignment_SFT_and_DPO/README.md))
- **Cosine Similarity**: The dot product of two vectors divided by the product of their Euclidean lengths, yielding a scale-invariant similarity metric bounded between $-1.0$ and $+1.0$. ([Module 03](03_Intuitive_Linear_Algebra/README.md))
- **Cross-Entropy Loss**: A loss function derived from information theory measuring the divergence between the true target probability distribution and a model's predicted probability distribution. ([Module 05](05_Intuitive_Probability_and_Information_Theory/README.md))

## D
- **Direct Preference Optimization (DPO)**: An alignment algorithm that optimizes language models directly on human preference pairs (chosen vs. rejected) using an implicit reward formulation, bypassing the need for a separate reward model or RL loop. ([Module 15](15_Post_Training_and_Alignment_SFT_and_DPO/README.md))
- **Discount Factor ($\gamma$)**: A scalar parameter between $0$ and $1$ in reinforcement learning determining the present value of future rewards. ([Module 13](13_Foundations_of_Reinforcement_Learning/README.md))
- **Dot Product**: An algebraic operation taking two equal-length vectors and returning a single scalar equal to the sum of their element-wise products, geometrically reflecting directional alignment. ([Module 03](03_Intuitive_Linear_Algebra/README.md))
- **Dropout**: A regularization technique during neural network training that randomly zeroes out a fraction of neuron activations to prevent co-adaptation. ([Module 08](08_PyTorch_Deep_Learning_Core/README.md))

## E
- **Embedding**: A continuous vector representation of discrete tokens where semantic relationships correspond to spatial proximity and geometric alignment. ([Module 10](10_Tokenization_and_Byte_Pair_Encoding/README.md))
- **Entropy (Shannon Entropy)**: The mathematical measure of the average uncertainty or information content produced by a stochastic probability distribution. ([Module 05](05_Intuitive_Probability_and_Information_Theory/README.md))
- **Expectation**: The probability-weighted arithmetic average of a random variable across all potential outcomes. ([Module 05](05_Intuitive_Probability_and_Information_Theory/README.md))

## G
- **GELU (Gaussian Error Linear Unit)**: A smooth non-linear activation function that weights inputs by their cumulative standard normal probability, widely used in GPT-2 and BERT. ([Module 09](09_Deep_Learning_Dynamics_and_Architectures/README.md))
- **Gradient**: A vector collecting all partial derivatives of a multivariable function, pointing in the direction of greatest instantaneous rate of increase. ([Module 04](04_Intuitive_Calculus_and_Optimization/README.md))
- **Group Relative Policy Optimization (GRPO)**: An efficient reinforcement learning algorithm introduced in DeepSeekMath that eliminates the Critic network by sampling a group of $G$ outputs per prompt and standardizing advantages across the group. ([Module 16](16_RLHF_and_Group_Relative_Policy_Optimization_GRPO/README.md), [Module 20](20_Capstone_GRPO_Loss_Walkthrough/README.md))

## I
- **Induction Head**: A two-layer transformer attention circuit that detects recurring token patterns $[A][B] \dots [A]$ and copies $[B]$ to enable in-context few-shot learning. ([Module 18](18_Mechanistic_Interpretability_and_AI_Safety/README.md))

## K
- **Kullback-Leibler (KL) Divergence**: An asymmetric statistical divergence measuring the relative entropy lost when approximating a true probability distribution $P$ with a model distribution $Q$. ([Module 05](05_Intuitive_Probability_and_Information_Theory/README.md), [Module 16](16_RLHF_and_Group_Relative_Policy_Optimization_GRPO/README.md))
- **KV Cache**: An inference optimization that stores previously computed Key and Value attention tensors in GPU memory to avoid redundant matrix operations during autoregressive generation. ([Module 11](11_Attention_and_The_Transformer/README.md))

## L
- **Layer Normalization (LayerNorm)**: A normalization layer that standardizes activations across all hidden features for each token independently, free from batch-size dependencies. ([Module 09](09_Deep_Learning_Dynamics_and_Architectures/README.md))
- **LoRA (Low-Rank Adaptation)**: A parameter-efficient fine-tuning method that freezes base model weights and trains low-rank decomposition matrices $B \cdot A$ for updates. ([Module 03](03_Intuitive_Linear_Algebra/README.md), [Module 15](15_Post_Training_and_Alignment_SFT_and_DPO/README.md))

## M
- **Markov Decision Process (MDP)**: A formal mathematical framework for decision-making under uncertainty consisting of states, actions, transition dynamics, rewards, and discount factors. ([Module 13](13_Foundations_of_Reinforcement_Learning/README.md))
- **Mechanistic Interpretability**: The reverse-engineering of deep neural network weights into human-comprehensible circuits and functional algorithms. ([Module 18](18_Mechanistic_Interpretability_and_AI_Safety/README.md))
- **Multi-Head Attention (MHA)**: An attention architecture that projects Queries, Keys, and Values into multiple lower-dimensional subspaces, allowing parallel computation across diverse relationships. ([Module 11](11_Attention_and_The_Transformer/README.md))

## P
- **Policy ($\pi$)**: A probabilistic or deterministic mapping defining an agent's chosen action given a specific observed state. ([Module 13](13_Foundations_of_Reinforcement_Learning/README.md))
- **Polysemanticity**: The neural network phenomenon where a single neuron activates for multiple unrelated semantic concepts. ([Module 18](18_Mechanistic_Interpretability_and_AI_Safety/README.md))
- **Process Reward Model (PRM)**: A reward model trained to evaluate and assign credit to each individual reasoning step rather than just the final answer. ([Module 17](17_Reasoning_Models_and_Test_Time_Compute/README.md))
- **Proximal Policy Optimization (PPO)**: A policy gradient algorithm that uses a clipped surrogate objective to prevent destabilizingly large policy updates during training. ([Module 14](14_Deep_RL_Policy_Gradients_and_PPO/README.md))

## R
- **Residual Connection (Skip Connection)**: An architectural connection $y = x + \mathcal{F}(x)$ passing input activations directly around a sub-layer, creating an unobstructed gradient path during backpropagation. ([Module 09](09_Deep_Learning_Dynamics_and_Architectures/README.md))
- **RMSNorm**: A high-efficiency variant of LayerNorm that scales activations by their root-mean-square without subtracting the mean, used in modern LLMs like LLaMA and DeepSeek. ([Module 09](09_Deep_Learning_Dynamics_and_Architectures/README.md))
- **Rotary Position Embedding (RoPE)**: A relative positional encoding mechanism that rotates token vectors in 2D planes by angles proportional to their sequence positions. ([Module 11](11_Attention_and_The_Transformer/README.md))

## S
- **Singular Value Decomposition (SVD)**: A matrix factorization technique expressing a matrix as $U \Sigma V^T$, establishing the mathematical foundation for low-rank approximation. ([Module 03](03_Intuitive_Linear_Algebra/README.md))
- **Sparse Autoencoder (SAE)**: An autoencoder trained with high sparsity penalties on neural activations to disentangle superimposed features into monosemantic, interpretable concepts. ([Module 18](18_Mechanistic_Interpretability_and_AI_Safety/README.md))
- **Superposition**: The mathematical phenomenon where neural networks represent more features than they have dimensions by packing them non-orthogonally. ([Module 18](18_Mechanistic_Interpretability_and_AI_Safety/README.md))
- **SwiGLU**: A bilinear gated linear unit using Swish activation, widely adopted in frontier foundation models. ([Module 09](09_Deep_Learning_Dynamics_and_Architectures/README.md))

## T
- **Temperature**: A hyperparameter scaling logits prior to softmax; lower values produce deterministic/greedy outputs, while higher values increase randomness. ([Module 12](12_Building_GPT_From_Scratch/README.md))
- **Test-Time Compute**: The practice of trading additional inference compute (via chain-of-thought tokens, search, or verification) for higher task accuracy. ([Module 17](17_Reasoning_Models_and_Test_Time_Compute/README.md))
