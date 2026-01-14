#set page(numbering: "1")

#set text(lang: "en", region: "GB")

#let make-venue = move(dy: -1.9cm, {
  box(rect(fill: blue.darken(30%), inset: 10pt, height: 2.5cm)[
    #set text(font: "TeX Gyre Pagella", fill: white, weight: 700, size: 20pt)
    #align(bottom)[University of Sussex]
  ])
  set text(22pt, font: "TeX Gyre Heros")
  box(pad(left: 10pt, bottom: 10pt, [Advanced Methods in Bio-Inspired AI]))
})

#let make-title(
  title,
  authors,
  abstract,
  keywords,
) = {
  set par(spacing: 1em)
  set text(font: "TeX Gyre Heros")
  
  par(
    justify: false,
    text(24pt, fill: rgb("004b71"), title, weight: "bold")
  )

  text(12pt,
    authors.enumerate()
    .map(((i, author)) => box[#author.name #super[#(i+1)]])
    .join(", ")
  )
  parbreak()

  for (i, author) in authors.enumerate() [
    #set text(8pt)
    #super[#(i+1)]
    #author.institution
    #link("mailto:" + author.mail) \
  ]

  v(8pt)
  set text(10pt)
  set par(justify: true)

  [
    #heading(outlined: false, bookmarked: false)[Abstract]
    #text(font: "TeX Gyre Pagella", abstract)
    #v(3pt)
    *Keywords:* #keywords.join(text(font: "TeX Gyre Pagella", "; "))
  ]
  v(18pt)
}

#let template(
    title: [],
    authors: (),
    date: [],
    doi: "",
    keywords: (),
    abstract: [],
    make-venue: make-venue,
    make-title: make-title,
    body,
) = {
    set page(
      paper: "a4",
      margin: (top: 1.9cm, bottom: 1in, x: 1.6cm),
      columns: 2
    )
    set par(justify: true)
    set text(10pt, font: "TeX Gyre Pagella")
    set list(indent: 8pt)
    // show link: set text(underline: false)
    show heading: set text(size: 11pt)
    show heading.where(level: 1): set text(font: "TeX Gyre Heros", fill: rgb("004b71"), size: 12pt)
    show heading: set block(below: 8pt)
    show heading.where(level: 1): set block(below: 12pt)

    place(make-venue, top, scope: "parent", float: true)
    place(
      make-title(title, authors, abstract, keywords), 
      top, 
      scope: "parent",
      float: true
    )


    show figure: align.with(center)
    show figure: set text(8pt)
    show figure.caption: pad.with(x: 10%)

    // show: columns.with(2)
    body
  }

#show: template.with(
  
  title: [Parallel Predictive Coding Networks for Robust Sensory Inference],
  authors: (
    (
      name: "Jason Fang (309734)",
      department: "Informatics",
      institution: "University of Sussex",
      city: "Brighton & Hove",
      country: "UK",
      mail: "",
    ),
  ),
  date: (
    year: 2022,
    month: "May",
    day: 17,
  ),
  keywords: (
    "Predictive Coding Networks",
    "Ensemble Diversity",
    "Robust Perception",
    "Sensory Corruption",
    "Hierarchical Inference",
  ),
  doi: "10.7891/120948510",
  abstract: [

Predictive coding networks (PCNs) are computational models inspired by how the brain processes information. They work by continuously making predictions about incoming sensory data and adjusting these predictions when errors occur, much like how our brains constantly update our understanding of what we see and hear. However, single predictive coding networks struggle when sensory information is corrupted by noise or degradation. This study investigates whether running multiple independent PCN models in parallel (horizontal parallelism), each with different characteristics, can improve the system's ability to maintain accurate predictions under challenging conditions. 

We trained ensembles of 10 PCN models to recognise handwritten digits (MNIST dataset), varying three key properties: initialisation diversity (starting each model with different random initial states), dynamics diversity (varying how quickly models process information), and architecture diversity (using different internal structures). Models were tested under progressively severe Gaussian noise corruption (σ = 0.0 to 0.5, where higher values represent more severe degradation). 

Results show that whilst dynamics diversity alone provides negligible benefit (\~0%), initialisation diversity yields +6.0% improvement in accuracy and architecture diversity +4.7%. Critically, combining all three mechanisms (MIXED diversity) produces +11.1% improvement at high corruption levels (σ=0.5), demonstrating synergistic effects—the combined benefit exceeds the sum of individual contributions. The ensemble's disagreement (variance) correlates strongly with corruption level, providing a reliable uncertainty signal: when models disagree, the system "knows" its predictions are unreliable. These findings suggest that diversity in cortical circuits—the brain's natural variation in neuron types, connection patterns, and temporal dynamics—may be a fundamental mechanism for robust perception under noisy real-world conditions.


  ],
)  



= 1 Introduction

== Background

The brain continuously infers (makes educated guesses about) the state of the world from noisy, ambiguous sensory input @Friston2010TheTheory. For example, when viewing a partially obscured object, your brain fills in missing information based on prior experience. Predictive coding theory proposes that the brain's cortical hierarchies (layered processing structures) achieve this through a simple principle: continuously generate predictions about incoming sensory signals, then minimise the difference (prediction error) between what was expected and what was actually received @Rao1999PredictiveCortex. When prediction errors are large, the brain updates its internal model; when errors are small, the current understanding is reinforced. This framework has gained substantial empirical support, explaining diverse phenomena from visual illusions (where predictions override sensory input) to action selection (predicting consequences of movements) @Clark2013WhateverMachine.

Despite its biological plausibility and theoretical elegance, predictive coding models face a critical challenge: robustness to sensory corruption. Real-world perception must operate reliably despite signal degradation from noise (random fluctuations), occlusion (missing information), and ambiguity (multiple possible interpretations). Whilst the brain exhibits remarkable robustness under such conditions—you can still recognise a friend's face in dim lighting or through a rain-spattered window—single predictive coding networks show significant performance degradation when confronted with corrupted inputs.

== The Diversity Hypothesis

Biological neural systems exhibit extensive diversity across multiple dimensions: neurons have varied shapes (morphologies), chemical signalling systems (neurotransmitter systems), response speeds (time constants), and connection patterns (connectivity) @Marder2011VariabilityFunction. Rather than being noise or inefficiency, this diversity may constitute a computational resource—a feature, not a bug. Ensemble coding theories propose that such diversity enables robust information processing: population-level representations (the combined activity of many diverse neurons), rather than individual neurons, carry sensory information @Pouget2000InformationCode. This is analogous to how polling many people produces more reliable estimates than asking a single person, provided their opinions are somewhat independent.

From an information-theoretic perspective, redundancy across partially independent estimators improves robustness to noise, as errors de-correlate while signal components reinforce @Shannon1948ATheory.

We hypothesise that horizontal parallelism in predictive coding—maintaining multiple diverse predictive hierarchies rather than a single canonical hierarchy—may provide analogous robustness benefits. Think of it like consulting multiple expert opinions before making a medical diagnosis: if the experts use different reasoning methods (diverse processes), their independent conclusions can be combined for a more reliable final judgement. Multiple parallel streams could make independent errors that cancel when aggregated, whilst reinforcing correct inferences.

== Research Question

*Does horizontal parallelism in predictive coding networks improve robustness and stability of sensory inference compared to a single predictive hierarchy?*

Specifically, we investigate:
1. Whether ensembles of parallel PCNs outperform single PCNs under sensory corruption
2. Which diversity mechanisms (initialisation, dynamics, architecture) contribute most to robustness
3. Whether diversity mechanisms combine synergistically or independently
4. Whether disagreement among parallel predictive coding networks reliably tracks input corruption, providing a proxy for uncertainty.

= 2 Methods


== Predictive Coding Network Architecture

We implemented predictive coding networks following the formulation of @Whittington2017AnApproximation. Each PCN consists of a hierarchical architecture with reciprocal (top-down and bottom-up) connections between layers. Each layer $l$ contains nodes with activity $bold(x)^l$ that are iteratively updated to minimise a global prediction-error energy function. This energy can be interpreted as a measure of surprise: high energy indicates a mismatch between predictions and sensory input, whilst low energy indicates accurate predictions. 

For a given scenario, the energy function that requires minimisation is:

$ E = sum_l norm(bold(x)^l - f(bold(W)^l bold(x)^(l-1)))^2 $

This sums across all layers the squared L2 norm of the difference between actual activity ($bold(x)^l$) and predicted activity ($f(bold(W)^l bold(x)^(l-1))$), where $bold(W)^l$ are connection weights and $f$ is a nonlinear activation function (hyperbolic tangent, tanh) applied element wise.


Inference (the process of settling on an interpretation of the input) proceeds by iteratively adjusting node activity to minimise this energy via gradient descent. 


The inference update equation computes activity changes for all neurons in a layer simultaneously using vector notation (bold symbols indicate vectors). This update implements gradient descent on the energy function—the formula below is mathematically equivalent to $Delta bold(x)^l = -mu (partial E)/(partial bold(x)^l)$, the negative gradient of energy with respect to activities:



$ Delta bold(x)^l = mu [(bold(epsilon)^(l+1) dot.o f'(bold(W)^(l+1) bold(x)^l)) bold(W)^(l+1)^T - bold(epsilon)^l] $


Here, $bold(x)^l$ is the vector of all neuron activities in layer $l$ (e.g., 300 values for the first hidden layer), and $Delta bold(x)^l$ is the corresponding vector of activity updates—each element $Delta x_i^l$ specifies how much neuron $i$ should change. The term $bold(epsilon)^l = bold(x)^l - f(bold(W)^l bold(x)^(l-1))$ is the prediction error vector at layer $l$ (the mismatch between predicted and actual activity for each neuron), $f'$ denotes the activation function derivative (applied elementwise), $dot.o$ represents elementwise multiplication, $bold(W)^T$ denotes the matrix transpose (flipping rows and columns to propagate errors in the reverse direction), and $mu$ is the inference step size (controlling how quickly the network settles). This update has two components: the first term $(bold(epsilon)^(l+1) dot.o f'(bold(W)^(l+1) bold(x)^l)) bold(W)^(l+1)^T$ represents top-down error signals (pulling activities to satisfy predictions from above), whilst the second term $-bold(epsilon)^l$ represents bottom-up error signals (pushing activities toward predictions from below).

The inference process operates differently during testing versus training, though both use the same iterative settling mechanism:

*Testing (Inference Phase)*: At test time, the network makes predictions on novel images it has never seen before. The input layer is clamped to the test image whilst the output and hidden layers are initialised to zero (or small random values). With weights frozen (no learning occurs), all non-input layers (both hidden and output) iteratively update their activities according to the inference rule above for 100 iterations. Each iteration performs a gradient descent step, moving activities toward lower energy states where prediction errors are minimised. The settling proceeds bidirectionally: top-down predictions from higher layers and bottom-up signals from lower layers simultaneously influence each layer's activity. After 100 iterations, the final classification is determined by selecting the output node with the highest activation—this settled state represents the network's best guess about which digit class is present, given its learned weights and the current input. This settling process can be understood as the network "explaining" the sensory input through the lens of its learned internal model: each layer negotiates with its neighbours through bidirectional error signals until a mutually consistent interpretation emerges across the hierarchy.

*Training (Learning Phase)*: For supervised classification, the network performs inference with both boundary layers clamped: the input layer is clamped to sensory data (the training image) and the output layer is clamped to one-hot encoded target labels (the correct digit category). With both ends fixed, only the hidden layers are free to update their activities. These hidden layers perform gradient descent for 100 iterations, with each iteration moving activities toward lower energy states via bidirectional signals—top-down from the clamped output layer and bottom-up from the clamped input layer. After 100 iterations, the hidden layers settle into states that minimise prediction error across the hierarchy, essentially finding intermediate representations that bridge the known input to the known output. After this inference phase, synaptic weights are updated once according to a local prediction-error minimisation rule:

$ bold(delta)^l = bold(epsilon)^l dot.o f'(bold(W)^l bold(x)^(l-1)) $

$ Delta bold(W)^l = eta (bold(x)^(l-1))^T bold(delta)^l $

The first equation computes $bold(delta)^l$, the modulated error signal at layer $l$ (prediction error $bold(epsilon)^l$ weighted by the activation derivative $f'$, applied elementwise via $dot.o$). The second equation updates the weights: the transpose $(bold(x)^(l-1))^T$ converts the input column vector into a row, and the matrix product with $bold(delta)^l$ forms the outer product. For each connection from input node $i$ to output node $j$, the weight change $Delta W_(i j)^l = eta x_i^(l-1) delta_j^l$. When both $x_i^(l-1)$ (the input node's value) and $delta_j^l$ (the modulated error) are large and positive, the weight increases substantially; when they have opposite signs, the weight decreases. This resembles Hebbian learning ("neurons that fire together, wire together"): connections strengthen when pre-synaptic activity and post-synaptic error signals are correlated, though here the correlation is error-modulated rather than purely activity-based. Learning rate $eta = 10^(-3)$ scales all updates.

The base architecture used 4 layers: [784, 300, 100, 10] nodes. The first layer (784 nodes) corresponds to flattened 28×28 pixel MNIST images (28 × 28 = 784), followed by two hidden layers (300 and 100 nodes for intermediate processing), and 10 output nodes representing the 10 digit classes (0-9).

== Parallel PCN Implementation

The parallel PCN model consists of $N$ independent PCN streams running in parallel. Each stream $i$ receives the same input image but performs independent inference. The final prediction combines outputs via ensemble averaging:

$ bold(p)_("ensemble") = 1/N sum_(i=1)^N bold(p)_i $

where $bold(p)_i$ is the output activation vector from stream $i$ (a 10-dimensional vector with one value per digit class). The averaged output is then used for classification by selecting the digit class with the highest activation.

We implemented three diversity mechanisms to differentiate parallel streams:

*Initialisation Diversity* (`init`): All models have identical architecture (same layer structure) and dynamics ($mu = 0.01$, same processing speed) but different random weight initialisations. In other words, the 10 models start with different random connection strengths but are otherwise identical. PyTorch's random state advances naturally between model creation, automatically producing different initial weights for each stream. This is like training 10 students with the same curriculum but giving each a different starting knowledge base.

*Dynamics Diversity* (`dynamics`): All models have identical architecture and weight initialisation (we reset the random state before creating each model to ensure they start identically) but varied inference step sizes. The parameter $mu$ is linearly spaced across $[0.005, 0.03]$, creating models with fast vs. slow inference dynamics. Fast models ($mu=0.03$) reach conclusions quickly but may overshoot the optimal solution; slow models ($mu=0.005$) take more iterations but settle more precisely. This is like having 10 identical experts who think at different speeds—some rush to quick judgements, others deliberate carefully.

The choice of $mu in [0.005, 0.03]$ was motivated by several considerations. First, $mu = 0.01$ represents a standard baseline step size for gradient descent in predictive coding networks. The selected range spans 0.5× to 3× this baseline, creating a 6-fold variation in convergence rates—sufficient to produce substantially different inference trajectories over 100 iterations. Whilst individual step sizes appear small, their cumulative effect is significant: 100 iterations amplify per-step differences into qualitatively distinct settling dynamics. Higher values ($mu > 0.05$) risk numerical instability and overshooting, whilst smaller ranges would reduce functional diversity. This parameterisation tests whether temporal processing differences alone—independent of structural or initialisation variation—contribute to ensemble robustness.

*Architecture Diversity* (`architecture`): All models have identical dynamics ($mu = 0.01$, same processing speed) and initial weight patterns (reset random state to ensure identical starting weights) but different layer configurations. Architectures vary between deep-narrow structures (e.g., [784, 200, 50, 10]—many layers with fewer nodes each, good for hierarchical feature extraction) and shallow-wide structures (e.g., [784, 400, 10]—fewer layers with more nodes each, good for direct pattern matching) whilst maintaining roughly similar total parameter counts. This is like having 10 experts with identical knowledge and thinking speed but different reasoning strategies—some prefer step-by-step hierarchical analysis, others prefer holistic pattern matching.

*MIXED Diversity* (`mixed`): Combines all three mechanisms—varied initialisation (different starting weights), varied $mu in [0.005, 0.03]$ (different processing speeds), and varied architectures (different internal structures)—to maximise diversity across parallel streams. This creates 10 models that differ in starting knowledge, reasoning strategy, AND thinking speed—maximally independent experts.

== Experimental Design

*Dataset*: We used the MNIST dataset—a standard benchmark consisting of 28×28 pixel greyscale images of handwritten digits (0-9). Each image has been centred and size-normalised, making it a clean test of digit recognition ability. To amplify ensemble benefits (smaller datasets make differences between single and parallel models more apparent) and reduce training time, we used a reduced training set of 2,000 samples and test set of 1,000 samples. Images were unnormalised (pixel values in [0,1], where 0 represents black and 1 represents white).

*Training Protocol*: All models (single baseline and parallel ensembles) were trained identically:
- 5 epochs, batch size 640
- 100 inference iterations per training sample
- Adam optimiser, learning rate $10^(-3)$, gradient clipping at 50
- Input and output layer activities clamped during training (supervised mode)

*Experimental Controls*: To ensure fair comparison, we reset the random seed (42) before training each model type and recreated data loaders, ensuring all models saw identical training batches in identical order. During testing, we reset seeds before each corruption level to ensure all models were evaluated on identically corrupted images.

*Corruption Protocol*: Robustness was assessed by systematically degrading test images with Gaussian noise (random pixel-wise perturbations drawn from a bell-curve distribution):

$ bold(x)_("corrupted") = "clamp"(bold(x)_("clean") + cal(N)(0, sigma^2), 0, 1) $

This adds random noise with standard deviation $sigma$ to each pixel, then clamps values to the valid range [0,1] to ensure pixel intensities remain valid. We tested corruption levels $sigma in {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}$, where $sigma = 0.0$ represents pristine images, $sigma = 0.3$ represents severe corruption, and $sigma = 0.5$ represents maximum corruption. This mimics real-world scenarios like poor lighting, camera noise, or transmission errors.

*Ensemble Size*: All parallel models used $N=10$ streams. This balances robustness benefits against computational cost, based on preliminary experiments showing diminishing returns beyond 10 models.

*Metrics*:
- Classification accuracy: proportion of correctly classified digits
- Accuracy degradation: difference between clean ($sigma=0$) and corrupted accuracy
- Ensemble variance: variance across model predictions, measuring disagreement
- Improvement: percentage gain of parallel over single PCN

== Implementation

The implementation extended an existing PyTorch-based predictive coding framework. Key code components:
- `ParallelPCModel` class managing ensemble of PCN streams
- Diversity mechanism control via random state management
- Corruption utilities for systematic noise injection
- Automated experimental pipeline with seed control and result logging

All experiments were conducted on CPU (Apple M1). Training time per model configuration: \~3-5 minutes.


= 3 Results

== Baseline Performance on Clean Data

Before examining robustness to corruption, we assessed baseline performance on pristine test images (@fig:clean-bar). Architecture diversity achieves the highest clean-data accuracy (76.8%), whilst initialisation and dynamics diversity show minimal impact. Critically, ensemble methods do not sacrifice baseline accuracy for robustness—architecture and MIXED diversity both improve clean performance (@fig:clean-improvement). However, these modest clean-data gains (+0% to +2%) pale compared to corruption benefits (+0% to +11.1%), indicating that diversity primarily enhances robustness rather than baseline discrimination.

#figure(
  image("clean_performance_bar.pdf"),
  caption: [
    *Baseline performance on clean data.* Accuracy on uncorrupted images (σ=0.0): Architecture (76.8%), MIXED (76.5%), Dynamics (75.0%), Init (74.8%), Single (74.8%). Architecture diversity's +2.0% advantage suggests varied structures capture complementary features even without corruption. N=2,000 training samples, 1,000 test samples.
  ]
) <fig:clean-bar>

#figure(
  image("clean_performance_improvement.pdf"),
  caption: [
    *Ensemble benefit on clean data.* Improvement over single PCN: Architecture +2.0%, MIXED +1.7%, Dynamics +0.2%, Init 0.0%. Modest clean gains contrast with corruption benefits (up to +11.1%), showing diversity primarily aids robustness, not baseline discrimination.
  ]
) <fig:clean-improvement>

== Overall Robustness Comparison

@fig:robustness shows three key patterns. First, all models degrade monotonically with corruption—the single PCN collapses from 74.8% (clean) to 38.7% at σ=0.5 (36.1 percentage points). Second, ensemble advantage scales with corruption severity: modest at σ=0.0 (+0-2%), substantial at σ=0.5 (+11.1% for MIXED). Third, the effectiveness hierarchy (MIXED > Init ≈ Arch >> Dyn ≈ Single) persists across all corruption levels, confirming temporal diversity alone provides negligible benefit. Ensemble benefits scale with task difficulty—harder problems demand diverse processing.

#figure(
  image("robustness_comparison.pdf"),
  caption: [
    *Robustness across corruption levels.* Accuracy vs. Gaussian noise (σ=0.0–0.5) for Single (red), Init (blue), Dynamics (green), Architecture (purple), and MIXED (orange). N=10 models per ensemble. MIXED maintains highest accuracy throughout; dynamics tracks single baseline. Mean over 1,000 test samples.
  ]
) <fig:robustness>

== Individual Diversity Mechanism Contributions

To quantify the isolated contribution of each diversity mechanism, we focus on performance at $sigma=0.5$ (maximum corruption)—chosen as the focal comparison point because it represents the most extreme tested corruption where differences between approaches are most pronounced.

@fig:bar-comparison summarises the accuracy hierarchy at $sigma=0.5$. The three diversity mechanisms show markedly different effectiveness:

*Initialisation Diversity*: Random seed variation alone produces +6.0% improvement (44.7% vs. 38.7% baseline). This suggests that different weight initialisations lead models to converge to different local minima (different stable solutions), creating complementary feature representations—each model learns to detect slightly different patterns. When averaged, errors in individual models cancel out whilst correct detections reinforce, improving overall accuracy.

*Dynamics Diversity*: Varying only inference speed (while maintaining identical weights and architecture) yields effectively zero improvement (\~0%, 38.6% vs. 38.7% baseline). Models with $mu=0.005$ (slow) and $mu=0.03$ (fast) reach similar accuracy when starting from identical initial states, indicating that inference speed alone does not create functionally diverse predictions. The near-perfect overlap with the single baseline confirms that temporal diversity alone provides no robustness benefit.

*Architecture Diversity*: Varying network structure alone produces +4.7% improvement (43.4% vs. 38.7% baseline). Different layer configurations (deep-narrow vs. shallow-wide) create distinct computational pathways, leading to partially independent errors despite identical initial weights.

These results demonstrate that diversity in initial conditions and computational structure matters significantly, while diversity in temporal dynamics contributes minimally when other factors are controlled.

#figure(
  image("accuracy_bar_chart.pdf"),
  caption: [
    *Accuracy at σ=0.5.* MIXED (49.8%), Init (44.7%), Arch (43.4%), Dyn (38.6%), Single (38.7%). MIXED's +11.1% gain represents 28.7% relative improvement, indicating synergistic effects.
  ]
) <fig:bar-comparison>

== Synergistic Effects in MIXED Diversity

MIXED diversity achieves +11.1% at σ=0.5 (49.8% vs. 38.7%)—exceeding the additive sum:

- Expected: $6.0% + (-0.1%) + 4.7% = 10.6%$
- Observed: $11.1%$

This +0.5% synergy demonstrates constructive interaction: combining varied initialisation, architecture, and dynamics creates maximally uncorrelated errors across multiple independent dimensions (@fig:corruption-improvement).

#figure(
  image("corruption_performance_improvement.pdf"),
  caption: [
    *Ensemble benefit at σ=0.5.* MIXED +11.1%, Init +6.0%, Arch +4.7%, Dyn ~0%. Contrast with @fig:clean-improvement: 5.5× amplification from clean to corrupted conditions. MIXED exceeds additive sum (+10.6%), showing constructive interaction.
  ]
) <fig:corruption-improvement>

== Uncertainty Quantification

Ensemble variance provides a natural uncertainty estimate. @fig:variance shows monotonic increase with corruption: 0.0026 (clean) to 0.058 (σ=0.5)—a 22-fold increase. At σ=0.3, variance reaches 0.043 (16.5× clean levels). This strong correlation means the ensemble "knows when it doesn't know"—high disagreement signals unreliable predictions. This calibrated uncertainty is crucial for real-world applications: knowing *when* you don't know enables systems to flag uncertain cases for human review.

#figure(
  image("ensemble_variance.pdf"),
  caption: [
    *Variance as uncertainty signal.* Prediction variance for MIXED ensemble (N=10) across σ=0.0–0.5. Color-coded: clean (dark blue), light (blue), moderate (yellow-orange), severe (orange), extreme (red-orange), maximum (dark red). Monotonic increase (0.0026→0.058) provides calibrated confidence signal.
  ]
) <fig:variance>

= 4 Discussion

== Principal Findings

Four key findings emerge:

*1. Ensemble benefits are substantial.* MIXED diversity maintains 49.8% accuracy at $sigma=0.5$ vs. 38.7% for single PCNs—a 28.7% relative gain representing qualitative differences in usability.

*2. Diversity sources differ dramatically.* Architecture and initialisation diversity contribute meaningfully (+4.7%, +6.0%), whilst dynamics alone contributes nothing (\~0%). This suggests functional and sampling diversity are critical; temporal diversity insufficient.

*3. Mechanisms combine synergistically.* MIXED outperforms additive expectations, indicating diversity across multiple dimensions produces maximally independent errors.

*4. Uncertainty is calibrated.* Ensemble variance increases 22-fold from clean to maximum corruption (0.0026→0.058), providing a reliable confidence signal—the ensemble "knows when it doesn't know."

== Interpretation: Why Does Diversity Help?

Ensemble learning theory explains the results @Dietterich2000EnsembleNets: independent errors cancel when averaged whilst correct inferences reinforce. The key is error decorrelation—models must fail differently. Our findings show:

- *Initialisation diversity* creates independent representations—each model develops distinct feature detectors based on starting conditions
- *Architecture diversity* implements distinct strategies—deep models build hierarchical features, shallow models perform direct pattern matching
- *Dynamics diversity fails* because temporal differences (fast/slow settling) don't translate to prediction differences when weights and architecture are identical. The path differs, but the destination remains similar.

This suggests final inference states matter more than trajectories—models with varied speeds converge to similar solutions when constrained by identical weights and structure.

The variance results reveal a fundamental property of PCN inference: when inputs are corrupted, the iterative settling process becomes increasingly sensitive to initial conditions and architectural constraints. Diverse models explore different regions of the energy landscape, and corrupted inputs create flatter, more ambiguous energy surfaces with multiple plausible local minima. The 22-fold variance increase indicates that PCNs naturally expose uncertainty through their settling dynamics—no explicit probabilistic machinery is required. This contrasts with feedforward networks, where uncertainty typically requires auxiliary mechanisms (dropout, Bayesian approximations). In PCNs, uncertainty emerges organically from the distributed, recurrent inference process itself.

== Biological Implications

The brain exhibits extensive diversity—neurons differ in morphology, connectivity, signalling, and temporal properties @Marder2011VariabilityFunction. Our results suggest this diversity, particularly in connectivity (analogous to architecture) and activity states (analogous to initialisation), serves a computational role in robust perception rather than being biological noise.

The failure of dynamics diversity alone is notable: whilst neural time constants vary widely, this matters primarily when combined with structural and state differences. This may explain why adaptation and gain control mechanisms are prevalent—they adjust temporal dynamics to maintain diversity alongside structural differences, creating independent processing streams.

The variance results align with confidence encoding theories @Pouget2013ProbabilisticComputation: population disagreement signals uncertainty, potentially triggering additional processing or behavioural caution.

== Limitations and Future Work

Several limitations warrant discussion:

*Sample size*: Reduced datasets (2,000 training samples) enabled rapid experimentation but may overestimate ensemble benefits. Larger datasets merit investigation.

*Corruption specificity*: Results focus on Gaussian noise. Other corruptions (occlusion, blur, adversarial perturbations) may show different patterns, though preliminary results suggest similar benefits for salt-and-pepper noise.

*Focal corruption level*: Synergy analysis targets σ=0.5 (maximum corruption). Synergy magnitude and presence may vary at other levels—preliminary inspection suggests additive or sub-additive interactions at some corruption values.

*Ensemble size*: N=10 balances cost and benefit. Optimal size likely depends on task difficulty and computational budget.

*Biological realism*: Implementation simplifies cortical computation (spiking dynamics, dendritic integration, neuromodulation). More realistic models merit testing.

*Learning mechanisms*: Independent training may underestimate biological diversity benefits. Competitive/cooperative learning (e.g., lateral inhibition) could enhance specialisation.

Future work should explore diversity-corruption interactions, hierarchical ensembles, online adaptation, attention mechanisms, and whether benefits persist with higher-capacity individual models.


== Practical Applications

These findings address machine learning brittleness—distributional shift and adversarial attacks. Parallel PCN ensembles offer principled robustness without expensive adversarial training. Calibrated uncertainty estimates could improve safety-critical applications (autonomous vehicles, medical diagnosis) by flagging unreliable predictions for human oversight.

The architecture diversity finding suggests that diverse model collections trained on the same task may provide natural robustness properties without explicit diversity training.

#v(1em)

= Conclusion

Horizontal parallelism in diverse predictive coding networks substantially improves robustness to sensory corruption. Ensembles of 10 diverse PCNs maintain significantly higher accuracy under Gaussian noise, with MIXED diversity providing +11.1% improvement at σ=0.5—a 28.7% relative gain.

Critically, architecture (+4.7%) and initialisation (+6.0%) diversity contribute substantially, whilst dynamics alone provides negligible benefit. The combined MIXED approach produces synergistic effects (+11.1% > +10.6% expected), demonstrating constructive interaction. Ensemble variance correlates monotonically with corruption (16.5-fold increase at σ=0.3), providing calibrated uncertainty signals.

These findings support the hypothesis that diversity in neural systems—particularly connectivity and initial states—serves a fundamental computational role in robust perception. Horizontal parallelism offers a biologically plausible mechanism for reliable inference under degraded conditions, with implications for neuroscience and machine learning alike.



= Authour Contributions
Claude Sonnet 4.5 was used to assist in creating a wrapper for existing PCN code and for restructuring and proofreading this report.




#bibliography("refs.bib", style: "harvard-cite-them-right")




