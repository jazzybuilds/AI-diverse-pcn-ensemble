# Parallel Predictive Coding Networks for Robust Sensory Inference

## Quick Start (5 minutes to first results)

### 1. Run your first experiment
```bash
source venv/bin/activate
python PCN_robustness_experiment.py
```

This will:
- Train a single PCN (baseline)
- Train a parallel PCN ensemble (3 models)
- Test both under Gaussian noise corruption
- Generate plots in `results/` folder
- Save numerical results as JSON

**Expected runtime**: ~10-15 minutes on CPU, ~3-5 minutes on GPU

---

## Understanding the Output

### Files generated in `results/`:
- `robustness_gaussian_YYYYMMDD_HHMMSS.png` - Accuracy vs corruption level
- `variance_gaussian_YYYYMMDD_HHMMSS.png` - Ensemble disagreement 
- `results_gaussian_YYYYMMDD_HHMMSS.json` - Raw numerical data

### Key metrics:
- **Accuracy**: Classification accuracy at each corruption level
- **Variance**: Prediction disagreement across parallel streams (higher = more uncertainty)

---

## Experimental Variables to Sweep

Edit the config section at the bottom of `PCN_robustness_experiment.py`:

### 1. Ensemble size (`cf.n_models`)
```python
cf.n_models = 3  # Try: 1, 2, 3, 5, 7
```
**Hypothesis**: More models = better robustness but higher compute cost

### 2. Diversity type (`cf.diversity_type`)
```python
cf.diversity_type = 'init'  # Try: 'init', 'dynamics', 'architecture'
```
- `'init'`: Same architecture, different random seeds (cheap diversity)
- `'dynamics'`: Different inference step sizes (fast + slow streams)
- `'architecture'`: Different layer configurations (deep vs wide)

**Hypothesis**: 'dynamics' diversity should show best speed-accuracy tradeoff

### 3. Corruption type (`cf.corruption_type`)
```python
cf.corruption_type = 'gaussian'  # Try: 'gaussian', 'salt_pepper', 'occlude'
cf.corruption_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
```

**Hypothesis**: Parallel PCNs should be most robust to occlusion (spatial structure)

### 4. Ensemble method (`cf.ensemble_method`)
```python
cf.ensemble_method = 'average'  # Try: 'average', 'vote', 'max'
```
- `'average'`: Mean of softmax outputs (standard)
- `'vote'`: Majority vote on argmax (robust to outliers)
- `'max'`: Take maximum confidence (optimistic)

**Hypothesis**: Averaging should be most stable

### 5. Training epochs (`cf.n_epochs`)
```python
cf.n_epochs = 10  # Try: 5, 10, 20
```
Trade speed vs accuracy during development

---

## Recommended Experiment Plan (1 week)

### Day 1-2: Core results
Run 4 experiments (change one variable at a time from baseline):

**Baseline**: `n_models=1, diversity='init', corruption='gaussian'`

1. **Ensemble size effect**: n_models = 1, 3, 5
2. **Diversity type effect**: diversity = 'init', 'dynamics', 'architecture' (n_models=3)
3. **Corruption type effect**: corruption = 'gaussian', 'salt_pepper', 'occlude' (n_models=3)
4. **Ensemble method**: ensemble_method = 'average', 'vote' (n_models=3)

**Total**: ~12 runs × 15 min = 3 hours compute (can run overnight)

### Day 3-4: Analysis
- Compare accuracy curves (which setup degrades slowest?)
- Check variance curves (does parallel reduce uncertainty?)
- Calculate area-under-curve for each configuration
- Identify "best" configuration for each corruption type

### Day 5-6: Write report
Key sections:
1. **Introduction**: Biological motivation for parallel streams
2. **Methods**: Describe ParallelPCModel, diversity types, corruption
3. **Results**: Show 3-4 key plots with clear captions
4. **Discussion**: When does parallelism help? Trade-offs?

### Day 7: Polish
- Add error bars (run 3 seeds for final config)
- Clean up figures
- Proofread

---

## Expected Results (Hypotheses to test)

### Strong predictions:
1. **Robustness improvement**: Parallel PCN should maintain higher accuracy under corruption
2. **Graceful degradation**: Ensemble accuracy curve should decay slower than single model
3. **Uncertainty calibration**: Variance should increase with corruption level
4. **Diversity matters**: 'dynamics' diversity should outperform 'init' alone
5. **Diminishing returns**: 5 models should be only slightly better than 3

### Potential surprises:
- Single model might be competitive at low corruption (overhead not worth it)
- 'architecture' diversity might hurt if models vary too much
- Voting might outperform averaging for high corruption

---

## Troubleshooting

### Experiment runs slowly
- Reduce `cf.test_size` (default: 2000, can go to 1000)
- Reduce `cf.n_epochs` (10 → 5 for quick tests)
- Reduce `cf.n_train_iters` (200 → 100)

### Accuracy is low
- Check `cf.normalize = False` matches training
- Increase `cf.n_epochs` (try 20)
- Try `cf.kaiming_init = True`

### Parallel model doesn't improve
- Try `cf.diversity_type = 'dynamics'` (init alone might not be enough)
- Increase `cf.n_models` (3 → 5)
- Check corruption levels aren't too extreme

---

## Tips for Strong Report

### Great figures to include:
1. **Main result**: Accuracy vs corruption (3-4 lines: single, parallel-init, parallel-dynamics, parallel-arch)
2. **Ensemble size ablation**: Accuracy @ high corruption vs n_models (bar chart)
3. **Variance plot**: Shows parallel models "know when they don't know"
4. **Sample corrupted images**: Show what 0.0, 0.1, 0.2, 0.4 corruption looks like

### Key claims to support:
- "Parallel PCNs maintain X% higher accuracy at corruption level Y"
- "Dynamics diversity reduces variance by Z% compared to init diversity"
- "Ensemble uncertainty correlates with classification errors (r = ...)"

### Biological connections:
- Multiple cortical pathways (ventral/dorsal streams)
- Fast vs slow inference (System 1 vs System 2)
- Robustness via redundancy
- Cross-validation between streams

---

## Next Steps

1. Run baseline experiment now: `python PCN_robustness_experiment.py`
2. Check `results/` folder for plots
3. Modify config and run 2-3 more experiments
4. Start analyzing patterns

**Questions?** Check the code comments in:
- `pypc/models.py` - ParallelPCModel implementation
- `pypc/utils.py` - Corruption functions
- `PCN_robustness_experiment.py` - Experiment loop

Good luck with your research! 🧠
