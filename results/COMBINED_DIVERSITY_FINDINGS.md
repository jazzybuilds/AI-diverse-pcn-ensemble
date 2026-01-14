# Key Experimental Findings - Combined Diversity Test

## Research Question
Does combining multiple diversity mechanisms (e.g., dynamics + architecture) in a parallel PCN compound the robustness benefits?

## Hypothesis
If init diversity gives +7% and dynamics gives +8%, then combining them should give even more (compounding effect).

## Experimental Design
Compared 5 configurations on MNIST with Gaussian noise:
1. Single PCN (baseline)
2. Parallel with init diversity only (3 models, different random seeds)
3. Parallel with dynamics diversity only (3 models, different μ_dt)
4. Parallel with architecture diversity only (3 models, different layer configs)
5. Parallel with MIXED diversity (3 models, dynamics + architecture combined)

## Results (Accuracy at σ=0.3)

| Configuration | Accuracy | Improvement vs Single |
|--------------|----------|----------------------|
| Single PCN | 58.0% | baseline |
| Init only | 65.3% | +7.3% |
| Dynamics only | 65.9% | +7.9% |
| Architecture only | 66.1% | +8.1% ⭐ |
| **MIXED** | **63.5%** | **+5.5%** ❌ |

## Conclusion
**Hypothesis REJECTED**: Combined diversity does NOT compound benefits.

### Why This Matters
This is a **valuable negative result** that shows:

1. **More diversity ≠ always better**
   - Each diversity source may saturate the benefit
   - Too much diversity could cause interference

2. **Simplicity wins**
   - Single diversity strategies work just as well (or better)
   - Easier to implement, tune, and understand
   - Less computational overhead

3. **Scientific honesty**
   - Not all "logical" hypotheses are correct
   - Testing and reporting negative results is good science
   - Makes your work more credible

### Implications for Report
**Frame this as a strength, not a weakness:**
- "We tested whether combining diversity mechanisms would compound benefits"
- "Surprisingly, mixed diversity performed similar to or slightly worse than individual strategies"
- "This suggests diminishing returns or potential interference between diversity types"
- "Recommendation: Use a single, well-tuned diversity mechanism for simplicity and effectiveness"

### Biological Interpretation
This actually makes biological sense:
- Brain doesn't necessarily combine ALL possible sources of diversity
- Evolution optimizes for efficiency, not maximum redundancy
- Suggests there's an optimal "diversity budget"
- Too much diversity could slow consensus/decision-making

## Recommendation
**For your experiments, use:**
- **3-5 parallel models** with **one diversity type** (init, dynamics, or architecture)
- Don't over-engineer with mixed strategies
- Focus on the robust finding: parallel > single (regardless of strategy)

---

This negative result actually **strengthens your report** by showing:
✅ You tested multiple hypotheses
✅ You're honest about what doesn't work
✅ You understand scientific method
✅ Your main finding (parallel improves robustness) is robust
