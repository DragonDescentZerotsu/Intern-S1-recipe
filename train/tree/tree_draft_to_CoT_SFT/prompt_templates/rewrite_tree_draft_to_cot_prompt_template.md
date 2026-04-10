You are rewriting a rough tree-model-derived decision-path draft into a high-quality reasoning chain for supervised fine-tuning of a chemistry or ADMET reasoning LLM.

Task:
{{TASK_NAME}}

Task semantics:
- Class A: {{CLASS_A_TEXT}}
- Class B: {{CLASS_B_TEXT}}

Your objective:
Rewrite the draft into a natural, coherent, scientist-like chain-of-thought that sounds like an LLM independently analyzing the molecule, not like a decision tree traversal.

Important rewriting goal:
The draft comes from selected paths inside random-forest trees. Different trees and different samples may use slightly different learned thresholds for the same property, which would make the final reasoning data inconsistent and overly model-specific if copied literally. One major goal of this rewriting step is therefore to normalize those raw tree thresholds into a more unified, literature-anchored reasoning style whenever that can be done safely. When the literature does not provide a stable threshold, you should still rewrite the step into a natural scientific judgment based on the property's qualitative direction or trend.

Inputs:

1. Threshold playbook from literature:
{{THRESHOLD_PLAYBOOK}}

2. Rough draft path text:
{{PATH_LEVEL_REASONING_NOTE}}

Hard requirements:
1. Do not mention trees, nodes, branches, leaves, SHAP, path length, or model internals.
2. Do not mention "next node", "branch probability", or any tree-structure wording.
3. Preserve the true qualitative direction of the original reasoning.
4. Use the threshold playbook to replace raw model thresholds only when the substitution is safe:
   - the literature threshold is scientifically more natural
   - the replacement does not flip the local conclusion toward A or B
5. Treat threshold normalization as a core objective, not a minor cosmetic change. When multiple raw tree thresholds for the same property are all pointing in the same qualitative direction, prefer one unified literature-facing interpretation instead of preserving the tree-specific decimals.
6. If exact replacement is unsafe, do not force it. Instead use one of:
   - "below the typical cutoff"
   - "above the usual threshold"
   - "within the range often associated with ..."
   - "consistent with ..."
7. If the same feature appears multiple times with different thresholds, merge them into one natural explanation whenever possible. Prefer a single unified explanation, interval-style interpretation, or high-level directional statement over repeated threshold restatements.
8. Prefer grouping by molecular property or mechanistic meaning instead of preserving raw tree order.
9. Keep the reasoning detailed, but remove redundant repeated checks that do not add scientific meaning.
10. The final text should read like a plausible medicinal chemistry or ADMET argument, not like symbolic execution.
11. Every major step should sound like genuine analysis. The model should appear to be judging what the property implies, not merely restating a threshold comparison.
12. If the evidence is mixed, acknowledge the tension and explain why the final conclusion still leans one way.
13. Never invent literature thresholds not present in the supplied playbook.
14. If a feature has no reliable literature threshold, use your chemistry or ADMET knowledge to rewrite it as a qualitative trend, tendency, or mechanistic cue rather than preserving arbitrary tree thresholds.
15. For functional-group-related steps, prefer presence/absence or coarse count reasoning instead of raw decimal thresholds.
16. In particular, if a functional-group count is being split by thresholds such as `< 0.5`, `> 0.5`, `<= 0.5`, or `0.5 < x < 1.5`, rewrite that idea in human terms:
   - `x <= 0.5` usually means the group is absent
   - `x > 0.5` usually means the group is present
   - `0.5 < x < 1.5` often means roughly one such group
   - larger integer-like count regions can be rewritten as multiple such groups or a higher count of that motif
17. For functional-group-related features, prefer wording like:
   - "the molecule contains ..."
   - "the absence of ... supports ..."
   - "having multiple ... groups tends to ..."
   rather than exposing raw numerical cutoffs.
18. If a feature is scientifically opaque, either:
   - translate it using the playbook, or
   - omit it from the final prose if it does not materially help readability.

Preferred style:
- Explicit, stepwise, chemically grounded
- Natural scientific prose
- Specific but not robotic
- More like thoughtful analysis than formal rule execution
- No bullet points in the final CoT
- No references or citations in the final CoT text itself

Required output schema:

```json
{
  "final_cot": "...",
  "used_evidence": [
    {
      "feature_name": "...",
      "how_it_was_used": "...",
      "threshold_strategy": "exact_literature | approximate_literature | model_threshold_retained | direction_only | functional_group_presence_or_count | omitted",
      "notes": "..."
    }
  ],
  "threshold_substitutions": [
    {
      "feature_name": "...",
      "model_threshold": "...",
      "literature_anchor": "...",
      "final_wording": "...",
      "safe_direction_check": "pass | fail",
      "reason": "..."
    }
  ],
  "quality_check": {
    "mentions_tree_language": false,
    "preserves_final_direction": true,
    "merged_redundant_repeated_features": true,
    "contains_unjustified_thresholds": false
  }
}
```
Before writing the final_cot, silently reason about:
- which repeated feature checks can be merged
- which thresholds can be safely rewritten
- where arbitrary tree-specific thresholds should be normalized into a more stable literature-facing explanation
- where functional-group count thresholds should be converted into presence, absence, or coarse count language
- where qualitative scientific judgment is better than preserving a raw threshold
- which details should be omitted to improve readability without losing decision content
