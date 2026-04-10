You are rewriting a rough tree-model-derived decision-path draft into a high-quality reasoning chain for supervised fine-tuning of a chemistry or ADMET reasoning LLM.

Task:
BBB_Martins

Task semantics:
- Class A: does not cross the BBB
- Class B: crosses the BBB

Your objective:
Rewrite the draft into a natural, coherent, scientist-like chain-of-thought that sounds like an LLM independently analyzing the molecule, not like a decision tree traversal.

Important rewriting goal:
The draft comes from selected paths inside random-forest trees. Different trees and different samples may use slightly different learned thresholds for the same property, which would make the final reasoning data inconsistent and overly model-specific if copied literally. One major goal of this rewriting step is therefore to normalize those raw tree thresholds into a more unified, literature-anchored reasoning style whenever that can be done safely. When the literature does not provide a stable threshold, you should still rewrite the step into a natural scientific judgment based on the property's qualitative direction or trend.

Inputs:

1. Threshold playbook from literature:
# BBB_Martins threshold playbook for literature-grounded BBB crossing reasoning

## topological polar surface area
- Common threshold(s) or range(s): TPSA/PSA is commonly kept **< ~90 Å²** for BBB/CNS penetration, with many sources emphasizing **~60–70 Å²** as a practical target region; in CNS MPO-style desirability ranges, **40 < TPSA ≤ 90 Å²** is “desirable” and **TPSA > 120 Å²** is “undesirable.”  
- Usually associated with: **Lower TPSA → more likely Class B (BBB+)**; **higher TPSA → more likely Class A (BBB−)**.  
- Brief note: PSA/TPSA is repeatedly highlighted as a dominant driver of passive membrane transit; transporter effects (e.g., efflux) can still override “good” PSA/TPSA.  
- Source: citeturn32view0turn36view0turn24view0turn29view0turn22view0

## ring count
- Common threshold(s) or range(s): no stable literature threshold found  
- Usually associated with: context-dependent (can reduce flexibility, but can also increase size/lipophilicity depending on scaffold).  
- Brief note: CNS/BBB heuristics more often specify **flexibility (rotatable bonds)** and **polarity (PSA/TPSA, HBD/HBA)** than total ring count; ring count is discussed more as a contributor to conformational range/volume than as a standalone cutoff.  
- Source: citeturn32view0turn29view0

## saturated ring count
- Common threshold(s) or range(s): no stable literature threshold found  
- Usually associated with: weak, indirect association; higher saturation can align with improved developability and different 3D shape, but BBB impact is typically mediated through TPSA/logP/logD and ionization.  
- Brief note: Saturation (and related 3D character metrics like Fsp³) is widely used as a general medicinal chemistry heuristic rather than a BBB-specific cutoff.  
- Source: citeturn35view0turn32view0

## saturated heterocycle count
- Common threshold(s) or range(s): no stable literature threshold found  
- Usually associated with: context-dependent; can **increase HBA/TPSA** (hurting Class B) while also offering **tunable basicity/ionization** (which can help or hurt depending on pKa and neutral fraction).  
- Brief note: BBB-directed guidance tends to anchor on **net polarity and ionization** rather than the count of saturated heterocycles per se.  
- Source: citeturn32view0turn24view2

## saturated carbocycle count
- Common threshold(s) or range(s): no stable literature threshold found  
- Usually associated with: context-dependent; saturated carbocycles can reduce H-bonding liability versus heterocycles and can lower rotatable bonds (sometimes favoring Class B if size stays controlled).  
- Brief note: Use as a **shape/rigidity proxy** only; BBB literature rarely states a standalone saturated-carbocycle cutoff.  
- Source: citeturn32view0turn29view0

## rotatable-bond count
- Common threshold(s) or range(s): Common CNS-oriented guidance places rotatable bonds **~≤5** as typical for many centrally acting drugs; other commonly quoted practical filters use **<8** rotatable bonds (and broader oral-bioavailability context notes **>10** rotatable bonds as unfavorable).  
- Usually associated with: **Lower rotatable-bond count → more likely Class B** (less conformational mobility, often better permeability); **higher counts → more likely Class A**.  
- Brief note: BBB discussions frame this as “molecular flexibility”; it is widely treated as a practical screening/triage knob.  
- Source: citeturn32view0turn29view0turn18view0

## heteroatom count
- Common threshold(s) or range(s): no stable literature threshold found  
- Usually associated with: **Higher heteroatom counts (especially H-bonding heteroatoms) → more likely Class A** via increased polarity/hydrogen bonding; **lower counts → more likely Class B**.  
- Brief note: In BBB/CNS heuristics, “heteroatom burden” is most often expressed as **N+O count**, **HBA/HBD**, or **PSA/TPSA**, rather than total heteroatoms including sulfur/halogens.  
- Source: citeturn32view0turn27search4turn29view0

## hydrogen-bond donor count
- Common threshold(s) or range(s): Frequently quoted CNS guidelines include **HBD < 3**; CNS MPO desirability uses an even tighter “desirable” anchor at **HBD ≤ 0.5** with **HBD > 3.5** undesirable (reflecting a strong penalty for multiple donors).  
- Usually associated with: **Lower HBD → more likely Class B**; **higher HBD → more likely Class A**.  
- Brief note: Donors are repeatedly treated as high-impact because they raise desolvation cost and correlate with both reduced passive permeability and higher efflux interaction risk in many workflows.  
- Source: citeturn29view0turn24view0turn32view0

## hydrogen-bond acceptor count
- Common threshold(s) or range(s): Frequently quoted CNS guidelines include **HBA < 7**, often paired with a **total H-bonding count < 8** heuristic (donors + acceptors).  
- Usually associated with: **Lower HBA → more likely Class B**; **higher HBA → more likely Class A**.  
- Brief note: Acceptors correlate with polarity and PSA/TPSA; many BBB rules use acceptors directly or indirectly via N+O and PSA/TPSA.  
- Source: citeturn29view0turn32view0turn22view0

## aromatic ring count
- Common threshold(s) or range(s): In the BBB Score framework, aromatic ring count is explicitly step-scored with **strong penalties beyond 4** (i.e., **>4 aromatic rings scored as 0 contribution**), and the highest desirability occurs around **2 aromatic rings** (still favorable around **1–3** depending on the scoring function).  
- Usually associated with: **Very high aromatic ring counts (≥4–5) → more likely Class A** in rule-based BBB scoring; moderate aromatic ring counts can be compatible with Class B when PSA/TPSA and H-bonding stay controlled.  
- Brief note: Aromatic ring count is used as a practical proxy for “aromaticity burden” and is directly embedded in BBB screening scores (rather than being a universal standalone BBB cutoff).  
- Source: citeturn22view0turn21search1

## aromatic heterocycle count
- Common threshold(s) or range(s): no stable literature threshold found  
- Usually associated with: often trends toward **Class A** as aromatic heterocycles commonly add HBA/TPSA; can still be **Class B-compatible** if overall HBA/HBD and TPSA remain in CNS ranges.  
- Brief note: Literature and scoring rules usually threshold **HBA/HBD/TPSA/pKa** rather than splitting aromatic rings into heteroaromatic vs carbocyclic subcounts.  
- Source: citeturn32view0turn29view0turn24view0

## aromatic carbocycle count
- Common threshold(s) or range(s): no stable literature threshold found  
- Usually associated with: context-dependent; can support lipophilicity (helping passive diffusion) but too many aromatic carbocycles can push “aromaticity burden” into unfavorable developability space and may not rescue high TPSA/H-bonding.  
- Brief note: Consider this subcount mainly as a decomposition of “aromatic rings” used by some descriptor sets; BBB rules more commonly reference total aromatic rings.  
- Source: citeturn22view0turn32view0

## aliphatic ring count
- Common threshold(s) or range(s): no stable literature threshold found  
- Usually associated with: context-dependent; added rings can reduce rotatable bonds (sometimes favoring Class B if MW/TPSA remain low).  
- Brief note: BBB/CNS literature discusses **rigidity/flexibility** primarily via rotatable bonds; aliphatic ring subcounts are rarely given hard cutoffs.  
- Source: citeturn32view0turn29view0

## aliphatic heterocycle count
- Common threshold(s) or range(s): no stable literature threshold found  
- Usually associated with: context-dependent; can raise basic-site count and tune pKa (sometimes helpful), but may also raise HBA/TPSA (often harmful).  
- Brief note: For BBB, ionization state at physiological pH is emphasized more than “aliphatic heterocycle count” itself.  
- Source: citeturn32view0turn24view2turn22view0

## aliphatic carbocycle count
- Common threshold(s) or range(s): no stable literature threshold found  
- Usually associated with: weak and indirect; may support Class B via reduced H-bonding while controlling rotatable bonds, but can also increase size/lipophilicity.  
- Brief note: Treat as a structural “shape/rigidity” proxy; no consensus BBB cutoff.  
- Source: citeturn32view0turn29view0

## nitrogen/oxygen atom count
- Common threshold(s) or range(s): A widely cited rule set uses **(N + O) ≤ 5** as indicating a high chance of brain entry; a paired rule states **if logP > (N + O)** then **logBB is positive** (i.e., higher brain than blood concentration).  
- Usually associated with: **Lower N+O → more likely Class B**; higher N+O generally pushes toward Class A via increased polarity/H-bonding capacity.  
- Brief note: This is a convenient “fast rule” style anchor; it compresses polarity into a single integer count and is often treated as a coarse screening heuristic.  
- Source: citeturn27search4turn32view0turn24view2

## NH/OH group count
- Common threshold(s) or range(s): Often operationalized via hydrogen-bonding rules: **HBD < 3** is a frequently quoted CNS threshold; some BBB-permeable profiles emphasize **very few polar hydrogens** (e.g., “<3, typically 0–1” in one guideline set).  
- Usually associated with: **Lower NH/OH (polar H) counts → more likely Class B**; higher counts → more likely Class A.  
- Brief note: NH/OH groups are a direct handle on donor burden and often track with both TPSA and desolvation penalties.  
- Source: citeturn29view0turn24view2turn32view0

## molecular weight
- Common threshold(s) or range(s): Classical BBB filters often use **MW < 450**; additional commonly cited anchors include **~400 as a cutoff** in some rulesets, and CNS MPO-style desirability marks **MW ≤ 360** as desirable and **MW > 500** as undesirable.  
- Usually associated with: **Lower MW → more likely Class B**; higher MW (especially beyond ~450–500) → more likely Class A.  
- Brief note: MW is treated as a size/transport proxy; exceptions exist (influx transporters, prodrugs, high lipophilicity), but MW remains a standard screening anchor.  
- Source: citeturn36view0turn32view0turn24view0turn29view0

## estimated logP
- Common threshold(s) or range(s): Reported optimum BBB penetration for multiple CNS-active classes has been cited around **logP ~1.5–2.7** (mean ~2.1); other CNS library rules use broader windows such as **~2–5**; CNS MPO desirability treats **ClogP ≤ 3** as desirable with **ClogP > 5** undesirable.  
- Usually associated with: **Moderate logP** (not too low, not too high) is most often associated with Class B; very low logP tends to Class A via poor permeability, while very high logP can increase liabilities (even if permeability rises).  
- Brief note: logP is repeatedly framed as entangled with size/surface area and H-bonding; interpret alongside TPSA/HBD/HBA and ionization.  
- Source: citeturn32view0turn24view0turn24view2turn29view0

## Labute surface area
- Common threshold(s) or range(s): no stable literature threshold found  
- Usually associated with: smaller overall accessible surface area generally trends toward Class B (as a size proxy), but the effect is indirect.  
- Brief note: A commonly cited BBB-permeable guideline set includes **solvent-accessible surface area ~460–580 Å²** (with additional constraints like TPSA and polar hydrogens); this is a **proxy** anchor for “surface area”-type descriptors and should not be treated as a direct Labute-ASA equivalence.  
- Source: citeturn24view2turn32view0

## heavy-atom molecular weight
- Common threshold(s) or range(s): no stable literature threshold found  
- Usually associated with: behaves as a **size proxy** similar to MW; larger values tend toward Class A when they reflect larger molecules (especially when TPSA/H-bonding also rise).  
- Brief note: BBB/CNS heuristics and scores almost always specify **MW** directly; “heavy-atom MW” is typically an internal descriptor choice rather than a literature-anchored cutoff.  
- Source: citeturn32view0turn24view0turn36view0

## heavy-atom count
- Common threshold(s) or range(s): In the BBB Score framework, heavy-atom count is scored with an explicit **0 contribution below 6 or above 45 heavy atoms**, and nonzero scoring in the **6–45** range (polynomial weighting).  
- Usually associated with: Extremely low or extremely high heavy-atom counts are treated as unfavorable for Class B in BBB Score-style screening; mid-range heavy-atom counts are more compatible with Class B if TPSA/ionization are aligned.  
- Brief note: This is best used as an **algorithmic anchor** (BBB Score) rather than a universal BBB cutoff; in practice it largely tracks molecular size.  
- Source: citeturn22view0turn21search1

## fraction of sp3 carbons
- Common threshold(s) or range(s): no stable literature threshold found  
- Usually associated with: not BBB-specific; higher saturation (higher Fsp³) is often used as a developability/solubility heuristic and can indirectly help by reducing excessive aromaticity.  
- Brief note: One large-scale analysis trend reported across drug discovery phases is an increase in mean Fsp³ from **~0.36 (research compounds)** to **~0.47 (drugs)**; this is not a BBB cutoff, but provides a practical “typical range” anchor for rewriting feature-based rationales.  
- Source: citeturn35view0

## exact molecular weight
- Common threshold(s) or range(s): Same practical anchors as MW are typically used: **<450** is a common BBB filter; CNS MPO “desirable” anchor **≤360** and “undesirable” **>500**.  
- Usually associated with: **Lower exact MW → more likely Class B**; higher exact MW → more likely Class A.  
- Brief note: Exact MW vs average/isotopic MW is rarely distinguished in BBB heuristic rules; the screening logic is effectively “size constraint.”  
- Source: citeturn36view0turn24view0turn32view0

## number of ionizable sites
- Common threshold(s) or range(s): no stable literature threshold found  
- Usually associated with: **More ionizable sites → more likely Class A** (lower neutral fraction at pH ~7.4, higher polarity); fewer ionizable sites can support Class B when PSA/logD are aligned.  
- Brief note: BBB discussions emphasize that passive membrane permeation is driven by the **neutral species fraction** in aqueous phase; strong acids/bases are often described as poor BBB penetrants, and a commonly cited pKa window for BBB penetration is **~4 to 10** (reflecting “weak” acids/bases).  
- Source: citeturn32view0turn24view2turn22view0

## number of basic sites
- Common threshold(s) or range(s): no stable literature threshold found  
- Usually associated with: Presence of a **weakly basic center** is frequently compatible with Class B; multiple basic sites can increase polarity and reduce neutral fraction, pushing toward Class A unless compensated.  
- Brief note: Guidance is typically expressed in terms of **basic pKa limits** (e.g., CNS MPO desirability uses **pKa ≤ 8** as desirable and **>10** as undesirable; another analysis reports no CNS drugs with **basic pKa > 10.5**), rather than “count of basic sites.”  
- Source: citeturn24view0turn24view2turn32view0

## number of acidic sites
- Common threshold(s) or range(s): no stable literature threshold found  
- Usually associated with: **Acidic groups/sites (especially strong acids) → more likely Class A** because acids are ionized at physiological pH and have low neutral fraction; neutral/weakly basic scaffolds are more compatible with Class B.  
- Brief note: CNS-focused reviews often highlight the general difficulty of carboxylic acids in BBB penetration and emphasize the criticality of the neutral species fraction; a commonly cited pKa window for BBB penetration is **~4–10** (weak acids/bases).  
- Source: citeturn32view0turn24view2

## logD pH setting
- Common threshold(s) or range(s): **pH 7.4** is the default/standard setting for logD in BBB/CNS MPO-type rules and tables (physiologic buffer lipophilicity).  
- Usually associated with: logD measured/estimated at **pH 7.4** is the conventional anchor for comparing Class A vs Class B across ionizable compounds.  
- Brief note: Because ionization changes with pH, logD (pH-specified) is preferred over logP for comparing ionizable molecules in many BBB/CNS screening contexts.  
- Source: citeturn24view0turn12view1turn32view0

## estimated logD
- Common threshold(s) or range(s): One commonly cited BBB/CNS anchor is **0 < logD < 3** for better brain permeation (and intestinal permeability in neighboring contexts); CNS MPO desirability uses **ClogD7.4 ≤ 2** as desirable and **>4** as undesirable; other CNS library rules have used broader windows such as **~2–5**.  
- Usually associated with: **Moderate logD7.4** tends to favor Class B; very low logD suggests poor permeability (Class A), while very high logD can raise nonspecific binding and other liabilities even if permeability improves.  
- Brief note: Use logD7.4 as the “ionization-aware lipophilicity” anchor; interpret together with TPSA and the neutral fraction.  
- Source: citeturn32view0turn24view0turn24view2

## neutral fraction
- Common threshold(s) or range(s): no stable literature threshold found  
- Usually associated with: **Higher neutral fraction at physiologic pH → more likely Class B** (supports passive diffusion); low neutral fraction → more likely Class A.  
- Brief note: BBB-oriented reviews emphasize that the neutral species in the aqueous phase is critical for membrane penetration; thus pKa/logD7.4 are often used as practical surrogates rather than a single universal “neutral fraction cutoff.”  
- Source: citeturn32view0turn24view2turn24view0

## sum basic site pKa
- Common threshold(s) or range(s): no stable literature threshold found  
- Usually associated with: A “too-basic” profile (high basic pKa values) tends toward Class A due to high ionization at pH ~7.4; moderate basicity is more compatible with Class B.  
- Brief note: BBB/CNS rules are typically expressed using **maximum/most-basic pKa** rather than a sum; commonly used anchors include **pKa ≤ 8 (desirable)** and **pKa > 10 (undesirable)** in CNS MPO-type frameworks, and an additional report that no CNS drugs had **basic pKa > 10.5** in one comparative analysis.  
- Source: citeturn24view0turn24view2turn32view0turn22view0

## sum acidic site pKa
- Common threshold(s) or range(s): no stable literature threshold found  
- Usually associated with: More/stronger acidity tends toward Class A (greater ionization at pH ~7.4); weak-acid behavior can be compatible with Class B if neutral fraction is nontrivial.  
- Brief note: BBB/CNS guidance usually relies on per-site or limiting pKa values (not sums); one comparative analysis reports CNS drugs rarely having **acidic pKa below ~6**, and another commonly cited BBB penetration pKa window is **~4–10** (weak acids/bases).  
- Source: citeturn24view2turn32view0turn22view0

2. Rough draft path text:
This tree's reasoning repeatedly relies on important features such as estimated logD, has an acidic site, nitrogen/oxygen atom count. First, estimated logD has raw value 4.634, which is > the raw threshold 0.37165; this steers the path toward option (B): crosses the BBB. This feature is also in the SHAP top set (abs SHAP 0.011571). Next, has an acidic site has raw value 0, which is <= the raw threshold 0.5; this steers the path toward option (B): crosses the BBB. This feature is also in the SHAP top set (abs SHAP 0.014273). Then, carboxylic ester has raw value 1, which is <= the raw threshold 1.5; this steers the path toward option (B): crosses the BBB. After that, aliphatic heterocycle count has raw value 0, which is <= the raw threshold 0.5; this steers the path toward option (B): crosses the BBB. Finally, nitrogen/oxygen atom count has raw value 3, which is > the raw threshold 1.5; this steers the path toward option (B): crosses the BBB. This feature is also in the SHAP top set (abs SHAP 0.016624). Step 6, methoxy fragment count has raw value 0, which is <= the raw threshold 2.5; this steers the path toward option (B): crosses the BBB. Step 7, dialkyl thioether has raw value 0, which is <= the raw threshold 0.5; this steers the path toward option (B): crosses the BBB. Step 8, topological polar surface area has raw value 29.54, which is <= the raw threshold 42.885; this steers the path toward option (B): crosses the BBB. This feature is also in the SHAP top set (abs SHAP 0.033952). Step 9, fraction of sp3 carbons has raw value 0.611111, which is > the raw threshold 0.056061; this steers the path toward option (B): crosses the BBB. This feature is also in the SHAP top set (abs SHAP 0.00876). Step 10, thiazole has raw value 0, which is <= the raw threshold 0.5; this steers the path toward option (B): crosses the BBB. Step 11, sum basic site pKa has raw value 4.7646, which is <= the raw threshold 15.1903; this steers the path toward option (B): crosses the BBB. Step 12, minimum absolute partial charge has raw value 0.305813, which is > the raw threshold 0.115509; this steers the path toward option (B): crosses the BBB. This feature is also in the SHAP top set (abs SHAP 0.013447). Step 13, aliphatic carbocycle count has raw value 0, which is <= the raw threshold 3; this steers the path toward option (B): crosses the BBB. Step 14, aliphatic carbocycle count has raw value 0, which is <= the raw threshold 0.5; this steers the path toward option (B): crosses the BBB. Step 15, neutral fraction has raw value 0.9977, which is > the raw threshold 0.02145; this steers the path toward option (B): crosses the BBB. This feature is also in the SHAP top set (abs SHAP 0.014674). Step 16, hydrogen-bond acceptor count has raw value 3, which is <= the raw threshold 3.5; this steers the path toward option (B): crosses the BBB. This feature is also in the SHAP top set (abs SHAP 0.010552). Step 17, number of ionizable sites has raw value 1, which is <= the raw threshold 1.5; this steers the path toward option (B): crosses the BBB. This feature is also in the SHAP top set (abs SHAP 0.008513). Step 18, Labute surface area has raw value 149.45, which is > the raw threshold 83.1391; this steers the path toward option (B): crosses the BBB. Step 19, minimum absolute partial charge has raw value 0.305813, which is > the raw threshold 0.119979; this steers the path toward option (B): crosses the BBB. This feature is also in the SHAP top set (abs SHAP 0.013447). Step 20, estimated logP has raw value 4.635, which is > the raw threshold 3.1452; this steers the path toward option (B): crosses the BBB. This feature is also in the SHAP top set (abs SHAP 0.008458). Under this sequence of conditions, the tree supports option (B): crosses the BBB.

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
9. Keep the reasoning detailed and make sure to mention each feature's exact value in the the final CoT, don't omit any.
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
18. Do not omit any feature used in the original draft.

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
    "mentions_tree_language": true or false,
    "preserves_final_direction": true or false,
    "merged_redundant_repeated_features": true or false,
    "contains_unjustified_thresholds": true or false
  }
}
```
Before writing the final_cot, reason about:
- which repeated feature checks can be merged
- which thresholds can be safely rewritten
- where arbitrary tree-specific thresholds should be normalized into a more stable literature-facing explanation
- where functional-group count thresholds should be converted into presence, absence, or coarse count language
- where qualitative scientific judgment is better than preserving a raw threshold
- which details should be omitted to improve readability without losing decision content
