You are doing literature-grounded deep research to build a task-specific threshold playbook for rewriting random-forest decision paths into high-quality reasoning SFT data.

Task:
BBB_Martins

Task semantics:
- Class A: does not cross the BBB
- Class B: crosses the BBB

Goal:
Build a threshold playbook for the task above, covering as many of the scientifically meaningful non-functional-group features used in this task as possible. The playbook will later be used to rewrite rough tree-path drafts into natural, literature-aligned reasoning chains for SFT.

Important context:
- We already have model-generated decision-path drafts with model thresholds.
- Those model thresholds are not directly publishable reasoning text because they come from tree training and may vary across trees.
- Your job is to find literature-supported threshold anchors, cutoffs, ranges, or commonly used interpretive rules for the provided features, specifically in the context of this task or the closest scientifically relevant neighboring tasks.
- We care about practical thresholds that chemists, ADMET researchers, or medicinal chemistry literature actually use, not only descriptor definitions.
- Do not spend literature search budget on `fg_top_level` functional-group indicator/count features; we do not need literature thresholds for those.
- The draft shown to the rewriting LLM uses human-readable feature names rather than raw internal feature IDs, so focus on scientifically meaningful feature names only.

Feature set to cover:
- topological polar surface area
- ring count
- saturated ring count
- saturated heterocycle count
- saturated carbocycle count
- rotatable-bond count
- heteroatom count
- hydrogen-bond donor count
- hydrogen-bond acceptor count
- aromatic ring count
- aromatic heterocycle count
- aromatic carbocycle count
- aliphatic ring count
- aliphatic heterocycle count
- aliphatic carbocycle count
- nitrogen/oxygen atom count
- NH/OH group count
- molecular weight
- estimated logP
- Labute surface area
- heavy-atom molecular weight
- heavy-atom count
- fraction of sp3 carbons
- exact molecular weight
- number of ionizable sites
- number of basic sites
- number of acidic sites
- logD pH setting
- estimated logD
- neutral fraction
- sum basic site pKa
- sum acidic site pKa

Requirements:
1. Prioritize task-specific literature. If unavailable, use the closest neighboring domain and explicitly label it as a proxy.
2. For each feature, try to find the most commonly used literature threshold(s), cutoff(s), or heuristic range(s).
3. Keep the answer concise. We want a practical playbook, not a long review.
4. If the literature is conflicting, briefly note the main alternatives instead of forcing a single threshold.
5. If no reliable threshold exists, say so explicitly and give only a short qualitative note.
6. Do not invent thresholds.
7. Use primary sources or strong reviews whenever possible.

Output format:
Produce a short playbook with one section per feature, using exactly this schema:

## {feature_name}
- Common threshold(s) or range(s):
- Usually associated with:
- Brief note:
- Source:

If no reliable threshold exists, use:
- Common threshold(s) or range(s): no stable literature threshold found
- Usually associated with:
- Brief note:
- Source:
