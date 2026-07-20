# TextComp Revision Plan for Resubmission

## 0. Goal

This document summarizes two previous review rounds and converts reviewer concerns into concrete revision actions for the current manuscript:

Main paper:
`/home/neimedia/gmk/BIKE/paper/AAAI/Anonymous_Textcomp2027.tex`

Related files:
- `rebuttal.tex`
- `Rebuttal_ECCV.tex`
- `Supplementary_Textcomp.tex`
- `Supplementary.tex`
- `aaai2027.bib`
- figures under `figure/`

The goal is not to write another rebuttal, but to revise the paper so that previously raised concerns are proactively addressed in the manuscript.

---

# 1. Global Diagnosis

Across two review rounds, reviewers generally agreed that:

## Strengths
- The problem of compressed-domain action recognition is meaningful and practical.
- Introducing language guidance into compressed-domain modeling is intuitive and potentially useful.
- The framework is coherent and reasonably well designed.
- The paper reports broad experiments, including fully-supervised, zero-shot, few-shot, ablation, robustness, and efficiency results.
- The efficiency advantage over raw-video methods is practically relevant.

## Main Rejection Reasons
Two recurring concerns caused rejection:

1. **Limited novelty / incremental combination**
   Reviewers repeatedly felt that TextComp combines existing ingredients:
   - CLIP-style supervision
   - LLM-generated action attributes
   - cross-modal attention or adaptive fusion

   The current manuscript should therefore reposition the contribution around the unique problem of compressed-domain semantic sparsity and semantic navigation, rather than presenting the method as merely adding LLM attributes to CLIP.

2. **Weak SSv2 performance**
   Reviewers were not convinced by the argument that weak SSv2 performance is due to coarse motion vectors. They argued that handling noisy/coarse codec signals is exactly where the proposed text-guided routing mechanism should help. Therefore, claims about learning discriminative motion patterns must be toned down or better supported.

---

# 2. Review Concern Taxonomy

## C1. Limited Novelty / Existing Component Combination

### Source
- ECCV Reviewer jnRs
- ECCV Reviewer om4n
- ECCV Reviewer Bg4a
- ECCV Area Chair
- CVPR Reviewer f7AA
- CVPR Area Chair

### Representative Comments
- The method appears to combine standard components: CLIP/VLM supervision, LLM-generated attributes, and dynamic cross-modal fusion.
- Applying this combination to compressed-domain action recognition is interesting but algorithmically incremental.
- The paper should position the contribution more precisely.

### Current Status
Partially addressed in rebuttal, but not sufficiently solved in the manuscript.

### Required Revision
- Rewrite Abstract and Introduction to define a sharper problem:
  **semantic sparsity in compressed-domain video recognition**.
- Present TextComp as a semantic navigation framework for codec-domain signals, not as a generic CLIP+LLM+fusion combination.
- Add a clear paragraph explaining why compressed-domain recognition needs semantic guidance more than RGB-domain recognition.
- Add explicit comparison with:
  - RGB VLM action recognition methods
  - compressed-domain visual-only fusion methods
  - compressed prompt tuning methods such as CVPT
  - Efficient motion-centric CLIP for compressed video action recognition, if relevant and citable

### Suggested Manuscript Locations
- Abstract
- Introduction
- Related Work
- Method Overview
- Conclusion

---

## C2. Weak SSv2 Results and Motion-Centric Claim

### Source
- ECCV Reviewer jnRs
- ECCV Reviewer om4n
- ECCV Area Chair
- CVPR Reviewer f7AA

### Representative Comments
- SSv2 is the most important dataset for testing motion-centric reasoning.
- TextComp underperforms strong compressed-domain methods on SSv2.
- The explanation that MV resolution is coarse is not convincing because noisy/coarse signals are exactly the challenge the method claims to address.
- Weak SSv2 results undermine the claim that the method learns discriminative motion patterns.

### Current Status
Not fully resolved.

### Required Revision
- Tone down all claims suggesting strong fine-grained motion reasoning.
- Avoid saying the method "consistently outperforms prior SOTA" across all datasets.
- Add a dedicated SSv2 discussion:
  - SSv2 requires fine-grained object-motion interaction.
  - codec-domain MVs are block-level and coarse.
  - text attributes help routing but cannot recover missing fine-grained temporal information.
- Emphasize that TextComp improves the accuracy-efficiency trade-off in compressed-domain recognition, rather than fully solving fine-grained temporal reasoning.
- Add or formalize SSv2-specific evidence:
  - IADF improves SSv2 from 57.7 to 60.4.
  - IADF increases MV/Res weights on SSv2.
  - include Fig. `fig1.pdf` or a better version as modality-weight visualization.
- Add limitations on motion-centric benchmarks.

### Suggested Manuscript Locations
- Experimental Results
- Ablation Study
- Limitations
- Conclusion

---

## C3. Static Attribute Guidance vs. Instance-Level Interaction

### Source
- CVPR Reviewer f7AA
- CVPR Area Chair

### Representative Comments
- Attributes are generated only from action labels, so all videos in the same class share identical textual guidance.
- This appears to be label-side augmentation rather than true video-text interaction.
- The rebuttal did not provide sufficient instance-level heatmaps or variance metrics proving dynamic adaptation to content/background noise.

### Current Status
Partially addressed through IADF design and some weight analysis, but still a key risk.

### Required Revision
- Explicitly acknowledge that LLM attributes are class-level semantic priors.
- Clarify that instance-awareness comes from visual-feature queries in cross-modal attention and frame-level dynamic routing, not from regenerating text per video.
- Add instance-level evidence:
  - per-video IADF weight variance
  - class-level and instance-level modality weight examples
  - attention/activation visualizations
  - MV noise injection adaptation
- Avoid overstating "instance-specific textual guidance"; use "class-level semantic prior with instance-conditioned visual querying".

### Suggested Manuscript Locations
- Method: IADF
- Ablation Study
- Qualitative Analysis
- Limitations

---

## C4. Method Complexity vs. Modest Gains

### Source
- ECCV Reviewer om4n
- ECCV Reviewer jnRs

### Representative Comments
- Gains from prompt design and IADF are moderate.
- It is unclear whether the full complexity of the method is necessary.

### Current Status
Partially addressed.

### Required Revision
- Explain the value of IADF beyond top-1 accuracy:
  - consistent gains
  - robustness to corrupted MV
  - adaptive modality reliability
  - few-shot/zero-shot improvements
  - interpretability
- Add a design rationale paragraph:
  1. compressed signals are semantically sparse,
  2. class names are insufficient,
  3. modality reliability varies by instance/frame,
  4. therefore attributes + IADF are needed.
- Make the method description more compact and purposeful.

### Suggested Manuscript Locations
- Method Overview
- Ablation Study
- Robustness Analysis

---

## C5. Fairness of Comparison and Overclaimed SOTA

### Source
- ECCV Reviewer om4n
- ECCV Reviewer Bg4a
- CVPR Reviewer i1dZ

### Representative Comments
- Raw-video and compressed-domain methods are not fully comparable.
- Some methods use different backbones and computational budgets.
- Claims such as "consistently outperforms prior state-of-the-art" are too strong.
- CVPT ViT-L and MM-ViT outperform TextComp on SSv2 or some datasets.

### Current Status
Partially addressed in rebuttal.

### Required Revision
- Replace strong SOTA claims with qualified wording:
  - "competitive compressed-domain performance"
  - "favorable accuracy-efficiency trade-off"
  - "strong performance under comparable computational cost"
- Clearly separate tables or discussion into:
  1. raw-video methods as reference only,
  2. compressed-domain methods,
  3. backbone-matched comparisons,
  4. efficiency comparisons.
- Include rebuttal comparisons where appropriate:
  - HSINet-T / HSINet-B
  - CompViT
  - TextComp-L vs CVPT ViT-L
- State that raw-video VLM methods use richer decoded RGB inputs and are not apples-to-apples.

### Suggested Manuscript Locations
- Experimental Results
- Table captions
- Main comparison discussion
- Abstract and Conclusion

---

## C6. LLM Attribute Robustness and Quality

### Source
- ECCV Reviewer om4n
- ECCV Reviewer Bg4a
- CVPR Reviewer uVHP

### Representative Comments
- Attribute generation is central but under-analyzed.
- Need prompt sensitivity, generation variance, manual attributes, and attribute quantity analysis.
- Need failure cases where attributes are generic or misleading.

### Current Status
Mostly addressed in rebuttal and supplementary, but should be incorporated.

### Existing Evidence
- Regenerating attributes 5 times: 75.8 ± 0.3%.
- Prompt template sensitivity across 5 variants: 76.0 ± 0.4%.
- Manual vs LLM vs LLM+manual selection on 10 HMDB-51 classes:
  - baseline 73.2%
  - manual 74.1%
  - LLM 75.6%
  - LLM+manual selection 77.8%
- Attribute quantity:
  - L=16 lower
  - L=32 best
  - L=64 slightly worse
- Qwen vs Llama comparison.

### Required Revision
- Add a compact subsection on attribute generation robustness.
- Move details to supplementary if space is limited.
- Add failure cases:
  - ambiguous classes such as wave/stand
  - similar action pairs such as pick/put
- Include sample attribute words in supplementary.

### Suggested Manuscript Locations
- Attribute Generation
- Ablation Study
- Supplementary

---

## C7. Dataset Name in Prompt and In-the-Wild Generalization

### Source
- CVPR Reviewer i1dZ
- CVPR Area Chair

### Representative Comments
- The prompt includes the dataset name.
- This may limit in-the-wild applicability.
- What happens without dataset-specific context?

### Current Status
Partially addressed in supplementary through semantic similarity analysis.

### Existing Evidence
From supplementary:
- Base off-diagonal similarity: 0.8418
- w/o Dataset Name: 0.8018
- w/ Dataset Name: 0.7593
This shows attributes improve separability even without dataset name.

### Required Revision
- In the main paper, explicitly state:
  - dataset name is used only for benchmark-specific semantic disambiguation;
  - it is not required at inference;
  - in open-world use, the prompt can omit dataset name.
- Add the w/o dataset name ablation to the main paper or appendix.
- Avoid framing dataset name as essential.

### Suggested Manuscript Locations
- Attribute Generation
- Ablation Study
- Supplementary

---

## C8. Broader Applicability Beyond Classification

### Source
- CVPR Reviewer i1dZ

### Representative Comments
- The approach may be specific to action classification.
- Can similar representations be used for retrieval or other video-language tasks?

### Current Status
Addressed in supplementary with zero-shot video-text retrieval, but should be clearly included.

### Existing Evidence
HMDB-51 zero-shot video-text retrieval:
- Text-to-Video R@1: 49.02 → 50.98
- Text-to-Video R@5: 66.67 → 72.55
- Text-to-Video R@10: 78.43 → 82.35
- MdR: 2.0 → 1.0
- Video-to-Text slightly improves.

### Required Revision
- Add a "Broader Applicability" subsection in supplementary, possibly briefly mention in main paper.
- Frame attribute enhancement as query expansion for video-text retrieval.
- Avoid overclaiming; present as preliminary evidence.

### Suggested Manuscript Locations
- Supplementary
- Optional short paragraph in Experiments

---

## C9. Qualitative Analysis and Failure Cases

### Source
- CVPR Area Chair
- CVPR Reviewer uVHP
- CVPR Reviewer f7AA
- ECCV Reviewer Bg4a

### Representative Comments
- Need qualitative examples.
- Need visualizations showing how attributes modulate modalities.
- Need failure cases.

### Current Status
Partially addressed, but should be strengthened.

### Required Revision
- Add qualitative visualization:
  - attention maps
  - IADF modality weights
  - examples for motion-centric and appearance-centric actions
- Add failure case discussion:
  1. ambiguous classes with generic attributes
  2. similar action pairs with overlapping attributes
  3. SSv2 pick/put/open/close confusion
- Add instance-level examples showing different weights for videos in the same class if possible.

### Suggested Manuscript Locations
- Qualitative Analysis
- Limitations
- Supplementary

---

## C10. Codec Generalization

### Source
- CVPR Reviewer uVHP
- CVPR Reviewer i1dZ

### Representative Comments
- Experiments use MPEG-4.
- Modern codecs such as H.265/HEVC and AV1 have different I/P-frame structures and motion vectors.
- Codec generalization is not analyzed.

### Current Status
Unresolved.

### Required Revision
Option A, if experiments are available:
- Add small-scale H.265/HEVC evaluation.

Option B, if experiments are not available:
- Add explicit limitation:
  "Our experiments follow MPEG-4 settings used in prior compressed-domain recognition work. Extending TextComp to HEVC/AV1 is an important future direction because codec structures and motion representations differ."

### Suggested Manuscript Locations
- Implementation Details
- Limitations

---

## C11. Heuristic Semantic-Modality Decomposition

### Source
- CVPR Reviewer f7AA

### Representative Comments
- Mapping appearance/motion/detail to IF/MV/Res is heuristic.
- Codec signals are coupled; residuals also contain motion compensation information.
- A clean semantic-modality mapping lacks theoretical justification.

### Current Status
May be addressed if current method uses implicit semantic routing instead of rigid decomposition.

### Required Revision
- Remove or soften any rigid claim that appearance maps to IF, motion maps to MV, and details map to Res.
- Emphasize implicit semantic routing:
  - all attributes are available to all modalities;
  - each modality queries the relevant textual cues through cross-modal attention;
  - no fixed semantic subspace is enforced.
- If the old Eq. 1-2 still imposes static decomposition, revise it.

### Suggested Manuscript Locations
- Attribute Generation
- IADF
- Method Overview

---

## C12. CLIP Initialization vs. Text Branch Contribution

### Source
- ECCV Reviewer Bg4a
- CVPR Reviewer i1dZ

### Representative Comments
- Improvements may come from CLIP initialization rather than the text branch.
- Multiple things change at once in comparisons.

### Current Status
Addressed in rebuttal but should be formalized.

### Existing Evidence
- Compressed modalities + classification head only:
  - K400-PT: 62.9%
  - CLIP-PT: 60.9%
- Full model: 76.2%
- Class name + IADF: 73.3/95.1
- Attribute-enhanced model: 76.2/96.7

### Required Revision
- Add a text-branch isolation ablation.
- Clarify pretrained backbone use in implementation.
- Use fair comparisons in tables.

### Suggested Manuscript Locations
- Ablation Study
- Implementation Details

---

# 3. High-Priority Revision Checklist

## Must Fix Before Resubmission

1. Rewrite the Abstract to avoid overclaiming and sharpen the problem definition.
2. Rewrite the Introduction around "semantic sparsity in compressed-domain recognition".
3. Add clear differences from RGB VLM methods, CVPT, MM-ViT, and compressed-domain visual-only methods.
4. Tone down all SOTA and motion reasoning claims.
5. Add a dedicated SSv2 discussion and limitation.
6. Clarify that attributes are class-level priors, while instance-awareness comes from visual-query-based routing.
7. Add evidence for dynamic routing:
   - MV noise injection,
   - class-level modality weights,
   - SSv2 modality weights,
   - qualitative visualization.
8. Add prompt robustness and w/o dataset name analysis.
9. Add failure cases.
10. Add limitation about MPEG-4 codec dependency if no HEVC/AV1 experiment exists.

---

# 4. Suggested New Paper Framing

The paper should be framed as follows:

"Compressed-domain video recognition suffers from semantic sparsity: I-frames provide sparse appearance snapshots, while motion vectors and residuals are low-level, noisy, and heterogeneous. Existing methods improve visual architectures within the compressed domain but lack high-level semantic navigation. TextComp introduces class-level LLM-generated action attributes as semantic anchors and uses instance-conditioned visual queries to dynamically inject and route these semantics across IF, MV, and Res streams. The method improves the accuracy-efficiency trade-off and provides strong few-shot/zero-shot generalization, while still being limited on fine-grained motion-centric datasets such as SSv2."

Avoid framing it as:
"TextComp solves motion-centric compressed video recognition" or "TextComp consistently achieves SOTA across all benchmarks."

---

# 5. Recommended Wording Changes

## Replace Strong Claims

### Bad
"TextComp consistently outperforms prior state-of-the-art approaches."

### Better
"TextComp achieves competitive compressed-domain performance with a favorable accuracy-efficiency trade-off."

---

### Bad
"Our method learns discriminative motion patterns even from heavily degraded signals."

### Better
"Our method adaptively reweights compressed modalities and improves robustness under noisy motion vectors, although fine-grained motion-centric reasoning remains challenging."

---

### Bad
"Text attributes provide instance-aware guidance."

### Better
"LLM-generated attributes provide class-level semantic priors, while instance-awareness is achieved through visual-query-based cross-modal attention and frame-level dynamic routing."

---

# 6. Suggested New Sections or Paragraphs

## Add to Introduction
- Semantic sparsity as the central problem.
- Why compressed-domain recognition needs semantic navigation.
- Why this is different from RGB VLMs.
- Acknowledge that TextComp targets accuracy-efficiency trade-off rather than fully replacing raw RGB methods.

## Add to Related Work
- Difference from RGB-domain VLM action recognition.
- Difference from compressed prompt tuning.
- Difference from visual-only compressed-domain fusion.
- Include or discuss "Efficient motion-centric CLIP for compressed video action recognition" if appropriate.

## Add to Method Overview
- Design rationale:
  1. compressed signals are sparse/noisy,
  2. class names are too coarse,
  3. modality reliability varies across frames and instances,
  4. semantic attributes + IADF address these issues.
- Clarify class-level attributes vs instance-level routing.

## Add to Experiments
- Backbone-matched comparisons.
- SSv2-specific analysis.
- Text branch isolation.
- Prompt robustness.
- Semantic specificity: shuffled attributes and dummy prompt.
- Modality weight analysis.
- Qualitative visualization.
- Failure cases.

## Add to Limitations
- SSv2/fine-grained motion reasoning remains challenging.
- Attributes are class-level priors.
- Experiments mainly use MPEG-4.
- LLM attribute quality can affect performance.
- Current framework is primarily evaluated for action recognition, with preliminary retrieval evidence.

---

# 7. Existing Rebuttal/Supplementary Results to Integrate

## From rebuttal.tex
- I-frame-only baseline.
- Shuffled attributes and dummy prompt.
- SSv2 IADF weight analysis.
- Text branch vs CLIP initialization.
- Attribute robustness:
  - seed variance,
  - prompt template sensitivity,
  - manual vs LLM attributes.
- Motion vs appearance class-level modality weights.
- Failure cases.

## From Supplementary.tex
- Attribute word generation details.
- Sample generated attributes.
- Semantic discriminability and prompt robustness.
- w/o dataset name vs w/ dataset name.
- Zero-shot video-text retrieval.

---

# 8. Final Risk Assessment

Even after revision, the highest-risk concerns are:

1. Novelty may still be considered incremental.
2. SSv2 performance may still be considered weak.
3. Static class-level attributes may still be viewed as label augmentation.
4. Lack of codec generalization may remain a limitation.

The revised paper should therefore proactively acknowledge these boundaries while strengthening the evidence for the actual contribution:
semantic enrichment and adaptive routing for efficient compressed-domain recognition.

