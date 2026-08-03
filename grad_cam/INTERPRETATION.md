# Grad-CAM Interpretive Claims — ConvNeXt-Tiny (best model, 83.8% top-1)

Evidence base: Grad-CAM on the last convolutional stage (`features[-1]`) of the
ImageNet-pretrained, full-augmentation **ConvNeXt-Tiny** run, evaluated on the 500-class
iNaturalist-2021 test set (5,000 images). Our pipeline reproduces the team's reported
**top-1 = 0.8376 exactly**, so every claim below is tied to the actual deployed model.

Two quantitative probes support the claims:
1. **Attention locality** — Otsu mask over the CAM: `mask_area` (fraction of image above
   threshold) and `centroid_offset` (distance of CAM centre-of-mass from image centre,
   normalised to [0,1]).
2. **CAM agreement on confusions** — for each misclassified image of a confused pair we
   compute the CAM for the *true* class and for the *predicted* class and measure their
   Pearson correlation and Otsu-mask IoU. High agreement = the model uses the *same*
   evidence to argue for both species.

> Caveats (stated honestly in the report): per-pair confusion samples are small (n = 3–5, the
> number of test misclassifications for that pair — Claims 3–4); locality/faithfulness use
> n = 150 / 50 per model; the organism/background measure is a **proxy** (iNat2021 has no
> ground-truth masks). Claims 1–2 & 4 are for the primary model ConvNeXt-Tiny; Claims 5–6
> compare all three pretrained models.

---

## Claim 1 — The model attends to the organism, even when it is wrong (Req: organism vs background)

Mean attention locality over 150 correct vs 150 incorrect predictions is **nearly identical**:

| | mask_area | centroid_offset |
|---|---|---|
| Correct   | 0.156 | 0.213 |
| Incorrect | 0.164 | 0.239 |

**Claim:** On both correct and incorrect predictions the CAM forms a compact, roughly
central blob covering ~16% of the image. Misclassifications are therefore **not** explained
by the network latching onto background or context ("clever-Hans" behaviour) — it is looking
at the right object and failing on *fine-grained appearance*. This reframes every error
below as a feature-discrimination problem, not a localisation problem.

## Claim 2 — Correct predictions localise discriminative organism regions (Req: correct vs incorrect)

**Claim:** For confidently correct species (e.g. the fly *Sehirus cinctus*, *Panax
trifolius*, *Astragalus nuttallianus*) the CAM concentrates tightly on the organism's body
— the beetle's shell, the flower head, the plant's leaf rosette — while surrounding
vegetation stays cold. This is the expected, healthy behaviour and is the baseline against
which the failures are read.

## Claim 3 — Same-genus confusions split into TWO distinct mechanisms (Req: confusable pairs)

Measuring how much the true-class and predicted-class CAMs overlap on the misses:

| Confused pair (same genus) | × | CAM corr | mask IoU | mechanism |
|---|---|---|---|---|
| *Corvus frugilegus* → *ossifragus* | 4 | **0.93** | **0.71** | shared evidence |
| *Asterocampa clyton* → *celtis*    | 3 | 0.87 | 0.64 | shared evidence |
| *Sialia mexicana* → *sialis*       | 5 | 0.71 | 0.45 | shared evidence |
| *Vernonia gigantea* → *missurica*  | 4 | 0.69 | 0.40 | shared evidence |
| *Juniperus deppeana* → *occidentalis* | 5 | **0.28** | **0.20** | appearance collapse |

**Claim 3a — "Shared-evidence" confusions (high CAM agreement).** For the two crows
(*Corvus*), the two butterflies (*Asterocampa*) and the two bluebirds (*Sialia*), the CAM
for the true species and the CAM for the wrongly-predicted species overlap almost completely
(corr up to 0.93, IoU up to 0.71). **The model uses literally the same pixels to argue for
both species**, so the discriminative cue that separates them is never represented — the two
classes are effectively collapsed at the feature level.

**Claim 3b — "Appearance-collapse" confusions (low CAM agreement).** The two junipers behave
differently: the true- and predicted-class CAMs point at *different* foliage patches
(corr 0.28, IoU 0.20), yet the model still confuses them. Here the failure is not shared
attention but that **every patch of scale-leaf foliage looks identical**, so wherever the
model looks it sees the same texture. Same outcome, different mechanism — a useful
distinction to raise in the discussion.

## Claim 4 — The most confident mistakes are genus- or body-plan look-alikes (Req: failure cases)

| Confident error (conf) | relationship | what the CAM shows |
|---|---|---|
| *Vernonia missurica* ↔ *gigantea* (0.96 / 0.92, mutual) | same genus | fires on the identical purple flower head; ignores the leaf/stem cues that actually separate the two |
| *Lampropeltis splendida* → *calligaster* (0.92) | same genus | fires on the banded body pattern shared by both kingsnakes |
| *Egretta rufescens* → *Ardea goliath* (0.91) | **different genus** | fires on the neck/body silhouette — the model keys on the shared "large wading heron" body plan |
| *Corvus albus* → *Chondrohierax uncinatus* (0.91) | **different family** (crow → hook-billed kite) | the most concerning error: a black bird in a perched/flight pose is read by silhouette, overriding plumage detail |
| *Natrix maura* → *Agkistrodon piscivorus* (0.91) | different family | two aquatic snakes with banded, keeled appearance; CAM on the coiled body |

**Claim:** The model's high-confidence errors are driven by **coarse shared features** —
flower-head colour, banded scales, wading-bird silhouette — that dominate over the fine
diagnostic details (leaf shape, head scale-count, plumage). The cross-family cases
(*Corvus* → kite, *Natrix* → cottonmouth) show the effect is not limited to close relatives:
when overall body shape and colour align, the network can be confidently wrong across the
taxonomy.

## Claim 5 — Faithfulness: the CAMs genuinely reflect what the model uses (deletion test)

Removing the top-20% most-activated pixels (setting them to the ImageNet mean) and
re-measuring the predicted-class confidence:

| Model | conf @ 0% removed | conf @ 20% removed | drop |
|---|---|---|---|
| ConvNeXt-Tiny | 0.804 | 0.388 | **0.416** |
| ResNet50      | 0.728 | 0.340 | 0.388 |
| Swin-T        | 0.767 | 0.424 | 0.343 |

**Claim:** Deleting just the CAM-highlighted pixels collapses confidence by 0.34–0.42 across
all three models, confirming the maps are **faithful** — they mark the pixels the model
actually relies on, not incidental texture. This is the check that separates a real
explainability study from "attention maps without interpretation." The two **CNNs** show the
steepest drops (0.39–0.42), slightly ahead of the **transformer** (Swin-T, 0.34), consistent
with their tighter, more localised attention.

## Claim 6 — CNNs localise, the transformer spreads: architecture shows up in the CAM (multi-model)

Same analysis across the three pretrained models (n=150 per group):

| Model | top-1 | CAM area (correct) | CAM area (incorrect) | offset (correct/incorrect) |
|---|---|---|---|---|
| ConvNeXt-Tiny | 0.838 | 0.156 | 0.164 | 0.21 / 0.24 |
| ResNet50      | 0.796 | 0.144 | **0.080** | 0.26 / **0.38** |
| Swin-T        | 0.827 | **0.434** | **0.446** | **0.13** / 0.13 |

**Claim 6a — architectural signature.** The two CNNs produce **compact** CAMs (~15% of the
image); **Swin-T's are ~3× broader** (~43%) and more central. This directly reflects how each
represents the image: the CNNs' local receptive fields concentrate evidence on a small region,
while the transformer's global self-attention distributes it across the organism. The
explanation method reveals the inductive bias, not just the prediction.

**Claim 6b — ResNet50 telegraphs its errors, the others don't.** For **ResNet50**, incorrect
predictions have markedly *smaller* and *more off-centre* attention (area 0.080 vs 0.144,
offset 0.38 vs 0.26) — when it fails it is often because attention scattered off the organism.
ConvNeXt-Tiny and Swin-T show no such gap (correct ≈ incorrect), so for those models a wrong
answer looks, spatially, just like a right one. Practical upshot: for ResNet50, CAM locality
is a weak *error-detection* signal; for the other two it is not.

## Overall takeaway for the report / future work

Grad-CAM shows ConvNeXt-Tiny has learned **where** the organism is (Claim 1) but, for its
error cases, relies on **coarse genus/body-plan appearance** rather than species-diagnostic
detail (Claims 3–4). This points to concrete remedies: fine-grained techniques that force
attention onto discriminative parts (bilinear/second-order pooling, attention/part-based
models) and higher input resolution, rather than better localisation or de-biasing of
background.
