# Research question (restrict evidence to TOP journals/conferences only: e.g. Medi

**Question:** Research question (restrict evidence to TOP journals/conferences only: e.g. Medical Image Analysis, IEEE TMI, IEEE TPAMI, Nature Medicine/BME, MICCAI, CVPR, ICCV, ECCV, AAAI, NeurIPS, ICLR; exclude low-tier venues and preprints unless from these venues): How can TEMPORAL modeling in ultrasound VIDEO object detection exploit the DIFFERENT motion/appearance-change dynamics of blood vessels versus solid nodules under probe sweeping, to discriminate true nodules from vessel-mimicking false positives? Specifically: (1) thyroid/breast/abdominal ultrasound video nodule or lesion detection methods that use temporal context to suppress vessel/false-positive structures (e.g. negative temporal context, temporal consistency); (2) methods using LONG-RANGE / far-apart history frames (large temporal stride, memory banks) rather than adjacent frames, because at ~30fps adjacent frames barely change; (3) general video object detection long-range memory / motion-modeling mechanisms from top CV venues (MEGA CVPR2020, MAMBA AAAI2024, PTSEFormer ECCV2022, SELSA ICCV2019, FGFA ICCV2017, TransVOD) that are transferable to a real-time single-stage YOLO-style detector; (4) any top-venue work explicitly differentiating vessels from nodules/lesions using temporal or Doppler-free motion cues. For each relevant paper give: venue/year, the temporal mechanism (how history frames are sampled and fused, causal vs not), whether it targets false-positive/vessel suppression, reported gains, and code availability. Synthesize which mechanisms are most promising to port onto a causal real-time YOLO26 detector for thyroid nodule detection where vessels are unlabeled false-positive sources, and how long-range (far-frame) sampling should be designed.

# **Temporal modeling suppresses vessels** and false positives in ultrasound video detection via long-range memory and consistency.

Temporal modeling separates true anatomical lesions from tubular vessels or transient acoustic artifacts by decomposing the clinical video stream across persistent versus non-persistent representations, long-range historical context, and lightweight causal fusion modules [[1.1]](https://consensus.app/papers/gu-chu-automated-deep-learning-for-realtime-focal-liver/w:a9205972-241c-49a2-9c04-d2c744d9fb8b/) [[2.1]](https://consensus.app/papers/zhang-zhang-tracking-spatial-temporal-details-in-ultrasound/w:27e1a1ed-39ce-495e-a284-000f7499a2ac/) [[3.1]](https://consensus.app/papers/shi-zhang-practical-video-object-detection-via-feature/w:c16347af-57fe-4687-929a-88bc53dba9de/).

## Ultrasound Video Lesion Detection and False-Positive Suppression

Ultrasound video architectures exploit spatial persistence and background motion cues to filter out transient noise and vessel cross-sections that mimic solid nodules in 2D views [[1.1]](https://consensus.app/papers/gu-chu-automated-deep-learning-for-realtime-focal-liver/w:a9205972-241c-49a2-9c04-d2c744d9fb8b/) [[1.2]](https://consensus.app/papers/gu-chu-automated-deep-learning-for-realtime-focal-liver/w:a9205972-241c-49a2-9c04-d2c744d9fb8b/).

* **Auto-DFLLs (npj Digital Medicine 2026)** incorporates a causal sliding window ($T$ prior frames) using optical flow mapping and multi-round temporal attention to suppress transient false positives while operating at 30.4 FPS [[1.1]](https://consensus.app/papers/gu-chu-automated-deep-learning-for-realtime-focal-liver/w:a9205972-241c-49a2-9c04-d2c744d9fb8b/) [[1.3]](https://consensus.app/papers/gu-chu-automated-deep-learning-for-realtime-focal-liver/w:a9205972-241c-49a2-9c04-d2c744d9fb8b/) [[1.4]](https://consensus.app/papers/gu-chu-automated-deep-learning-for-realtime-focal-liver/w:a9205972-241c-49a2-9c04-d2c744d9fb8b/) [[1.2]](https://consensus.app/papers/gu-chu-automated-deep-learning-for-realtime-focal-liver/w:a9205972-241c-49a2-9c04-d2c744d9fb8b/).
* **STPFNet (IEEE TMM 2025)** applies progressive spatial-temporal memory fusion (STM-style) and feeds prior-frame prediction masks to suppress background noise and define blurred lesion contours [[4.1]](https://consensus.app/papers/tu-zhu-a-spatialtemporal-progressive-fusion-network-for/w:3fcf2a8a-0bb4-499d-bccd-d7fce89a129c/) [[4.2]](https://consensus.app/papers/tu-zhu-a-spatialtemporal-progressive-fusion-network-for/w:3fcf2a8a-0bb4-499d-bccd-d7fce89a129c/) [[4.3]](https://consensus.app/papers/tu-zhu-a-spatialtemporal-progressive-fusion-network-for/w:3fcf2a8a-0bb4-499d-bccd-d7fce89a129c/).
* **AFP & PASS (PMB 2023)** leverages adjacent-frame feature perception alongside self-supervised pre-training to resolve low tissue contrast, raising thyroid detection AP@50 from 0.256 to 0.425 [[5.1]](https://consensus.app/papers/liu-fenster-selfsupervised-enhanced-thyroid-nodule-detection/w:ee90e167-bbf5-4ed7-8700-9082c4ae9c82/) [[5.2]](https://consensus.app/papers/liu-fenster-selfsupervised-enhanced-thyroid-nodule-detection/w:ee90e167-bbf5-4ed7-8700-9082c4ae9c82/) [[5.3]](https://consensus.app/papers/liu-fenster-selfsupervised-enhanced-thyroid-nodule-detection/w:ee90e167-bbf5-4ed7-8700-9082c4ae9c82/).
* **HLST & BaS (MedIA 2026 / IEEE JBHI 2025)** selectively model intra- and peri-lesional dynamics or deploy lightweight inter-frame blocks to stabilize sequence-level boundary tracking on edge hardware [[6.1]](https://consensus.app/papers/zhang-yan-a-navigationguided-3d-breast-ultrasound-scanning/w:5bb5d93a-159f-4a6e-971e-624986038fe6/) [[7.1]](https://consensus.app/papers/xiao-zhang-deep-learningbased-medical-ultrasound-image-and/w:d9bfe6a2-1eb3-4473-829c-49065c60c141/) [[8.1]](https://consensus.app/papers/pang-li-efficient-breast-lesion-segmentation-from/w:9677516e-af36-4d22-a06e-2cc670016469/).

**Fig 1.** Overview of the Auto-DFLLs framework for real-time focal liver lesions detection in ultrasound videos. Framework Pipeline and Key Components. The system comprises four integrated stages: (1) standardized ultrasound video acquisition and preprocessing; (2) a deep learning model integrating a ResNet-based backbone with spatiotemporal fusion modules to enhance detection and reduce false positives; (3) model training using temporal context and data augmentation; and (4) comprehensive evaluation through internal and external validation, including a clinical reader study to assess Auto-DFLLs assistance for sonographers. [[1]](https://consensus.app/papers/gu-chu-automated-deep-learning-for-realtime-focal-liver/w:a9205972-241c-49a2-9c04-d2c744d9fb8b/)

*How to read: The diagram outlines the multi-stage Auto-DFLLs pipeline, tracing raw ultrasound input through ResNet-BiFPN extraction, spatiotemporal candidate fusion, and reader study evaluation.*

## Long-Range History Modeling and Memory Mechanisms

Because adjacent frames at ~30 FPS exhibit minimal motion under slow manual probe sweeping, temporal modeling requires long-range memory banks or dilated recurrence to avoid frame redundancy [[9.1]](https://consensus.app/papers/jiang-wang-subtle-motion-perception-via-memory-spatial-mamba/w:b9a9d5a5-cd81-4536-a241-eb981fb3dcbc/) [[9.2]](https://consensus.app/papers/jiang-wang-subtle-motion-perception-via-memory-spatial-mamba/w:b9a9d5a5-cd81-4536-a241-eb981fb3dcbc/) [[2.2]](https://consensus.app/papers/zhang-zhang-tracking-spatial-temporal-details-in-ultrasound/w:27e1a1ed-39ce-495e-a284-000f7499a2ac/).

| Model & Venue | Temporal Mechanism & Stride | Target FP Suppression | Gains / Efficiency | Code Availability |
| :--- | :--- | :--- | :--- | :--- |
| **Auto-DFLLs** (npj Digit Med 2026) [[1.3]](https://consensus.app/papers/gu-chu-automated-deep-learning-for-realtime-focal-liver/w:a9205972-241c-49a2-9c04-d2c744d9fb8b/) [[1.2]](https://consensus.app/papers/gu-chu-automated-deep-learning-for-realtime-focal-liver/w:a9205972-241c-49a2-9c04-d2c744d9fb8b/) | Causal sliding window ($T$ history frames), optical flow, attention [[1.4]](https://consensus.app/papers/gu-chu-automated-deep-learning-for-realtime-focal-liver/w:a9205972-241c-49a2-9c04-d2c744d9fb8b/) [[1.2]](https://consensus.app/papers/gu-chu-automated-deep-learning-for-realtime-focal-liver/w:a9205972-241c-49a2-9c04-d2c744d9fb8b/) | Suppresses transient false positives [[1.1]](https://consensus.app/papers/gu-chu-automated-deep-learning-for-realtime-focal-liver/w:a9205972-241c-49a2-9c04-d2c744d9fb8b/) | AP50: 0.7772, 30.4 FPS [[1.3]](https://consensus.app/papers/gu-chu-automated-deep-learning-for-realtime-focal-liver/w:a9205972-241c-49a2-9c04-d2c744d9fb8b/) [[1.5]](https://consensus.app/papers/gu-chu-automated-deep-learning-for-realtime-focal-liver/w:a9205972-241c-49a2-9c04-d2c744d9fb8b/) | Not stated [[1.6]](https://consensus.app/papers/gu-chu-automated-deep-learning-for-realtime-focal-liver/w:a9205972-241c-49a2-9c04-d2c744d9fb8b/) |
| **MWNet** (MedIA 2025) [[2.3]](https://consensus.app/papers/zhang-zhang-tracking-spatial-temporal-details-in-ultrasound/w:27e1a1ed-39ce-495e-a284-000f7499a2ac/) [[2.4]](https://consensus.app/papers/zhang-zhang-tracking-spatial-temporal-details-in-ultrasound/w:27e1a1ed-39ce-495e-a284-000f7499a2ac/) | Dynamic long short-term memory bank with multi-frame skip interval [[2.2]](https://consensus.app/papers/zhang-zhang-tracking-spatial-temporal-details-in-ultrasound/w:27e1a1ed-39ce-495e-a284-000f7499a2ac/) [[2]](https://consensus.app/papers/zhang-zhang-tracking-spatial-temporal-details-in-ultrasound/w:27e1a1ed-39ce-495e-a284-000f7499a2ac/) | Eliminates ambiguous location noise [[2.5]](https://consensus.app/papers/zhang-zhang-tracking-spatial-temporal-details-in-ultrasound/w:27e1a1ed-39ce-495e-a284-000f7499a2ac/) | SOTA small-nodule segmentation [[2.6]](https://consensus.app/papers/zhang-zhang-tracking-spatial-temporal-details-in-ultrasound/w:27e1a1ed-39ce-495e-a284-000f7499a2ac/) [[2.7]](https://consensus.app/papers/zhang-zhang-tracking-spatial-temporal-details-in-ultrasound/w:27e1a1ed-39ce-495e-a284-000f7499a2ac/) | Available on GitHub [[2.8]](https://consensus.app/papers/zhang-zhang-tracking-spatial-temporal-details-in-ultrasound/w:27e1a1ed-39ce-495e-a284-000f7499a2ac/) |
| **Vivim** (IEEE TCSVT 2025) [[10.1]](https://consensus.app/papers/yang-xing-vivim-a-video-vision-mamba-for-ultrasound-video/w:edf6ad50-d7e0-41d1-913a-adfc9ad6fae3/) [[10.2]](https://consensus.app/papers/yang-xing-vivim-a-video-vision-mamba-for-ultrasound-video/w:edf6ad50-d7e0-41d1-913a-adfc9ad6fae3/) | Causal Temporal Mamba Block (SSM) with cross-frame affine constraints [[10.3]](https://consensus.app/papers/yang-xing-vivim-a-video-vision-mamba-for-ultrasound-video/w:edf6ad50-d7e0-41d1-913a-adfc9ad6fae3/) [[10.2]](https://consensus.app/papers/yang-xing-vivim-a-video-vision-mamba-for-ultrasound-video/w:edf6ad50-d7e0-41d1-913a-adfc9ad6fae3/) [[10.4]](https://consensus.app/papers/yang-xing-vivim-a-video-vision-mamba-for-ultrasound-video/w:edf6ad50-d7e0-41d1-913a-adfc9ad6fae3/) | Suppresses ambiguous boundary artifacts [[10.4]](https://consensus.app/papers/yang-xing-vivim-a-video-vision-mamba-for-ultrasound-video/w:edf6ad50-d7e0-41d1-913a-adfc9ad6fae3/) | Linear complexity, outperforms CNNs [[10.5]](https://consensus.app/papers/yang-xing-vivim-a-video-vision-mamba-for-ultrasound-video/w:edf6ad50-d7e0-41d1-913a-adfc9ad6fae3/) [[10.6]](https://consensus.app/papers/yang-xing-vivim-a-video-vision-mamba-for-ultrasound-video/w:edf6ad50-d7e0-41d1-913a-adfc9ad6fae3/) | Available on GitHub [[10.7]](https://consensus.app/papers/yang-xing-vivim-a-video-vision-mamba-for-ultrasound-video/w:edf6ad50-d7e0-41d1-913a-adfc9ad6fae3/) |
| **YOLOV / YOLOV++** (IJCV 2026) [[3.2]](https://consensus.app/papers/shi-zhang-practical-video-object-detection-via-feature/w:c16347af-57fe-4687-929a-88bc53dba9de/) [[3.3]](https://consensus.app/papers/shi-zhang-practical-video-object-detection-via-feature/w:c16347af-57fe-4687-929a-88bc53dba9de/) | Candidate proposal selection and similarity-guided reference aggregation [[3.4]](https://consensus.app/papers/shi-zhang-practical-video-object-detection-via-feature/w:c16347af-57fe-4687-929a-88bc53dba9de/) [[3.5]](https://consensus.app/papers/shi-zhang-practical-video-object-detection-via-feature/w:c16347af-57fe-4687-929a-88bc53dba9de/) | Reduces false detections from blur/occlusion [[3.6]](https://consensus.app/papers/shi-zhang-practical-video-object-detection-via-feature/w:c16347af-57fe-4687-929a-88bc53dba9de/) [[3.7]](https://consensus.app/papers/shi-zhang-practical-video-object-detection-via-feature/w:c16347af-57fe-4687-929a-88bc53dba9de/) | 93.0% AP50, >30 FPS [[3.1]](https://consensus.app/papers/shi-zhang-practical-video-object-detection-via-feature/w:c16347af-57fe-4687-929a-88bc53dba9de/) | Available on GitHub [[3.8]](https://consensus.app/papers/shi-zhang-practical-video-object-detection-via-feature/w:c16347af-57fe-4687-929a-88bc53dba9de/) |
| **TransVOD / Lite** (IEEE TPAMI 2022) [[11.1]](https://consensus.app/papers/zhou-li-transvod-endtoend-video-object-detection-with/w:1d709d9c-9fbd-4697-802e-f7b35a36ef3b/) [[11.2]](https://consensus.app/papers/zhou-li-transvod-endtoend-video-object-detection-with/w:1d709d9c-9fbd-4697-802e-f7b35a36ef3b/) | Temporal Query Encoder + Temporal Deformable Attention across clips [[11.3]](https://consensus.app/papers/zhou-li-transvod-endtoend-video-object-detection-with/w:1d709d9c-9fbd-4697-802e-f7b35a36ef3b/) [[11.2]](https://consensus.app/papers/zhou-li-transvod-endtoend-video-object-detection-with/w:1d709d9c-9fbd-4697-802e-f7b35a36ef3b/) | Eliminates misclassification and false alarms [[11.4]](https://consensus.app/papers/zhou-li-transvod-endtoend-video-object-detection-with/w:1d709d9c-9fbd-4697-802e-f7b35a36ef3b/) [[11.5]](https://consensus.app/papers/zhou-li-transvod-endtoend-video-object-detection-with/w:1d709d9c-9fbd-4697-802e-f7b35a36ef3b/) | 90.0% mAP (Lite: 30 FPS) [[11.6]](https://consensus.app/papers/zhou-li-transvod-endtoend-video-object-detection-with/w:1d709d9c-9fbd-4697-802e-f7b35a36ef3b/) [[11.7]](https://consensus.app/papers/zhou-li-transvod-endtoend-video-object-detection-with/w:1d709d9c-9fbd-4697-802e-f7b35a36ef3b/) | Available on GitHub [[11.8]](https://consensus.app/papers/zhou-li-transvod-endtoend-video-object-detection-with/w:1d709d9c-9fbd-4697-802e-f7b35a36ef3b/) |

**Figure 1:** Comparison of top-tier video object detection and ultrasound video architectures across temporal fusion, false-positive suppression, and inference efficiency.

Beyond these frameworks, SMP-Mamba incorporates a similarity-guided memory bank to store non-redundant historical descriptors without computing optical flow [[9.3]](https://consensus.app/papers/jiang-wang-subtle-motion-perception-via-memory-spatial-mamba/w:b9a9d5a5-cd81-4536-a241-eb981fb3dcbc/) [[9.4]](https://consensus.app/papers/jiang-wang-subtle-motion-perception-via-memory-spatial-mamba/w:b9a9d5a5-cd81-4536-a241-eb981fb3dcbc/). Circular window transformers (CWC-Trans) reuse intermediate states in a circular queue to model long horizons at high frame rates [[12.1]](https://consensus.app/papers/cao-luo-a-circular-windowbased-cascade-transformer-for/w:55b0eb0a-ac9f-477a-9681-5d8ea7d9174b/) [[12]](https://consensus.app/papers/cao-luo-a-circular-windowbased-cascade-transformer-for/w:55b0eb0a-ac9f-477a-9681-5d8ea7d9174b/). Similarly, VideoBEV recurrently updates a single memory feature state, avoiding the memory growth typical of dense parallel windows [[13.1]](https://consensus.app/papers/han-yang-exploring-recurrent-longterm-temporal-fusion-for/w:cd5a7beb-5d33-4126-b990-4d46da36e1c9/) [[13.2]](https://consensus.app/papers/han-yang-exploring-recurrent-longterm-temporal-fusion-for/w:cd5a7beb-5d33-4126-b990-4d46da36e1c9/) [[13.3]](https://consensus.app/papers/han-yang-exploring-recurrent-longterm-temporal-fusion-for/w:cd5a7beb-5d33-4126-b990-4d46da36e1c9/) [[13.4]](https://consensus.app/papers/han-yang-exploring-recurrent-longterm-temporal-fusion-for/w:cd5a7beb-5d33-4126-b990-4d46da36e1c9/).

## Differentiating Vessels From Nodules and Adapting to YOLO26

True nodules maintain a stable 3D mass boundary that alters smoothly across probe sweeps, whereas vessels collapse, branch, elongate into tubular channels, or display dynamic fluid-motion variations [[1.1]](https://consensus.app/papers/gu-chu-automated-deep-learning-for-realtime-focal-liver/w:a9205972-241c-49a2-9c04-d2c744d9fb8b/) [[14.1]](https://consensus.app/papers/li-zhao-stnmamba-mambabased-spatialtemporal-normality/w:2c02a180-8812-4de6-bb75-ad9e99bb3481/) [[14.2]](https://consensus.app/papers/li-zhao-stnmamba-mambabased-spatialtemporal-normality/w:2c02a180-8812-4de6-bb75-ad9e99bb3481/).

```
Incoming Stream (t) ──> YOLO26 Neck (C3k2 / PAN) ──> Top-K Proposals (x_t)
                             │                                 │
   Causal Readout / Fusion <─┴─ State Space Update (Mamba) <───┘
               │                    │
               ▼                    ▼
     Temporal Consistency Head   Keyframe Bank [t-k, t-2k] (Cosine Sim Filter)
```

Dual-stream motion learning methods such as STNMamba capture motion patterns by processing consecutive frame differences without heavy optical flow overhead, separating dynamic boundary shifts from static tissue [[14.3]](https://consensus.app/papers/li-zhao-stnmamba-mambabased-spatialtemporal-normality/w:2c02a180-8812-4de6-bb75-ad9e99bb3481/) [[14.4]](https://consensus.app/papers/li-zhao-stnmamba-mambabased-spatialtemporal-normality/w:2c02a180-8812-4de6-bb75-ad9e99bb3481/) [[14.2]](https://consensus.app/papers/li-zhao-stnmamba-mambabased-spatialtemporal-normality/w:2c02a180-8812-4de6-bb75-ad9e99bb3481/) [[14.5]](https://consensus.app/papers/li-zhao-stnmamba-mambabased-spatialtemporal-normality/w:2c02a180-8812-4de6-bb75-ad9e99bb3481/). Similarly, motion-aware boundary supervision learns subtle temporal boundary variations between adjacent time steps to refine spatial saliency [[15.1]](https://consensus.app/papers/zhao-liang-motionaware-memory-network-for-fast-video-salient/w:a63714a5-c997-4831-80f9-839c96684197/) [[15.2]](https://consensus.app/papers/zhao-liang-motionaware-memory-network-for-fast-video-salient/w:a63714a5-c997-4831-80f9-839c96684197/) [[15.3]](https://consensus.app/papers/zhao-liang-motionaware-memory-network-for-fast-video-salient/w:a63714a5-c997-4831-80f9-839c96684197/). For integrating temporal modeling into a real-time, single-stage causal YOLO26 architecture:

1. **Decouple Candidate Selection from Dense Maps**: Mimicking YOLOV, extract top-$K$ candidate feature tokens from the detector neck rather than computing dense spatial attention across all pixels, cutting memory bottlenecks during multi-frame reasoning [[3.2]](https://consensus.app/papers/shi-zhang-practical-video-object-detection-via-feature/w:c16347af-57fe-4687-929a-88bc53dba9de/) [[3.4]](https://consensus.app/papers/shi-zhang-practical-video-object-detection-via-feature/w:c16347af-57fe-4687-929a-88bc53dba9de/).
2. **Causal SSM State Maintenance**: Port a linear-complexity Temporal Mamba block (as in Vivim or MambaDETR) directly into the neck, maintaining an evolving hidden state vector causally without accessing future frames [[10.3]](https://consensus.app/papers/yang-xing-vivim-a-video-vision-mamba-for-ultrasound-video/w:edf6ad50-d7e0-41d1-913a-adfc9ad6fae3/) [[10.2]](https://consensus.app/papers/yang-xing-vivim-a-video-vision-mamba-for-ultrasound-video/w:edf6ad50-d7e0-41d1-913a-adfc9ad6fae3/) [[16.1]](https://consensus.app/papers/ning-lu-mambadetr-querybased-temporal-modeling-using-state/w:8f6731bc-573f-4387-aec8-3ee474809e63/).
3. **Sparse Strided Keyframe Bank**: Rather than sampling adjacent frames where features barely drift, maintain a small FIFO bank populated at fixed temporal strides ($\Delta t \approx 5\text{--}15$ frames) or triggered by probe displacement thresholds [[12.1]](https://consensus.app/papers/cao-luo-a-circular-windowbased-cascade-transformer-for/w:55b0eb0a-ac9f-477a-9681-5d8ea7d9174b/) [[9.2]](https://consensus.app/papers/jiang-wang-subtle-motion-perception-via-memory-spatial-mamba/w:b9a9d5a5-cd81-4536-a241-eb981fb3dcbc/) [[2.2]](https://consensus.app/papers/zhang-zhang-tracking-spatial-temporal-details-in-ultrasound/w:27e1a1ed-39ce-495e-a284-000f7499a2ac/) [[2]](https://consensus.app/papers/zhang-zhang-tracking-spatial-temporal-details-in-ultrasound/w:27e1a1ed-39ce-495e-a284-000f7499a2ac/).
4. **Consistency Verification Head**: Apply temporal query fusion (as shown in TransVOD) or contrastive similarity gates across the sparse memory to penalize candidates that lack cross-slice volumetric persistence, suppressing transient vessel cross-sections [[11.9]](https://consensus.app/papers/zhou-li-transvod-endtoend-video-object-detection-with/w:1d709d9c-9fbd-4697-802e-f7b35a36ef3b/) [[11.2]](https://consensus.app/papers/zhou-li-transvod-endtoend-video-object-detection-with/w:1d709d9c-9fbd-4697-802e-f7b35a36ef3b/) [[1.1]](https://consensus.app/papers/gu-chu-automated-deep-learning-for-realtime-focal-liver/w:a9205972-241c-49a2-9c04-d2c744d9fb8b/).
 
_These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at https://consensus.app. © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent._
 
## References
 
`1.` Gu J, Chu Y, Chen D, et al. Automated deep learning for real-time focal liver lesions detection in ultrasound videos a multicenter study. *npj Digital Medicine.* 2026;9. doi:10.1038/s41746-026-02629-y
 
> [1.1] Drawing on clinical observations that true lesions exhibit spatial continuity and temporal persistence across frames, while false positives are often transient, our model analyzes the entire frame sequence to suppress background noise and isolated artifacts. (Results)
 
> [1.2] The core architecture of Auto-DFLLs incorporated three specialized modules: (1) A lightweight ResNet34-BiFPN backbone for cross-scale feature fusion to enhance small lesion detection24,25; (2) A lesion-level spatiotemporal fusion module employing adaptive feature weighting and multi-round temporal attention to integrate candidate lesion features across frames; (3) An image-level spatiotemporal module utilizing optical flow mapping to suppress false positives through background motion analysis. (Results)
 
> [1.3] Furthermore, the Auto-DFLLs model achieves true clinical real-time performance, operating at a processing speed of 30.4 frames per second (FPS) with only two RTX 4090 GPUs. (Results)
 
> [1.4] During training, input samples were generated through a sliding window approach, with each sample comprising the current frame and T preceding frames (T+1 total frames). (Results)
 
> [1.5] On internal validation, Auto-DFLLs achieved an AP50 (average precision at IoU = 50%) of 0.7772, Pr70 (precision at 70% recall) of 0.7967, and FP70 (false positives at 70% recall) of 3.4688. (Abstract)
 
> [1.6] We developed Auto-DFLLs, an automated deep learning model based on ResNet and FPN architectures to detect FLLs in ultrasound videos. (Abstract)
 
`2.` Zhang C, Zhang R, Wang J. Tracking spatial temporal details in ultrasound long video via wavelet analysis and memory bank. *Medical Image Analysis.* 2025;109:103904-103904. doi:10.1016/j.media.2025.103904
 
> [2.1] This can be attributed to the small nodule sizes, which exacerbate the inherent challenges of ultrasound image segmentation and increase the likelihood of object loss for video tracking. (Discussion)
 
> [2.2] In contrast, this study proposed a dynamically updated long short-term memory bank to model long-range temporal dependencies without the annotation (Body)
 
> [2.3] Tracking spatial temporal details in ultrasound long video via wavelet analysis and memory bank Chenxiao Zhang 1 , Runshi Zhang 1 , Junchen Wang - We proposed a memory bank- and wavelet-based network for segmentation. (Results)
 
> [2.4] A long short-term memory bank using cross-attention and memory compression mechanisms is designed to track objects in long video. (Abstract)
 
> [2.5] Our method can effectively address challenges such as blurred boundaries, confusing locations, speckle noise in ultrasound videos and shows superior performance for small object in long-duration ultrasound sequences. (Discussion)
 
> [2.6] The cross-dataset comparison demonstrates the adaptability of our method to diverse ultrasound video types, with particularly notable improvements on thyroid nodule datasets. (Discussion)
 
> [2.7] In particular, our method can more accurately segment small thyroid nodules, demonstrating its effectiveness for cases involving small ultrasound objects in long video. (Abstract)
 
> [2.8] The code is available at https://github.com/XiAooZ/MWNet. (Abstract)
 
`3.` Shi Y, Zhang T, Guo X. Practical Video Object Detection via Feature Selection and Aggregation. *International Journal of Computer Vision.* 2026;134. doi:10.1007/s11263-025-02700-3
 
> [3.1] Notably, our model reaches a new record performance, i.e., 93.0% AP50 at over 30 FPS on the ImageNet VID dataset on a single 3090 GPU, making it a compelling option for large-scale or real-time applications. (Abstract)
 
> [3.2] | Method | Base Detector | # Processed Frames | # Candidates/frame | Memory(GB) | Time(ms) | |---------------------|----------------------|----------------------|----------------------|--------------|------------| | SELSA [61] | Faster R-CNN [44] | 21 | 300 | 1.8 | 7.4 | | TransVOD [25] | Deformable DETR [70] | 14 | 300 | 1.3 | 6.2 | (Results)
 
> [3.3] To tackle the above issues, this study invents a very simple yet potent strategy of feature selection and aggregation, gaining significant accuracy at marginal computational expense. (Abstract)
 
> [3.4] Concretely, for cutting the massive computation and memory consumption from the dense prediction characteristic of one-stage object detectors, we first condense candidate features from dense prediction maps. (Abstract)
 
> [3.5] Then, the relationship between a target frame and its reference frames is evaluated to guide the aggregation. (Abstract)
 
> [3.6] The visual results clearly reveal that our model can significantly boost the precision of predictions. (Results)
 
> [3.7] Additionally, it reliably detects occluded objects with high confidence. (Results)
 
> [3.8] The implementation is simple, and accessible at https://github.com/YuHengsss/YOLOV. (Abstract)
 
`4.` Tu Z, Zhu Z, Duan Y, Jiang B, Wang Q, Zhang C. A Spatial-Temporal Progressive Fusion Network for Breast Lesion Segmentation in Ultrasound Videos. *IEEE Transactions on Multimedia.* 2025;27:7470-7482. doi:10.1109/tmm.2025.3599028
 
> [4.1] Third, we propose to exploit the segmentation result of previous frame as the prior knowledge to suppress the noisy background and learn more robust representation. (Results)
 
> [4.2] The Temporal Fusion Module, similar to STM [31], fuses the current frame with the past frames to find similar regions between the current and the past frames in the video sequence. (Results)
 
> [4.3] It takes the predicted lesion region from the previous frame as a priori knowledge to suppress the noisy background and highlight the foreground to achieve the localisation of the lesion region. (Results)
 
`5.` Liu N, Fenster A, Tessier D, Jin C, Gou S, Chong J. Self-supervised enhanced thyroid nodule detection in ultrasound examination video sequences with multi-perspective evaluation. *Physics in Medicine and Biology.* 2023;68:235007-235007. doi:10.1088/1361-6560/ad092a
 
> [5.1] To address these challenges, we extend thyroid nodule detection from image-based to video-based using the temporal context information in ultrasound videos.Approach.We propose a video-based deep learning model with adjacent frame perception (AFP) for accurate and real-time thyroid nodule detection. (Abstract)
 
> [5.2] Compared to image-based methods, AFP can aggregate semantically similar contextual features in the video. (Abstract)
 
> [5.3] Our proposed AFP improved AP@50 from 0.256 to 0.390, while the PASS-enhanced AFP further improved the AP@50 to 0.425. (Abstract)
 
`6.` Zhang Y, Yan Y, Wang K, et al. A navigation-guided 3D breast ultrasound scanning and reconstruction system for automated multi-lesion spatial localization and diagnosis. *Medical Image Analysis.* 2026;110:103965-103965. doi:10.1016/j.media.2026.103965
 
> [6.1] Specifically, a Hybrid Lesion-informed Spatiotemporal Transformer (HLST) is proposed to selectively fuse intra- and peri-lesional dynamics augmented from a prompt-driven BUS-SAM-2 foundation model for sequence-level classification. (Abstract)
 
`7.` Xiao X, Zhang J, Shao Y, et al. Deep Learning-Based Medical Ultrasound Image and Video Segmentation Methods: Overview, Frontiers, and Challenges. *Sensors.* 2025;25:2361-2361. doi:10.3390/s25082361
 
> [7.1] [90] proposed BaS, a real-time breast lesion segmentation model for ultrasound videos, employing Stem modules and BaSBlock for inter-/intra-frame analysis. (Body)
 
`8.` Pang Y, Li Y, Huang T, et al. Efficient Breast Lesion Segmentation From Ultrasound Videos Across Multiple Source-Limited Platforms. *IEEE Journal of Biomedical and Health Informatics.* 2025;29:8890-8903. doi:10.1109/jbhi.2025.3543435
 
> [8.1] BaS integrates the Stem module and BaSBlock to refine representations through inter-and intra-frame analysis on ultrasound videos. (Introduction)
 
`9.` Jiang T, Wang Y, He X, Liao B. Subtle motion perception via memory spatial mamba for video camouflaged object detection. *Neurocomputing.* 2026;711:135469-135469. doi:10.1016/j.neucom.2026.135469
 
> [9.1] In video scenarios, camouflaged objects are characterized by subtle and intermittent motion, whereas adjacent frames contain substantial temporal redundancy. (Abstract)
 
> [9.2] Thus, the central temporal-modeling challenge is not to accumulate dense historical observations indiscriminately, but to preserve compact, non-redundant cues that can guide current-frame inference. (Abstract)
 
> [9.3] To this end, we propose Subtle Motion Perception via Memory Spatial Mamba (SMP-Mamba), a flow-free VCOD framework for redundancy-aware temporal modeling. (Abstract)
 
> [9.4] A Similarity-Guided Memory Bank selectively retains non-redundant historical descriptors for subsequent inference. (Abstract)
 
`10.` Yang Y, Xing Z, Yu L, Fu H, Huang C, Zhu L. Vivim: A Video Vision Mamba for Ultrasound Video Segmentation. *IEEE Transactions on Circuits and Systems for Video Technology.* 2025;35:10293-10304. doi:10.1109/tcsvt.2025.3563411
 
> [10.1] To this end, this paper presents a Video Vision Mamba-based framework, dubbed as Vivim, for ultrasound video segmentation tasks. (Abstract)
 
> [10.2] Our Vivim can effectively compress the long-term spatiotemporal representation into sequences at varying scales with our designed Temporal Mamba Block. (Abstract)
 
> [10.3] Unfortunately, vanilla SSMs failed to simultaneously capture causal temporal cues and preserve non-casual spatial information. (Abstract)
 
> [10.4] We also introduce an improved boundary-aware affine constraint across frames to enhance the discriminative ability of Vivim on ambiguous lesions. (Abstract)
 
> [10.5] Recently, state space models (SSMs), famous by Mamba, have exhibited linear complexity and impressive achievements in efficient long sequence modeling, which have developed deep neural networks by expanding the receptive field on many vision tasks significantly. (Abstract)
 
> [10.6] Extensive experiments on thyroid segmentation in ultrasound videos, breast lesion segmentation in ultrasound videos, and polyp segmentation in colonoscopy videos demonstrate the effectiveness and efficiency of our Vivim, superior to existing methods. (Abstract)
 
> [10.7] The code and dataset are available at: https://github.com/scott-yjyang/Vivim (Abstract)
 
`11.` Zhou Q, Li X, He L, et al. TransVOD: End-to-End Video Object Detection With Spatial-Temporal Transformers. *IEEE Transactions on Pattern Analysis and Machine Intelligence.* 2022;45:7853-7869. doi:10.1109/tpami.2022.3223955
 
> [11.1] In this paper, we present TransVOD, the first end-to-end video object detection system based on simple yet effective spatial-temporal Transformer architectures. (Abstract)
 
> [11.2] Our temporal transformer consists of two components: Temporal Query Encoder (TQE) to fuse object queries, and Temporal Deformable Transformer Decoder (TDTD) to obtain current frame detection results. (Abstract)
 
> [11.3] In particular, we present a temporal Transformer to aggregate both the spatial object queries and the feature memories of each frame. (Abstract)
 
> [11.4] The still image detector is easy to cause false detection ( e.g., turtle detected as a lizard) and missed detection ( e.g., zebra not detected), in the case of motion blur, part occlusion. (Results)
 
> [11.5] Thus, our TransVOD not only increases the confidence of correct prediction, but also effectively reduces the number of cases that are missed or falsely detected. (Results)
 
> [11.6] In particular, our proposed TransVOD++ sets a new state-of-the-art record in terms of accuracy on ImageNet VID with 90.0 % mAP. (Abstract)
 
> [11.7] Our proposed TransVOD Lite also achieves the best speed and accuracy trade-off with 83.7 % mAP while running at around 30 FPS on a single V100 GPU device. (Abstract)
 
> [11.8] Code and models are available at https://github.com/SJTU-LuHe/TransVOD. (Abstract)
 
> [11.9] As shown in the figure, for each frame in each clip, our temporal object query has more concentrated and precise results on foreground objects while the original spatial object query has more diffuse results. (Results)
 
`12.` Cao S, Luo W, Wang B, Song R, Zhang W,  L. A Circular Window-Based Cascade Transformer for Online Action Detection. *IEEE Transactions on Multimedia.* 2026;28:8151-8163. doi:10.1109/tmm.2026.3676664
 
> [12.1] (i) a window-based hierarchical architecture that captures fine-grained temporal dynamics, and a circular window updating mechanism that enables efficient long-term history modeling; (Abstract)
 
`13.` Han C, Yang J, Sun J, et al. Exploring Recurrent Long-Term Temporal Fusion for Multi-View 3D Perception. *IEEE Robotics and Automation Letters.* 2024;9:6544-6551. doi:10.1109/lra.2024.3401172
 
> [13.1] b) Efficiency of Temporal Recurrent Fusion: VideoBEV recurrently fuses the history BEV feature. (Results)
 
> [13.2] Hence, only one BEV feature memory needs to be stored during inference. (Results)
 
> [13.3] When a new frame comes, we only need to fuse its BEV feature to the stored one with a lightweight recurrent fusion module. (Results)
 
> [13.4] As shown in Fig. 5, when increasing the number of used history frames, the overhead of memory and latency is consistently lower compared to SOLOFusion [4], which is nearly the same as that without any history frame. (Results)
 
`14.` Li Z, Zhao M, Xuan Y, et al. STNMamba: Mamba-Based Spatial-Temporal Normality Learning for Video Anomaly Detection. *IEEE Transactions on Multimedia.* 2026;28:5131-5143. doi:10.1109/tmm.2026.3664969
 
> [14.1] It is widely recognized in the VAD community that detecting anomalies depends on both appearance and motion cues [8]. (Results)
 
> [14.2] To address these problems, we utilize the corresponding RGB differences Y t -n : t -1 as the input to E t to effectively capture the motion patterns in context frames, providing a more lightweight and efficient alternative. (Results)
 
> [14.3] Most existing two-stream frameworks [7], [9] use optical flow to capture motion information. (Results)
 
> [14.4] Moreover, the use of optical flow can lead to additional computational burden. (Results)
 
> [14.5] Firstly, we develop a dual-encoder architecture, where the spatial encoder equipped with Multi-Scale Vision Space State Blocks (MS-VSSB) extracts multi-scale appearance features, and the temporal encoder employs Channel-Aware Vision Space State Blocks (CA-VSSB) to capture significant motion patterns. (Results)
 
`15.` Zhao XY, Liang H, Li P, et al. Motion-Aware Memory Network for Fast Video Salient Object Detection. *IEEE Transactions on Image Processing.* 2024;33:709-721. doi:10.1109/tip.2023.3348659
 
> [15.1] Moreover, drawing inspiration from the boundary supervision employed in ISOD, we have integrated a comparable boundary motion supervision for VSOD. (Results)
 
> [15.2] During the encoding stage, we extract high-level temporal features from the current frame and its adjacent frames, which is more efficient and practical than methods reliant on optical flow. (Results)
 
> [15.3] Moreover, inspired by the boundary supervision prevalent in image salient object detection (ISOD), we design a motion-aware loss that predicts object boundary motion, and simultaneously perform multitask learning for VSOD and object motion prediction. (Results)
 
`16.` Ning T, Lu K, Jiang X, Lan X, Xue J. MambaDETR: Query-based Temporal Modeling using State Space Model for Multi-View 3D Object Detection. *IEEE Transactions on Multimedia.* 2026;abs/2411.13628:1-10. doi:10.1109/tmm.2026.3741644
 
> [16.1] In this paper, we propose a novel method called MambaDETR, whose main idea is to implement temporal fusion in the efficient state space. (Abstract)
