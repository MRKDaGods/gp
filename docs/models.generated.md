# Models & Checkpoints - Generated Snapshot

> Generated from `configs/model_registry.yaml`. Do not edit this Phase 1 snapshot by hand.

| ID | Name | Task | Dataset | Status | Runnable locally | Key metrics |
|---|---|---|---|---|---:|---|
| vehicle_mtmc_14e_b1 | Vehicle MTMC 14e B1 production | mtmc_vehicle | cityflowv2 | production | yes | mtmc_idf1=0.77936 (verified) |
| vehicle_mtmc_14k_v1_k7 | Vehicle MTMC 14k v1 K7 R50-IBN research | mtmc_vehicle | cityflowv2 | research | no | mtmc_idf1=0.78079 (verified) |
| person_mtmc_12b | Person MTMC 12b production | mtmc_person | wildtrack | production | yes | idf1_groundplane=0.947 (verified)<br>moda_groundplane=0.903 (verified) |
| person_detector_12a_mvdetr | Person detector 12a MVDeTr reference | detector_only | wildtrack | reference | no | moda=0.921 (unverified)<br>precision=0.947 (verified)<br>recall=0.966 (verified) |
| veri776_14t_fusion | VeRi-776 14t CLIP-SENet x TransReID fusion | single_cam_reid | veri776 | research | no | map=0.933 (verified)<br>r1=0.9845 (verified) |
| veri776_09v_v17_transreid | VeRi-776 09v v17 TransReID baseline | single_cam_reid | veri776 | production | no | map=0.8997 (verified)<br>r1=0.9833 (verified) |
| veri776_clipsenet_v6 | VeRi-776 CLIP-SENet v6 expert | single_cam_reid | veri776 | research | no | map_post_rerank=0.9154 (verified)<br>r1=0.9732 (verified) |
| deadend_vehicle_csls | Dead end vehicle CSLS association | mtmc_vehicle | cityflowv2 | dead_end | no | mtmc_idf1_delta_pp=-34.7 (verified) |
| deadend_vehicle_aflink | Dead end vehicle AFLink motion linking | mtmc_vehicle | cityflowv2 | dead_end | no | mtmc_idf1_delta_pp=-3.82 (verified) |
