# eval/ — Evaluation Results

## Files

| File | Rows | Description |
|------|------|-------------|
| `car_eval.csv` | 466 | All car pick-and-place rollouts |
| `pen_eval.csv` | 232 | All pick-pen rollouts |

---

## CSV Schema

### car_eval.csv

| Column | Description |
|--------|-------------|
| `task` | Always `car_pick_and_place` |
| `hf_model_id` | Full HF Hub model ID (`jonathm126/act-so101_car_pick_and_place-...`) |
| `hf_eval_dataset` | HF Hub dataset of recorded evaluation episodes for this model |
| `checkpoint` | Training step used for evaluation (integer) |
| `K` | Action chunk size (`n_action_steps`) |
| `TE` | Temporal ensembling coefficient (0 = disabled) |
| `episode` | Episode index within the evaluation run |
| `target_pos` | Target bin index {1, 2, 3} |
| `source_pos` | Source position {1–5} = in-sample; range e.g. `1-3` = out-of-sample |
| `orientation` | Car orientation in degrees; {0, 90, 180, 270} = in-sample; diagonal = out-of-sample |
| `score` | Episode success: 0, 0.5, or 1 |
| `eval_type` | `in_sample` or `out_of_sample` (auto-derived from pose values) |
| `experiment_note` | Context label for multi-config sheets (e.g. `K=50, TE=0, ckpt=100000`) |
| `row_note` | Free-text notes recorded during evaluation |

### pen_eval.csv

Same schema except the pose columns are:

| Column | Description |
|--------|-------------|
| `position` | Pen position {1–5} = in-sample; range e.g. `2-4` = out-of-sample |
| `rotation` | Pen rotation in degrees; {0, 90, 180, 270} = in-sample; diagonal = out-of-sample |

---


### In-sample vs Out-of-sample

**In-sample** poses match the training distribution: discrete integer positions and cardinal orientations only.

**Out-of-sample** poses were never seen during training: source/position given as a range (e.g. `1-3` = placed between positions 1 and 3) and/or a diagonal orientation (e.g. 135°).


---

### Car pick-and-place

| HF Model | HF Eval Dataset | Notes |
|----------|----------------|-------|
| `act-so101_car_pick_and_place-25_episodes_v0` | `eval_..._25_episodes_v0_real_v0` | 25 ep, (4,1), K=100 |
| `act-so101_car_pick_and_place-50_episodes_v2` | `eval_..._50_episodes_v2_12_test_case` | 50 ep, (4,1), K=100 & K=50 ablation |
| `act-so101_car_pick_and_place-50_episodes_v4` | `eval_..._50_episodes_v4_real_v0` | 50 ep, (6,2), K=100 |
| `act-so101_car_pick_and_place-96_episodes_v0` | `eval_...-96_episodes_v0-real_v0` | 96 ep, (4,1) — master model; covers K/TE sweep + in/out-of-sample |
| `act-so101_car_pick_and_place-96_episodes_v1` | `eval_..._96_episodes_v1_real_v0` | 96 ep, (4,1), K=150 |
| `act-so101_car_pick_and_place-bbox-yolo_v3` | `eval_...-bbox_yolo_v3-real_v0` | YOLO state augmentation |

### Pick pen

| HF Model | HF Eval Dataset | Notes |
|----------|----------------|-------|
| `act-so101_pick_pen-25_episodes_v0` | `eval_..._25_episodes_v0_real_v0` | 25 ep, (4,1), K=100 |
| `act-so101_pick_pen-baseline_v0` | `eval_...-v0-real_v0` | 50 ep, (6,2), K=100 |
| `act-so101_pick_pen-v1` | `eval_..._v1_real_v0` | 50 ep, (4,1), K=100 |
| `act-so101_pick_pen-v3` | `eval_..._v3_real_v0` | 50 ep, (8,4), K=100 |
| `act-so101_pick_pen-v4` | `eval_..._v4_real_v0` | 75 ep, (4,1) — covers K/TE sweep |
| `act-so101_pick_pen-v5` | `eval_..._v5_real_v0` | 75 ep, (4,1), K=150 |
| `act-so101_pick_pen-bbox-yolo_v0` | `eval_...-bbox_yolo_v0-real_v0` | YOLO state augmentation |