# Experiment results

## Accuracy (tuned threshold) by test slice

| slice | e1_baseline | e2_hinglish_aug | e3_tinymel_scratch | e4_no_pause_aug |
|---|---|---|---|---|
| overall | 0.930 (n=9329) | 0.938 (n=9329) | 0.871 (n=9329) | 0.937 (n=9329) |
| english | 0.938 (n=7820) | 0.938 (n=7820) | 0.868 (n=7820) | 0.937 (n=7820) |
| hindi | 0.931 (n=1284) | 0.932 (n=1284) | 0.884 (n=1284) | 0.928 (n=1284) |
| hinglish | 0.631 (n=225) | 0.951 (n=225) | 0.871 (n=225) | 0.964 (n=225) |
| filler | 0.912 (n=2381) | 0.914 (n=2381) | 0.856 (n=2381) | 0.905 (n=2381) |
| human_audio | 0.946 (n=5367) | 0.947 (n=5367) | 0.877 (n=5367) | 0.946 (n=5367) |

## AUC by test slice

| slice | e1_baseline | e2_hinglish_aug | e3_tinymel_scratch | e4_no_pause_aug |
|---|---|---|---|---|
| overall | 0.981 | 0.983 | 0.944 | 0.984 |
| english | 0.984 | 0.983 | 0.941 | 0.983 |
| hindi | 0.985 | 0.984 | 0.962 | 0.983 |
| hinglish | 0.612 | 0.986 | 0.949 | 0.993 |
| filler | 0.971 | 0.971 | 0.940 | 0.970 |
| human_audio | 0.988 | 0.987 | 0.948 | 0.988 |

## Model footprint & training

| metric | e1_baseline | e2_hinglish_aug | e3_tinymel_scratch | e4_no_pause_aug |
|---|---|---|---|---|
| params | 7,885,953 | 7,885,953 | 507,265 | 7,885,953 |
| int8 ONNX MB | 8.500 | 8.500 | 1.280 | 8.500 |
| int8 acc (subset) | 0.879 | 0.887 | 0.876 | 0.878 |
| int8 AUC (subset) | 0.976 | 0.978 | 0.949 | 0.975 |
| best val AUC | 0.986 | 0.984 | 0.947 | 0.984 |
| threshold | 0.350 | 0.630 | 0.530 | 0.350 |
| train minutes | 9.600 | 10.600 | 16.100 | 12.100 |
| train rows | 44,751 | 46,908 | 46,908 | 46,908 |
