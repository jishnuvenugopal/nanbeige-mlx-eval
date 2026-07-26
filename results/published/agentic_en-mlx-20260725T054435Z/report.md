# Run report — agentic_en (mlx)
- **run_id:** `20260725T054435Z`
- **model:** `models/nanbeige-mlx-4bit`
- **env:** python 3.12.12 · mlx 0.32.0 · mlx-lm 0.31.3 · arm64 · git 0a0cf2c
- **quantization:** 4-bit, group_size=64

## Summary
- pass rate: **8/8** (100.0%) [0.68, 1.00]

| grade kind | pass / n |
|---|---|
| json_schema | 1 / 1 |
| tool_call | 7 / 7 |

## Latency / memory (real-model cases only)
| metric | value |
|---|---|
| decode throughput (aggregate) | 31.59 tok/s |
| decode throughput (median) | 17.06 tok/s |
| TTFT median | 2.588 s |
| TTFT with tools / bare prompt | 2.586 / 0.677 s |
| mean generated tokens | 113.8 |
| peak allocator memory | 2845.0 MB |

## Cases
| id | kind | pass | detail | stop | tok/s | tokens |
|---|---|---|---|---|---|---|
| en-tc-weather-tokyo | tool_call | ✅ | args_subset | stop | 14.8 | 58 |
| en-tc-weather-paris-select | tool_call | ✅ | args_subset | stop | 14.58 | 71 |
| en-tc-email-alice | tool_call | ✅ | args_subset | stop | 21.0 | 163 |
| en-tc-translate-french | tool_call | ✅ | args_subset | stop | 13.77 | 65 |
| en-tc-time-tokyo | tool_call | ✅ | args_subset | stop | 14.22 | 66 |
| en-tc-calendar-date | tool_call | ✅ | args_subset | stop | 19.77 | 152 |
| en-tc-flight-sf | tool_call | ✅ | args_subset | stop | 19.33 | 147 |
| en-json-profile | json_schema | ✅ | schema_valid | stop | 29.46 | 188 |
