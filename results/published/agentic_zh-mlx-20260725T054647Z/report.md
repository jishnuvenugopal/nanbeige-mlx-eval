# Run report — agentic_zh (mlx)
- **run_id:** `20260725T054647Z`
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
| decode throughput (aggregate) | 31.47 tok/s |
| decode throughput (median) | 16.57 tok/s |
| TTFT median | 2.555 s |
| TTFT with tools / bare prompt | 2.523 / 0.601 s |
| mean generated tokens | 190 |
| peak allocator memory | 2841.7 MB |

## Cases
| id | kind | pass | detail | stop | tok/s | tokens |
|---|---|---|---|---|---|---|
| zh-tc-weather-beijing | tool_call | ✅ | args_subset | stop | 14.5 | 58 |
| zh-tc-weather-paris-select | tool_call | ✅ | args_subset | stop | 14.73 | 71 |
| zh-tc-email-bob | tool_call | ✅ | args_subset | stop | 22.81 | 228 |
| zh-tc-translate-japanese | tool_call | ✅ | args_subset | stop | 14.45 | 68 |
| zh-tc-time-shanghai | tool_call | ✅ | args_subset | stop | 15.1 | 73 |
| zh-tc-calendar-date | tool_call | ✅ | args_subset | stop | 18.05 | 113 |
| zh-tc-flight-bj-sh | tool_call | ✅ | args_subset | stop | 19.84 | 134 |
| zh-json-profile | json_schema | ✅ | schema_valid | stop | 30.77 | 775 |
