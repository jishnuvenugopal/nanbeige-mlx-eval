# Run report — agentic_zh (mlx)
- **run_id:** `20260726T075729Z`
- **model:** `models/nanbeige-mlx-6bit`
- **env:** python 3.12.12 · mlx 0.32.0 · mlx-lm 0.31.3 · arm64 · git c800313
- **quantization:** 6-bit, group_size=64

## Summary
- pass rate: **28/30** (93.3%) [0.79, 0.98]

| grade kind | pass / n |
|---|---|
| exact_match | 1 / 1 |
| json_schema | 2 / 2 |
| tool_call | 25 / 27 |

## Latency / memory (real-model cases only)
| metric | value |
|---|---|
| decode throughput (aggregate) | 26.98 tok/s |
| decode throughput (median) | 17.19 tok/s |
| TTFT median | 2.286 s |
| TTFT with tools / bare prompt | 2.285 / 0.587 s |
| mean generated tokens | 163.4 |
| peak allocator memory | 3784.8 MB |

## Cases
| id | kind | pass | detail | stop | tok/s | tokens |
|---|---|---|---|---|---|---|
| zh-tc-weather-beijing | tool_call | ✅ | args_subset | stop | 13.96 | 57 |
| zh-tc-weather-paris-select | tool_call | ✅ | args_subset | stop | 14.24 | 70 |
| zh-tc-email-bob | tool_call | ✅ | args_subset | stop | 23.33 | 402 |
| zh-tc-translate-japanese | tool_call | ✅ | args_subset | stop | 14.06 | 68 |
| zh-tc-time-shanghai | tool_call | ✅ | args_subset | stop | 14.45 | 71 |
| zh-tc-calendar-date | tool_call | ✅ | args_subset | stop | 17.13 | 115 |
| zh-tc-flight-bj-sh | tool_call | ✅ | args_subset | stop | 18.23 | 127 |
| zh-json-profile | json_schema | ✅ | schema_valid | stop | 26.89 | 594 |
| zh-tc-weather-shanghai | tool_call | ✅ | args_subset | stop | 13.02 | 58 |
| zh-tc-weather-guangzhou | tool_call | ✅ | args_subset | stop | 14.76 | 69 |
| zh-tc-email-lisi | tool_call | ✅ | args_subset | stop | 18.4 | 144 |
| zh-tc-email-wangwu | tool_call | ❌ | no_tool_call_found | stop | 22.79 | 343 |
| zh-tc-time-tokyo | tool_call | ✅ | args_subset | stop | 15.27 | 68 |
| zh-tc-time-newyork | tool_call | ✅ | args_subset | stop | 14.67 | 68 |
| zh-tc-calendar-meeting | tool_call | ✅ | args_subset | stop | 17.25 | 119 |
| zh-tc-calendar-deadline | tool_call | ✅ | args_subset | stop | 18.56 | 124 |
| zh-tc-translate-english | tool_call | ✅ | args_subset | stop | 15.82 | 74 |
| zh-tc-translate-korean | tool_call | ✅ | args_subset | stop | 16.16 | 92 |
| zh-tc-flight-bj-gz | tool_call | ✅ | args_subset | stop | 20.66 | 204 |
| zh-tc-flight-sh-gz | tool_call | ✅ | args_subset | stop | 17.03 | 90 |
| zh-tc-search-docs | tool_call | ✅ | args_subset | stop | 12.48 | 59 |
| zh-tc-search-recipe | tool_call | ❌ | missing:['query'] | stop | 13.98 | 67 |
| zh-tc-choice-routing | tool_call | ✅ | args_subset | stop | 21.88 | 254 |
| zh-tc-multiarg-calendar | tool_call | ✅ | args_subset | stop | 17.7 | 110 |
| zh-tc-multiarg-email | tool_call | ✅ | args_subset | stop | 22.46 | 290 |
| zh-tc-distractor-time | tool_call | ✅ | args_subset | stop | 17.94 | 150 |
| zh-tc-distractor-search | tool_call | ✅ | args_subset | stop | 14.61 | 84 |
| zh-tc-flight-date | tool_call | ✅ | args_subset | stop | 18.48 | 164 |
| zh-json-config | json_schema | ✅ | schema_valid | stop | 24.94 | 291 |
| zh-exact-capital | exact_match | ✅ | exact_match | stop | 26.0 | 477 |
