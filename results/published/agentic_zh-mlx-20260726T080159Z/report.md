# Run report — agentic_zh (mlx)
- **run_id:** `20260726T080159Z`
- **model:** `models/nanbeige-mlx-8bit`
- **env:** python 3.12.12 · mlx 0.32.0 · mlx-lm 0.31.3 · arm64 · git c800313
- **quantization:** 8-bit, group_size=64

## Summary
- pass rate: **27/30** (90.0%) [0.74, 0.97]

| grade kind | pass / n |
|---|---|
| exact_match | 1 / 1 |
| json_schema | 2 / 2 |
| tool_call | 24 / 27 |

## Latency / memory (real-model cases only)
| metric | value |
|---|---|
| decode throughput (aggregate) | 20.33 tok/s |
| decode throughput (median) | 14.45 tok/s |
| TTFT median | 2.227 s |
| TTFT with tools / bare prompt | 2.361 / 0.614 s |
| mean generated tokens | 178.1 |
| peak allocator memory | 4739.8 MB |

## Cases
| id | kind | pass | detail | stop | tok/s | tokens |
|---|---|---|---|---|---|---|
| zh-tc-weather-beijing | tool_call | ✅ | args_subset | stop | 10.24 | 57 |
| zh-tc-weather-paris-select | tool_call | ✅ | args_subset | stop | 10.76 | 70 |
| zh-tc-email-bob | tool_call | ✅ | args_subset | stop | 14.13 | 199 |
| zh-tc-translate-japanese | tool_call | ✅ | args_subset | stop | 10.49 | 68 |
| zh-tc-time-shanghai | tool_call | ✅ | args_subset | stop | 10.74 | 71 |
| zh-tc-calendar-date | tool_call | ✅ | args_subset | stop | 12.84 | 150 |
| zh-tc-flight-bj-sh | tool_call | ✅ | args_subset | stop | 14.52 | 245 |
| zh-json-profile | json_schema | ✅ | schema_valid | stop | 17.23 | 880 |
| zh-tc-weather-shanghai | tool_call | ✅ | args_subset | stop | 9.46 | 57 |
| zh-tc-weather-guangzhou | tool_call | ✅ | args_subset | stop | 12.21 | 69 |
| zh-tc-email-lisi | tool_call | ✅ | args_subset | stop | 15.91 | 145 |
| zh-tc-email-wangwu | tool_call | ❌ | no_tool_call_found | stop | 19.16 | 361 |
| zh-tc-time-tokyo | tool_call | ✅ | args_subset | stop | 13.6 | 68 |
| zh-tc-time-newyork | tool_call | ✅ | args_subset | stop | 13.62 | 77 |
| zh-tc-calendar-meeting | tool_call | ✅ | args_subset | stop | 15.99 | 148 |
| zh-tc-calendar-deadline | tool_call | ✅ | args_subset | stop | 15.76 | 117 |
| zh-tc-translate-english | tool_call | ✅ | args_subset | stop | 14.03 | 74 |
| zh-tc-translate-korean | tool_call | ✅ | args_subset | stop | 14.3 | 93 |
| zh-tc-flight-bj-gz | tool_call | ❌ | no_tool_call_found | stop | 17.92 | 228 |
| zh-tc-flight-sh-gz | tool_call | ✅ | args_subset | stop | 14.95 | 90 |
| zh-tc-search-docs | tool_call | ✅ | args_subset | stop | 12.35 | 59 |
| zh-tc-search-recipe | tool_call | ❌ | missing:['query'] | stop | 14.38 | 79 |
| zh-tc-choice-routing | tool_call | ✅ | args_subset | stop | 18.63 | 264 |
| zh-tc-multiarg-calendar | tool_call | ✅ | args_subset | stop | 15.27 | 106 |
| zh-tc-multiarg-email | tool_call | ✅ | args_subset | stop | 19.05 | 312 |
| zh-tc-distractor-time | tool_call | ✅ | args_subset | stop | 16.17 | 141 |
| zh-tc-distractor-search | tool_call | ✅ | args_subset | stop | 14.04 | 81 |
| zh-tc-flight-date | tool_call | ✅ | args_subset | stop | 17.09 | 162 |
| zh-json-config | json_schema | ✅ | schema_valid | stop | 21.65 | 528 |
| zh-exact-capital | exact_match | ✅ | exact_match | stop | 21.58 | 344 |
