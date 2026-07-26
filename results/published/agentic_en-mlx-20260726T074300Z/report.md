# Run report — agentic_en (mlx)
- **run_id:** `20260726T074300Z`
- **model:** `models/nanbeige-mlx-6bit`
- **env:** python 3.12.12 · mlx 0.32.0 · mlx-lm 0.31.3 · arm64 · git c800313
- **quantization:** 6-bit, group_size=64

## Summary
- pass rate: **26/30** (86.7%) [0.70, 0.95]

| grade kind | pass / n |
|---|---|
| exact_match | 1 / 1 |
| json_schema | 2 / 2 |
| tool_call | 23 / 27 |

## Latency / memory (real-model cases only)
| metric | value |
|---|---|
| decode throughput (aggregate) | 22.95 tok/s |
| decode throughput (median) | 14.75 tok/s |
| TTFT median | 2.631 s |
| TTFT with tools / bare prompt | 2.694 / 0.663 s |
| mean generated tokens | 162.2 |
| peak allocator memory | 3782.0 MB |

## Cases
| id | kind | pass | detail | stop | tok/s | tokens |
|---|---|---|---|---|---|---|
| en-tc-weather-tokyo | tool_call | ✅ | args_subset | stop | 11.47 | 60 |
| en-tc-weather-paris-select | tool_call | ✅ | args_subset | stop | 11.76 | 74 |
| en-tc-email-alice | tool_call | ✅ | args_subset | stop | 16.5 | 198 |
| en-tc-translate-french | tool_call | ✅ | args_subset | stop | 11.01 | 65 |
| en-tc-time-tokyo | tool_call | ✅ | args_subset | stop | 11.06 | 66 |
| en-tc-calendar-date | tool_call | ✅ | args_subset | stop | 14.29 | 132 |
| en-tc-flight-sf | tool_call | ✅ | args_subset | stop | 15.34 | 161 |
| en-json-profile | json_schema | ✅ | schema_valid | stop | 21.86 | 561 |
| en-tc-weather-london | tool_call | ✅ | args_subset | stop | 12.73 | 59 |
| en-tc-weather-singapore | tool_call | ✅ | args_subset | stop | 14.96 | 75 |
| en-tc-email-bob-cc | tool_call | ✅ | args_subset | stop | 16.36 | 138 |
| en-tc-email-carol | tool_call | ❌ | no_tool_call_found | stop | 18.33 | 394 |
| en-tc-time-ny | tool_call | ✅ | args_subset | stop | 12.65 | 71 |
| en-tc-time-london | tool_call | ❌ | missing:['timezone'] | stop | 10.94 | 63 |
| en-tc-calendar-meeting | tool_call | ✅ | args_subset | stop | 15.05 | 146 |
| en-tc-calendar-deadline | tool_call | ✅ | args_subset | stop | 15.05 | 122 |
| en-tc-translate-spanish | tool_call | ✅ | args_subset | stop | 11.93 | 65 |
| en-tc-translate-german | tool_call | ✅ | args_subset | stop | 13.72 | 98 |
| en-tc-flight-london-paris | tool_call | ❌ | no_tool_call_found | stop | 18.33 | 280 |
| en-tc-flight-la | tool_call | ✅ | args_subset | stop | 14.13 | 99 |
| en-tc-search-docs | tool_call | ✅ | args_subset | stop | 11.5 | 63 |
| en-tc-search-recipe | tool_call | ❌ | missing:['query'] | stop | 12.15 | 64 |
| en-tc-choice-routing | tool_call | ✅ | args_subset | stop | 18.0 | 273 |
| en-tc-multiarg-calendar | tool_call | ✅ | args_subset | stop | 14.3 | 111 |
| en-tc-multiarg-email | tool_call | ✅ | args_subset | stop | 16.91 | 176 |
| en-tc-distractor-time | tool_call | ✅ | args_subset | stop | 17.34 | 134 |
| en-tc-distractor-search | tool_call | ✅ | args_subset | stop | 14.54 | 73 |
| en-tc-flight-date | tool_call | ✅ | args_subset | stop | 19.51 | 144 |
| en-json-config | json_schema | ✅ | schema_valid | stop | 24.73 | 678 |
| en-exact-capital | exact_match | ✅ | exact_match | stop | 20.84 | 222 |
