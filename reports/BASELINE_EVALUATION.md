# Baseline Forecasting Benchmark Report (Week 2)

## Chronological Partitioning Plan
To prevent temporal data leakage, records are partitioned chronologically:
- **Training Window**: 492 operating days (14,790 records)
- **Validation Window**: 105 operating days (3,150 records)
- **Holdout Test Window**: 105 operating days (3,150 records)

---

## Validation Set Baseline Performance
| Model Benchmark | MAE (Units) | RMSE | WAPE (%) | MAPE (%) | Forecast Bias | Bias (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive (Lag 1)** | 7.319 | 12.728 | 31.34% | 51.57% | 0.074 | 0.32% |
| **Seasonal-Naive (Lag 7)** | 6.405 | 11.395 | 27.43% | 47.7% | 0.167 | 0.71% |
| **Rolling 7-Day Average** | 6.367 | 11.004 | 27.27% | 41.41% | 0.163 | 0.7% |

---

## Out-of-Sample Test Set Baseline Performance
| Model Benchmark | MAE (Units) | RMSE | WAPE (%) | MAPE (%) | Forecast Bias | Bias (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive (Lag 1)** | 6.371 | 9.587 | 30.8% | 48.42% | -0.041 | -0.2% |
| **Seasonal-Naive (Lag 7)** | 5.738 | 8.414 | 27.74% | 45.54% | -0.187 | -0.9% |
| **Rolling 7-Day Average** | 5.811 | 8.479 | 28.09% | 39.99% | -0.126 | -0.61% |

### Key Analytical Takeaways
1. **Seasonal-Naive (Lag 7) outperforms Naive (Lag 1)**: Across all stores and SKUs, the 7-day seasonal persistence captures consumer shopping cycles (weekend surges in grocery and bakery) significantly better than yesterday's value alone.
2. **Rolling 7-Day Mean** provides a stable smoothing benchmark with low bias, but lags behind rapid promotional shifts.
3. Machine learning models in Week 3 will target substantially beating the Seasonal-Naive benchmark by modeling multivariate interactions (price markdowns, annual seasonality, and lead-time dynamics).
