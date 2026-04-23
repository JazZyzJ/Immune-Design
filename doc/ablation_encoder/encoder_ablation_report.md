# Encoder Ablation Report (Module H)

## 1. Ranking Metrics (per-protein macro-averaged, best epoch)

| Encoder | pp_AUC | pp_AP | Recall@50 | Recall@100 | Seeds |
|---------|--------|-------|-----------|------------|-------|
| E0 | 0.8526±0.0082 | 0.0411±0.0107 | 0.2393 | 0.3128 | 3 |
| E1 | 0.9715±0.0016 | 0.2715±0.0179 | 0.7054 | 0.7734 | 3 |
| E2 | 0.7629±0.0110 | 0.0178±0.0040 | 0.1040 | 0.1578 | 3 |

## 2. Convergence / Overfit

| Encoder | Epoch to Best (mean) | Best→Last AUC Drop (mean) |
|---------|---------------------|---------------------------|
| E0 | 5.3 | 0.0264 |
| E1 | 10.0 | 0.0055 |
| E2 | 3.3 | 0.0942 |

## 3. Mutation Sensitivity

| Encoder | Δz mean | Δz std | |Δz| p90 | Sign Balance | Proteins |
|---------|---------|--------|---------|--------------:|----------|
| E0 | -0.0226 | 0.1953 | 0.1857 | 0.430 | 47 |
| E1 | -0.0849 | 0.7707 | 0.3312 | 0.200 | 47 |
| E2 | -0.0075 | 0.1329 | 0.0232 | 0.484 | 47 |

## 4. Latency (batch=1, encoder+head forward)

| Encoder | L=128 | L=256 | L=512 | L=1022 |
|---------|-------|-------|-------|--------|
| E0 | 50.5ms | 80.1ms | 150.4ms | 269.5ms |
| E1 | 2.9ms | 2.9ms | 2.9ms | 3.0ms |
| E2 | 4.5ms | 4.5ms | 4.5ms | 5.7ms |

## 5. Recommendation

*To be filled after all matrix runs complete.*

