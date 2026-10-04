# SpendLens glossary

- **Legacy fixture**: original 7,680 positive USD transactions, five merchant labels; basis of currently published scores.
- **Lifestyle demo**: 13,213 synthetic purchase/refund rows, 14 categories, six simulator profiles; not representative population data.
- **Merchant classifier**: text-to-MCC-category model. Does not infer demographics. A model's label schema is fixed by training.
- **Forecast**: next-month positive purchase amount from two previous complete months, in USD; the current UI shows historical backtests, not a live future projection.
- **Validation**: data used to choose candidate models or checkpoints.
- **Test**: held-out final reporting data. Historical project test labels have been inspected before and are not a fresh external benchmark.
- **Smoke run**: small real training/evaluation execution to test software. Not evidence of business performance.
- **Publication**: uploading an explicitly approved checkpoint and evidence to zhengzhihust on Hugging Face.
- **Promotion**: separately approving a pinned checkpoint for the serving manifest; publication does not imply promotion.
- **Disparity screening**: aggregate comparisons using authorized demographic labels, not an automated discrimination verdict.
