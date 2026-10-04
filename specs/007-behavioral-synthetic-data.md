# Behavioral synthetic demo v2

## Scope
The app uses model/data/scenarios.py, version lifestyle-demo-v2-seed5240. The original demo_transactions training fixture remains available unchanged in transaction values, merchant labels and temporal patterns. Existing Hub model weights and published evaluation results are not regenerated or attributed to v2.

## Design assumptions, not industry estimates
120 anonymous cards cover January–August 2025, all USD. Six scenario groups have 20 cards each: transit/city dining, home/essentials, driving/errands, travel/leisure, digital shopping, active/local. Profiles are simulator metadata, not inferred traits or demographic groups. Activity tiers vary within each profile; amounts reflect partial card-wallet activity, not household budgets. Daily timing is UTC simulation, not a geolocated local-time model.

14 reviewed MCC categories support distinct ticket-size distributions. Grocery/dining are frequent; travel and electronics are less frequent with larger tickets. Transit favors weekdays; dining, cinema and shopping favor weekends. Phone/utility/sports bills recur monthly for subscribed cards. Other purchases vary in date and amount. Balanced profile coverage does not imply equal category counts or representative population sampling.

Refund probabilities are explicit per-category assumptions (1–18%), not observed rates. Generated data has 13,213 records: 12,747 purchases, 294 full refunds, 172 partial refunds. Each refund has one original purchase, the same card/MCC/currency, a 2–21 day delay and an amount no greater than its purchase. Some refunds cross months. Post-August refunds are censored. No chargeback lifecycle, settlements, FX fees or split refunds are modeled.

## User experience and model boundaries
Data Explorer explains assumptions and exposes refund links, types and reasons. Spending Analytics shows gross purchases, refunds, net totals, lifestyle aggregates and refund rows. The 14-code reference also applies to uploaded CSVs. It is still a subset, not the full MCC taxonomy. The pretrained merchant classifier supports its original five labels only. Forecast thresholds and weights are unchanged; v2 forecasts are explicitly unvalidated. FairnessLens still uses independently generated or authorized demographic labels and positive-purchase metrics; no demographic stereotypes are encoded in scenarios.

## Verification
Tests cover deterministic generation, all categories, equal profile card counts, variable activity, unique IDs, positive/nonzero amounts, recurring bills, refund linkage/timing/bounds, cross-month examples, mapping round trips and retention of the legacy fixture. User-journey tests check all current demo records are browsable without a hardcoded old count.

## Evidence
Local knowledge health: 54 documents / 1,529 passages. Context capped at 1,500 tokens (1,470 returned), observation search-5bd49528-bfa2-4df8-af6b-6988bc9ce7d7. Retrieved passage-02ae52c4620ec6f31ffa3ec0edf38006 concerns options, not spending simulation; no relevant empirical distribution evidence was found. No local evidence is used to justify rates.

MCC definitions were checked against the user-supplied [Citi MCC manual](https://www.citibank.com/tts/solutions/commercial-cards/assets/docs/govt/Merchant-Category-Codes.pdf), printed pages 4, 5, 7, 8, 12 and 13 for newly added codes. Labels in the app are normalized category summaries. Merchant names are synthetic. The manual supports code meanings, not simulated spending frequencies.
