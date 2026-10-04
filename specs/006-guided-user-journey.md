# Guided analyst journey

## User and goal
An enterprise analyst needs to load an approved transaction extract, understand spending, optionally run predictions and screen authorized group differences without understanding model infrastructure first. The interface remains English.

## Navigation
Start here → 1 Prepare data → 2 Explore spending → 3 Model insights → 4 Compare groups. Model performance remains directly available as a supporting destination. Sidebar navigation allows non-linear access; next-step buttons support first-time users. Cohort category analysis is consolidated under Compare groups rather than a competing top-level page.

## Interaction rules
- Demo starts with USD and an explicit synthetic-data label. Display record count, card count, dates and reporting currency on every ready page.
- Home presents task cards, a one-click demo action and no evaluation tables above the main actions.
- Upload empty state lists the complete required schema, offers a template and explains how to return to demo mode.
- Preserve all 7,680 demo rows in the scrollable explorer and preserve downloads.
- Use business task labels for model selection and action buttons; keep model IDs in an expander.
- Move FX audit detail to a sidebar expander. Retain original rate protections and exports.
- FairnessLens keeps the interpretation warning visible; advanced settings and statistical details are progressively disclosed.
- No training, model weights, fairness mathematics or external publication behavior changes.

## Acceptance
Automated AppTest traverses home → data → spending → model → groups, checks default USD and all demo rows, and tests recovery from the upload empty state. Existing regression tests remain required. Verify the deployed first-use experience visually. No usability-study claims are made.

## Local evidence and gaps
Knowledge health: 54 documents / 1,529 passages. Context: 1,328 estimated tokens (cap 1,500), observation `search-d2414c4b-021b-4655-99c7-d18745266eb1`. Retrieved `passage-1f8ca2f934b7d93b26b88063755df798` is only a user-outcomes heading; other retrieved equity passages are unrelated. No relevant SpendLens UX evidence was found; design decisions are based on inspected application code and the user's requested journey, not those passages. No external sources were needed.
