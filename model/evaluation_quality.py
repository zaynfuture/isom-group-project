"""Diagnostics for shared descriptions in the legacy merchant evaluation fixture."""


def merchant_overlap(parts):
    """Report row-weighted test overlap; templates strip trailing branch numbers.

    This is a diagnostic for this synthetic generator, not universal entity resolution.
    """
    train, test = parts['train'], parts['test']
    def templates(part):
        return part.merchant_description.str.casefold().str.strip().str.replace(r'\s+\d+$', '', regex=True)
    train_templates = set(templates(train))
    return {
        'test_card_overlap_count': len(set(train.card_id) & set(test.card_id)),
        'train_template_count': len(train_templates),
        'test_template_count': templates(test).nunique(),
        'test_template_overlap_rate': float(templates(test).isin(train_templates).mean()),
        'test_exact_description_overlap_rate': float(test.merchant_description.isin(set(train.merchant_description)).mean()),
    }
