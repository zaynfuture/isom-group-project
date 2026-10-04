"""Reviewed subset of user-supplied Citi MCC manual; normalized descriptions."""
SOURCE = "https://www.citibank.com/tts/solutions/commercial-cards/assets/docs/govt/Merchant-Category-Codes.pdf"
VERSION = "citi-reviewed-subset-2026-10-03"
REFERENCE = {
    "5311": {"description": "Department-store retail", "category": "Department stores", "page": 6},
    "5411": {"description": "Supermarket and grocery retail", "category": "Grocery", "page": 6},
    "5541": {"description": "Service stations, including ancillary services", "category": "Fuel", "page": 7},
    "5812": {"description": "Restaurant and eating-place merchants", "category": "Dining", "page": 8},
    "7011": {"description": "Hotel, motel and resort accommodation, otherwise unclassified", "category": "Lodging", "page": 10},
}

# Analytics coverage is separate from the five-label trained classifier.
SPENDING_REFERENCE = {**REFERENCE,
    "4111": {"description": "Local passenger transit", "category": "Public transit", "page": 4},
    "4511": {"description": "General airline merchants", "category": "Air travel", "page": 4},
    "4814": {"description": "Telephone service providers", "category": "Telecom", "page": 5},
    "4900": {"description": "Household utility providers", "category": "Utilities", "page": 5},
    "5651": {"description": "Family apparel retailers", "category": "Clothing", "page": 7},
    "5732": {"description": "Consumer electronics merchants", "category": "Electronics", "page": 8},
    "5912": {"description": "Retail pharmacy merchants", "category": "Pharmacy", "page": 8},
    "7832": {"description": "Cinema venues", "category": "Cinema", "page": 12},
    "7997": {"description": "Membership sports and recreation clubs", "category": "Sports memberships", "page": 13},
}
SPENDING_VERSION = "citi-reviewed-14-code-subset-2026-10-04"
