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
