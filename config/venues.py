"""
config/venues.py
Canonical registry of all IPL venues.
Used by:
  - src/ingestion/weather_ingest.py   (lat/lon for Open-Meteo API calls)
  - src/ingestion/schedule_ingest.py  (haversine travel distance)
  - src/features/                     (venue encoding)
"""

VENUES = {
    "M Chinnaswamy Stadium":                                {"city": "Bengaluru",          "lat": 12.9791,  "lon": 77.5497,  "capacity": 35000},
    "Wankhede Stadium":                                     {"city": "Mumbai",             "lat": 18.9389,  "lon": 72.8258,  "capacity": 33108},
    "Eden Gardens":                                         {"city": "Kolkata",            "lat": 22.5645,  "lon": 88.3433,  "capacity": 66349},
    "MA Chidambaram Stadium":                               {"city": "Chennai",            "lat": 13.0628,  "lon": 80.2791,  "capacity": 50000},
    "Arun Jaitley Stadium":                                 {"city": "Delhi",              "lat": 28.6364,  "lon": 77.2195,  "capacity": 41842},
    "Rajiv Gandhi International Stadium":                   {"city": "Hyderabad",          "lat": 17.4042,  "lon": 78.5498,  "capacity": 55000},
    "Punjab Cricket Association IS Bindra Stadium":         {"city": "Mohali",             "lat": 30.6842,  "lon": 76.7154,  "capacity": 26000},
    "Sawai Mansingh Stadium":                               {"city": "Jaipur",             "lat": 26.8972,  "lon": 75.8024,  "capacity": 30000},
    "Narendra Modi Stadium":                                {"city": "Ahmedabad",          "lat": 23.0900,  "lon": 72.0847,  "capacity": 132000},
    "Brabourne Stadium":                                    {"city": "Mumbai",             "lat": 18.9322,  "lon": 72.8264,  "capacity": 20000},
    "DY Patil Stadium":                                     {"city": "Mumbai",             "lat": 19.0435,  "lon": 72.9987,  "capacity": 55000},
    "Maharashtra Cricket Association Stadium":              {"city": "Pune",               "lat": 18.6298,  "lon": 73.8015,  "capacity": 37406},
    "JSCA International Stadium Complex":                   {"city": "Ranchi",             "lat": 23.3441,  "lon": 85.3096,  "capacity": 40000},
    "Himachal Pradesh Cricket Association Stadium":         {"city": "Dharamshala",        "lat": 32.2190,  "lon": 76.3234,  "capacity": 23000},
    "Barsapara Cricket Stadium":                            {"city": "Guwahati",           "lat": 26.1433,  "lon": 91.7898,  "capacity": 40000},
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium": {"city": "Visakhapatnam",      "lat": 17.7231,  "lon": 83.2183,  "capacity": 27500},
    "Barabati Stadium":                                     {"city": "Cuttack",            "lat": 20.4686,  "lon": 85.8792,  "capacity": 45000},
    "Greenfield International Stadium":                     {"city": "Thiruvananthapuram", "lat": 8.5553,   "lon": 76.9063,  "capacity": 55000},
    "Holkar Cricket Stadium":                               {"city": "Indore",             "lat": 22.7215,  "lon": 75.8578,  "capacity": 30000},
    "Ekana Cricket Stadium":                                {"city": "Lucknow",            "lat": 26.8575,  "lon": 80.9346,  "capacity": 50000},
    # South Africa 2009
    "Newlands":                                             {"city": "Cape Town",          "lat": -33.9258, "lon": 18.4232,  "capacity": 25000},
    "St George's Park":                                     {"city": "Port Elizabeth",     "lat": -33.9608, "lon": 25.6022,  "capacity": 19000},
    "Kingsmead":                                            {"city": "Durban",             "lat": -29.8579, "lon": 31.0292,  "capacity": 25000},
    "SuperSport Park":                                      {"city": "Centurion",          "lat": -25.7547, "lon": 28.2267,  "capacity": 22000},
    "New Wanderers Stadium":                                {"city": "Johannesburg",       "lat": -26.1446, "lon": 28.0566,  "capacity": 34000},
    "De Beers Diamond Oval":                                {"city": "Kimberley",          "lat": -28.7377, "lon": 24.7478,  "capacity": 8000},
    "Buffalo Park":                                         {"city": "East London",        "lat": -32.9594, "lon": 27.9034,  "capacity": 10000},
    # UAE 2014, 2020, 2021
    "Dubai International Cricket Stadium":                  {"city": "Dubai",              "lat": 25.0359,  "lon": 55.2466,  "capacity": 25000},
    "Sheikh Zayed Stadium":                                 {"city": "Abu Dhabi",          "lat": 24.3886,  "lon": 54.5195,  "capacity": 20000},
    "Sharjah Cricket Stadium":                              {"city": "Sharjah",            "lat": 25.3396,  "lon": 55.3839,  "capacity": 16000},
}

VENUE_ALIASES = {
    "Punjab Cricket Association Stadium, Mohali":                              "Punjab Cricket Association IS Bindra Stadium",
    "PCA Stadium":                                                             "Punjab Cricket Association IS Bindra Stadium",
    "Feroz Shah Kotla":                                                        "Arun Jaitley Stadium",
    "Sardar Patel Stadium":                                                    "Narendra Modi Stadium",
    "Motera Stadium":                                                          "Narendra Modi Stadium",
    "Subrata Roy Sahara Stadium":                                              "Maharashtra Cricket Association Stadium",
    "Dr DY Patil Sports Academy":                                              "DY Patil Stadium",
    "Deccan Chronicle Holdings Limited Hyderabad Cricket Association Ground":  "Rajiv Gandhi International Stadium",
}

HOME_GROUNDS = {
    # RCB — Cricsheet uses both spellings across seasons
    "Royal Challengers Bangalore":   "M Chinnaswamy Stadium",
    "Royal Challengers Bengaluru":   "M Chinnaswamy Stadium",
    # MI
    "Mumbai Indians":                "Wankhede Stadium",
    # KKR
    "Kolkata Knight Riders":         "Eden Gardens",
    # CSK
    "Chennai Super Kings":           "MA Chidambaram Stadium",
    # Delhi — two franchise names
    "Delhi Daredevils":              "Arun Jaitley Stadium",
    "Delhi Capitals":                "Arun Jaitley Stadium",
    # SRH / Deccan
    "Sunrisers Hyderabad":           "Rajiv Gandhi International Stadium",
    "Deccan Chargers":               "Rajiv Gandhi International Stadium",
    # Punjab — three franchise names used in Cricsheet
    "Kings XI Punjab":               "Punjab Cricket Association IS Bindra Stadium",
    "Punjab Kings":                  "Punjab Cricket Association IS Bindra Stadium",
    "Punjab Cricket Association":    "Punjab Cricket Association IS Bindra Stadium",
    # RR
    "Rajasthan Royals":              "Sawai Mansingh Stadium",
    # GT
    "Gujarat Titans":                "Narendra Modi Stadium",
    # LSG
    "Lucknow Super Giants":          "Ekana Cricket Stadium",
    # Defunct franchises
    "Kochi Tuskers Kerala":          "Greenfield International Stadium",
    "Pune Warriors":                 "Maharashtra Cricket Association Stadium",
    "Rising Pune Supergiant":        "Maharashtra Cricket Association Stadium",
    "Rising Pune Supergiants":       "Maharashtra Cricket Association Stadium",
    "Gujarat Lions":                 "Narendra Modi Stadium",
    # Additional Cricsheet variant spellings seen in real data
    "Royal Challengers":             "M Chinnaswamy Stadium",
    "Delhi":                         "Arun Jaitley Stadium",
    "Hyderabad":                     "Rajiv Gandhi International Stadium",
}


def resolve_venue(raw_name: str) -> str:
    """Normalise a raw Cricsheet venue string to a canonical key in VENUES."""
    if raw_name in VENUES:
        return raw_name
    if raw_name in VENUE_ALIASES:
        return VENUE_ALIASES[raw_name]
    for canonical in VENUES:
        if canonical.lower() in raw_name.lower() or raw_name.lower() in canonical.lower():
            return canonical
    return raw_name  # unknown venue — returned as-is, flagged downstream