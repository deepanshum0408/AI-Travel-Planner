import os
import csv
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool

CITY_TO_IATA = {}

# Construct path to airports.dat relative to project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
csv_path = os.path.join(project_root, "data", "airports.dat")

with open(csv_path, encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if len(row) < 5:
            continue
        city = row[2].strip().lower()
        iata = row[4].strip().upper()
        # Only add valid IATA codes (3 letters, not \N or empty)
        if iata and len(iata) == 3 and iata != '\\N' and iata != 'N/A':
            # If city already exists, prefer the one with a valid IATA code
            if city not in CITY_TO_IATA or CITY_TO_IATA[city] == '\\N' or CITY_TO_IATA[city] == 'N/A':
                CITY_TO_IATA[city] = iata

print(f"[AirportLookup] Loaded {len(CITY_TO_IATA)} city-to-IATA mappings")

class AirportLookupInput(BaseModel):
    q: str = Field(description='City or country name to find airport code for.')

@tool(args_schema=AirportLookupInput)
def airport_code_lookup(q: str):
    '''
    Find the IATA airport code for a given city using the local OpenFlights CSV.
    Supports fuzzy matching for common city name variations.
    '''
    city = q.strip().lower()
    print(f"[AirportLookup] Looking up IATA for: {city}")
    
    # Direct lookup
    if city in CITY_TO_IATA:
        result = CITY_TO_IATA[city]
        print(f"[AirportLookup] Found direct match: {city} -> {result}")
        return result
    
    # Common city name variations and aliases
    city_aliases = {
        'delhi': 'new delhi',
        'new delhi': 'new delhi',
        'mumbai': 'mumbai',
        'bombay': 'mumbai',
        'bangalore': 'bangalore',
        'bengaluru': 'bangalore',
        'calcutta': 'kolkata',
        'kolkata': 'kolkata',
        'madras': 'chennai',
        'chennai': 'chennai',
        'hyderabad': 'hyderabad',
        'pune': 'pune',
        'ahmedabad': 'ahmedabad',
        'jaipur': 'jaipur',
        'lucknow': 'lucknow',
        'kolkata': 'kolkata',
        'goa': 'goa',
        'kochi': 'kochi',
        'cochin': 'kochi',
        'surat': 'surat',
        'kanpur': 'kanpur',
        'nagpur': 'nagpur',
        'indore': 'indore',
        'thane': 'mumbai',  # Thane is near Mumbai
        'gurgaon': 'new delhi',  # Gurgaon is near Delhi
        'noida': 'new delhi',  # Noida is near Delhi
    }
    
    # Check aliases
    normalized_city = city_aliases.get(city, city)
    if normalized_city in CITY_TO_IATA:
        result = CITY_TO_IATA[normalized_city]
        print(f"[AirportLookup] Found via alias: {city} -> {normalized_city} -> {result}")
        return result
    
    # Fuzzy matching: check if city name is contained in any CSV city name
    for csv_city, iata_code in CITY_TO_IATA.items():
        if city in csv_city or csv_city in city:
            print(f"[AirportLookup] Found fuzzy match: {city} -> {csv_city} -> {iata_code}")
            return iata_code
    
    # If still not found, try removing common prefixes/suffixes
    city_variations = [
        city,
        city.replace('new ', ''),
        city.replace('old ', ''),
        'new ' + city,
    ]
    
    for variation in city_variations:
        if variation in CITY_TO_IATA:
            result = CITY_TO_IATA[variation]
            print(f"[AirportLookup] Found via variation: {city} -> {variation} -> {result}")
            return result
    
    print(f"[AirportLookup] No match found for: {city}")
    return "N/A"