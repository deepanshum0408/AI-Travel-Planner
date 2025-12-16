# pylint: disable = http-used,print-used,no-self-use

import datetime
import operator
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

from agents.tools.flights_finder import flights_finder
from agents.tools.hotels_finder import hotels_finder
from agents.tools.airport_lookup import airport_code_lookup

_ = load_dotenv()

CURRENT_YEAR = datetime.datetime.now().year


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


TOOLS_SYSTEM_PROMPT = f"""You are a smart travel agency. Use the tools to look up information.
    You are allowed to make multiple calls (either together or in sequence).
    Only look up information when you are sure of what you want.
    The current year is {CURRENT_YEAR}.
    If you need to look up some information before asking a follow up question, you are allowed to do that!
    I want to have in your output links to hotels websites and flights websites (if possible).
    I want to have as well the logo of the hotel and the logo of the airline company (if possible).
    In your output always include the price of the flight and the price of the hotel and the currency as well (if possible).
    for example for hotels-
    Rate: $581 per night
    Total: $3,488
    """

TOOLS = [flights_finder, hotels_finder, airport_code_lookup]

EMAILS_SYSTEM_PROMPT = """Your task is to convert structured markdown-like text into a valid HTML email body.

- Do not include a ```html preamble in your response.
- The output should be in proper HTML format, ready to be used as the body of an email.
Here is an example:
<example>
Input:

I want to travel to New York from Madrid from October 1-7. Find me flights and 4-star hotels.

Expected Output:

<!DOCTYPE html>
<html>
<head>
    <title>Flight and Hotel Options</title>
</head>
<body>
    <h2>Flights from Madrid to New York</h2>
    <ol>
        <li>
            <strong>American Airlines</strong><br>
            <strong>Departure:</strong> Adolfo Suárez Madrid–Barajas Airport (MAD) at 10:25 AM<br>
            <strong>Arrival:</strong> John F. Kennedy International Airport (JFK) at 12:25 PM<br>
            <strong>Duration:</strong> 8 hours<br>
            <strong>Aircraft:</strong> Boeing 777<br>
            <strong>Class:</strong> Economy<br>
            <strong>Price:</strong> $702<br>
            <img src="https://www.gstatic.com/flights/airline_logos/70px/AA.png" alt="American Airlines"><br>
            <a href="https://www.google.com/flights">Book on Google Flights</a>
        </li>
        <li>
            <strong>Iberia</strong><br>
            <strong>Departure:</strong> Adolfo Suárez Madrid–Barajas Airport (MAD) at 12:25 PM<br>
            <strong>Arrival:</strong> John F. Kennedy International Airport (JFK) at 2:40 PM<br>
            <strong>Duration:</strong> 8 hours 15 minutes<br>
            <strong>Aircraft:</strong> Airbus A330<br>
            <strong>Class:</strong> Economy<br>
            <strong>Price:</strong> $702<br>
            <img src="https://www.gstatic.com/flights/airline_logos/70px/IB.png" alt="Iberia"><br>
            <a href="https://www.google.com/flights">Book on Google Flights</a>
        </li>
        <li>
            <strong>Delta Airlines</strong><br>
            <strong>Departure:</strong> Adolfo Suárez Madrid–Barajas Airport (MAD) at 10:00 AM<br>
            <strong>Arrival:</strong> John F. Kennedy International Airport (JFK) at 12:30 PM<br>
            <strong>Duration:</strong> 8 hours 30 minutes<br>
            <strong>Aircraft:</strong> Boeing 767<br>
            <strong>Class:</strong> Economy<br>
            <strong>Price:</strong> $738<br>
            <img src="https://www.gstatic.com/flights/airline_logos/70px/DL.png" alt="Delta Airlines"><br>
            <a href="https://www.google.com/flights">Book on Google Flights</a>
        </li>
    </ol>

    <h2>4-Star Hotels in New York</h2>
    <ol>
        <li>
            <strong>NobleDen Hotel</strong><br>
            <strong>Description:</strong> Modern, polished hotel offering sleek rooms, some with city-view balconies, plus free Wi-Fi.<br>
            <strong>Location:</strong> Near Washington Square Park, Grand St, and JFK Airport.<br>
            <strong>Rate per Night:</strong> $537<br>
            <strong>Total Rate:</strong> $3,223<br>
            <strong>Rating:</strong> 4.8/5 (656 reviews)<br>
            <strong>Amenities:</strong> Free Wi-Fi, Parking, Air conditioning, Restaurant, Accessible, Business centre, Child-friendly, Smoke-free property<br>
            <img src="https://lh5.googleusercontent.com/p/AF1QipNDUrPJwBhc9ysDhc8LA822H1ZzapAVa-WDJ2d6=s287-w287-h192-n-k-no-v1" alt="NobleDen Hotel"><br>
            <a href="http://www.nobleden.com/">Visit Website</a>
        </li>
        <!-- More hotel entries here -->
    </ol>
</body>
</html>

</example>


"""


class Agent:

    def __init__(self):
        self._tools = {t.name: t for t in TOOLS}
        from langchain_groq import ChatGroq
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables. Please add it to your .env file.")
        self._tools_llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=groq_api_key,
            temperature=0.3
        ).bind_tools(TOOLS)

        builder = StateGraph(AgentState)
        builder.add_node('call_tools_llm', self.call_tools_llm)
        builder.add_node('invoke_tools', self.invoke_tools)
        builder.add_node('email_sender', self.email_sender)
        builder.set_entry_point('call_tools_llm')
        # Only call tools once, then go to email_sender or END
        builder.add_conditional_edges('call_tools_llm', Agent.exists_action, {'more_tools': 'invoke_tools', 'email_sender': 'email_sender'})
        builder.add_edge('invoke_tools', 'email_sender')
        builder.add_edge('email_sender', END)
        memory = MemorySaver()
        self.graph = builder.compile(checkpointer=memory, interrupt_before=['email_sender'])
        print(self.graph.get_graph().draw_mermaid())

    def format_travel_itinerary(self, flights_result, hotels_result):
        # Format flights
        flights_info = ""
        # Handle list, dict, or string
        if isinstance(flights_result, list) and flights_result:
            flights_info += "\n✈️ FLIGHTS\n\n"
            for idx, flight_option in enumerate(flights_result, start=1):
                flights_info += f"<div style=\"font-weight:700;font-size:1.15em;\">Option {idx}:</div>\n"
                # flight_price = flight_option.get('price', 'N/A')
                # flight_currency = flight_option.get('currency', 'USD')
                # flights_info += f"- Price: {flight_price} {flight_currency}\n"

                for f in flight_option.get('flights', []):
                    departure_airport = f.get('departure_airport', {})
                    arrival_airport = f.get('arrival_airport', {})
                    airline = f.get('airline', 'Unknown Airline')
                    airline_logo_url = f.get('airline_logo', '')
                    flight_number = f.get('flight_number', '')
                    departure_time = departure_airport.get('time', '').split(' ')[1] if departure_airport.get('time') else ''
                    arrival_time = arrival_airport.get('time', '').split(' ')[1] if arrival_airport.get('time') else ''
                    outbound_date = departure_airport.get('time', '').split(' ')[0] if departure_airport.get('time') else ''
                    
                    flights_info += f"  - {airline} {flight_number} from {departure_airport.get('name', '')} ({departure_airport.get('id', '')}) at {departure_time} to {arrival_airport.get('name', '')} ({arrival_airport.get('id', '')}) at {arrival_time} on {outbound_date}\n"
                    if airline_logo_url:
                        flights_info += f"    <img src=\"{airline_logo_url}\" alt=\"{airline}\" width=\"70\" height=\"70\"><br>\n"

                # Link for the entire flight option
                if 'google_flights_url' in flight_option:
                    flights_info += f"  [Book on Google Flights]({flight_option['google_flights_url']})\n"
                elif 'link' in flight_option:
                    flights_info += f"  [Book]({flight_option['link']})\n"

        elif isinstance(flights_result, dict):
            flights_info += "\n✈️ FLIGHTS\n\n"
            # Assuming a single flight result directly has the structure of a flight_option, with a 'flights' list
            flight_price = flights_result.get('price', 'N/A')
            flight_currency = flights_result.get('currency', 'USD')
            flights_info += f"- Price: {flight_price} {flight_currency}\n"

            for flight_leg in flights_result.get('flights', []):
                departure_airport = flight_leg.get('departure_airport', {})
                arrival_airport = flight_leg.get('arrival_airport', {})
                airline = flight_leg.get('airline', 'Unknown Airline')
                airline_logo_url = flight_leg.get('airline_logo', '')
                flight_number = flight_leg.get('flight_number', '')
                departure_time = departure_airport.get('time', '').split(' ')[1] if departure_airport.get('time') else ''
                arrival_time = arrival_airport.get('time', '').split(' ')[1] if arrival_airport.get('time') else ''
                outbound_date = departure_airport.get('time', '').split(' ')[0] if departure_airport.get('time') else ''

                flights_info += f"  - {airline} {flight_number} from {departure_airport.get('name', '')} ({departure_airport.get('id', '')}) at {departure_time} to {arrival_airport.get('name', '')} ({arrival_airport.get('id', '')}) at {arrival_time} on {outbound_date}\n"
                if airline_logo_url:
                    flights_info += f"    <img src=\"{airline_logo_url}\" alt=\"{airline}\" width=\"70\" height=\"70\"><br>\n"
            
            if 'google_flights_url' in flights_result:
                flights_info += f"  [Book on Google Flights]({flights_result['google_flights_url']})\n"
            elif 'link' in flights_result:
                flights_info += f"  [Book]({flights_result['link']})\n"
        elif isinstance(flights_result, str):
            flights_info += f"Flights: {flights_result}\n"
        else:
            flights_info += "No flights found.\n"

        # Format hotels
        hotels_info = ""
        if isinstance(hotels_result, list) and hotels_result:
            hotels_info += "\n🏨 HOTELS\n\n"
            for idx, h in enumerate(hotels_result, start=1):
                hotels_info += f"\n<div style=\"font-weight:700;font-size:1.15em;\">Option {idx}:</div>\n"
                hotels_info += f"Hotel: {h.get('name', 'Unknown Hotel')}\n"
                hotels_info += f"Description: {h.get('description', '')}\n"
                hotels_info += f"Class: {h.get('hotel_class', '')}\n"
                hotels_info += f"Rating: {h.get('overall_rating', 'N/A')}/5 ({h.get('reviews', 'N/A')} reviews)\n"
                hotels_info += f"Check-in: {h.get('check_in_time', '')}, Check-out: {h.get('check_out_time', '')}\n"
                if 'rate_per_night' in h and isinstance(h['rate_per_night'], dict):
                    hotels_info += f"Rate per night: {h['rate_per_night'].get('extracted_lowest', h['rate_per_night'].get('lowest', 'N/A'))}\n"
                elif 'rate_per_night' in h:
                    hotels_info += f"Rate per night: {h['rate_per_night']}\n"
                if 'total_rate' in h and isinstance(h['total_rate'], dict):
                    hotels_info += f"Total rate: {h['total_rate'].get('extracted_lowest', h['total_rate'].get('lowest', 'N/A'))}\n"
                elif 'total_rate' in h:
                    hotels_info += f"Total rate: {h['total_rate']}\n"
                if 'amenities' in h:
                    hotels_info += f"Amenities: {', '.join(h['amenities'])}\n"
                if 'nearby_places' in h:
                    hotels_info += "Nearby places:\n"
                    for place in h['nearby_places']:
                        hotels_info += f"  - {place.get('name', '')}: "
                        if 'transportations' in place:
                            transports = [f"{t['type']} ({t['duration']})" for t in place['transportations']]
                            hotels_info += ", ".join(transports)
                        hotels_info += "\n"
                hotel_link = h.get('link', '')
                if hotel_link:
                    hotels_info += f"Website: <a href=\"{hotel_link}\">{hotel_link}</a>\n"
        elif isinstance(hotels_result, dict):
            hotels_info += "\n🏨 HOTELS\n\n"
            h = hotels_result
            hotels_info += f"\nHotel: {h.get('name', 'Unknown Hotel')}\n"
            hotels_info += f"Description: {h.get('description', '')}\n"
            hotels_info += f"Class: {h.get('hotel_class', '')}\n"
            hotels_info += f"Rating: {h.get('overall_rating', 'N/A')}/5 ({h.get('reviews', 'N/A')} reviews)\n"
            hotels_info += f"Check-in: {h.get('check_in_time', '')}, Check-out: {h.get('check_out_time', '')}\n"
            if 'rate_per_night' in h and isinstance(h['rate_per_night'], dict):
                hotels_info += f"Rate per night: {h['rate_per_night'].get('extracted_lowest', h['rate_per_night'].get('lowest', 'N/A'))}\n"
            elif 'rate_per_night' in h:
                hotels_info += f"Rate per night: {h['rate_per_night']}\n"
            if 'total_rate' in h and isinstance(h['total_rate'], dict):
                hotels_info += f"Total rate: {h['total_rate'].get('extracted_lowest', h['total_rate'].get('lowest', 'N/A'))}\n"
            elif 'total_rate' in h:
                hotels_info += f"Total rate: {h['total_rate']}\n"
            if 'amenities' in h:
                hotels_info += f"Amenities: {', '.join(h['amenities'])}\n"
            if 'nearby_places' in h:
                hotels_info += "Nearby places:\n"
                for place in h['nearby_places']:
                    hotels_info += f"  - {place.get('name', '')}: "
                    if 'transportations' in place:
                        transports = [f"{t['type']} ({t['duration']})" for t in place['transportations']]
                        hotels_info += ", ".join(transports)
                    hotels_info += "\n"
            hotel_link = h.get('link', '')
            if hotel_link:
                hotels_info += f"Website: <a href=\"{hotel_link}\">{hotel_link}</a>\n"
        elif isinstance(hotels_result, str):
            hotels_info += f"Hotels: {hotels_result}\n"
        else:
            hotels_info += "No hotels found.\n"

        return flights_info + "\n" + hotels_info

    def create_daily_itinerary(self, departure_city, arrival_city, check_in_date, check_out_date, hotel_info=None):
        """
        Create a detailed daily itinerary with time-wise planning for each day of the trip.
        """
        try:
            from datetime import datetime, timedelta
            
            # Parse dates
            if isinstance(check_in_date, str):
                check_in = datetime.strptime(check_in_date, "%Y-%m-%d")
            else:
                check_in = check_in_date
                
            if isinstance(check_out_date, str):
                check_out = datetime.strptime(check_out_date, "%Y-%m-%d")
            else:
                check_out = check_out_date
            
            # Calculate number of days
            num_days = (check_out - check_in).days
            
            itinerary = f"\n🗓️ **DAILY ITINERARY FOR {arrival_city.upper()}**\n"
            itinerary += f"📅 Trip Duration: {num_days} days ({check_in.strftime('%B %d, %Y')} - {check_out.strftime('%B %d, %Y')})\n\n"
            
            # Generate itinerary for each day
            for day in range(num_days):
                current_date = check_in + timedelta(days=day)
                day_number = day + 1
                
                itinerary += f"## 📅 **DAY {day_number} - {current_date.strftime('%A, %B %d, %Y')}**\n\n"
                
                if day == 0:  # Arrival day
                    itinerary += self._get_arrival_day_schedule(arrival_city, current_date)
                elif day == num_days - 1:  # Departure day
                    itinerary += self._get_departure_day_schedule(departure_city, current_date)
                else:  # Full days
                    itinerary += self._get_full_day_schedule(arrival_city, current_date, day_number)
                
                itinerary += "\n---\n\n"
            
            return itinerary
            
        except Exception as e:
            print(f"Error creating itinerary: {e}")
            return f"\n🗓️ **Daily Itinerary for {arrival_city}**\n\nUnable to generate detailed itinerary due to date parsing error."

    def _get_arrival_day_schedule(self, city, date):
        """Generate schedule for arrival day"""
        return f"""**🌅 MORNING (8:00 AM - 12:00 PM)**
- 8:00 AM: Arrive at airport
- 9:00 AM: Immigration & baggage claim
- 10:00 AM: Airport transfer to hotel
- 11:00 AM: Hotel check-in and freshen up

**🌞 AFTERNOON (12:00 PM - 6:00 PM)**
- 12:00 PM: Lunch at local restaurant
- 2:00 PM: Explore {city} city center
- 4:00 PM: Visit local market or shopping area
- 5:00 PM: Coffee break at café

**🌙 EVENING (6:00 PM - 10:00 PM)**
- 6:00 PM: Return to hotel
- 7:00 PM: Dinner at hotel or nearby restaurant
- 9:00 PM: Relax and prepare for next day
- 10:00 PM: Early rest for tomorrow's adventures"""

    def _get_departure_day_schedule(self, departure_city, date):
        """Generate schedule for departure day"""
        return f"""**🌅 MORNING (8:00 AM - 12:00 PM)**
- 8:00 AM: Hotel check-out
- 9:00 AM: Final shopping or last-minute sightseeing
- 10:00 AM: Return to hotel for luggage
- 11:00 AM: Airport transfer

**🌞 AFTERNOON (12:00 PM - 6:00 PM)**
- 12:00 PM: Arrive at airport
- 1:00 PM: Check-in and security
- 2:00 PM: Duty-free shopping or airport lounge
- 3:00 PM: Boarding for flight to {departure_city}

**🌙 EVENING (6:00 PM - 10:00 PM)**
- 6:00 PM: In-flight meal and entertainment
- 8:00 PM: Rest on flight
- 10:00 PM: Arrival at {departure_city}"""

    def _get_full_day_schedule(self, city, date, day_number):
        """Generate schedule for full days"""
        activities = {
            1: ["Visit main historical sites", "Explore local museums", "City walking tour"],
            2: ["Day trip to nearby attractions", "Local food tour", "Cultural experiences"],
            3: ["Nature and outdoor activities", "Shopping districts", "Local entertainment"],
            4: ["Hidden gems and local spots", "Relaxation activities", "Evening entertainment"],
            5: ["Adventure activities", "Photography spots", "Local festivals or events"]
        }
        
        day_activities = activities.get(day_number, ["Explore local attractions", "Visit recommended spots", "Enjoy local culture"])
        
        return f"""**🌅 MORNING (8:00 AM - 12:00 PM)**
- 8:00 AM: Hotel breakfast
- 9:00 AM: {day_activities[0]}
- 11:00 AM: Coffee break and rest

**🌞 AFTERNOON (12:00 PM - 6:00 PM)**
- 12:00 PM: Local lunch
- 2:00 PM: {day_activities[1]}
- 4:00 PM: {day_activities[2]}
- 5:30 PM: Return to hotel for rest

**🌙 EVENING (6:00 PM - 10:00 PM)**
- 6:00 PM: Hotel rest and freshen up
- 7:30 PM: Dinner at recommended restaurant
- 9:00 PM: Evening stroll or local entertainment
- 10:00 PM: Return to hotel"""

    @staticmethod
    def exists_action(state: AgentState):
        result = state['messages'][-1]
        print("exists_action tool_calls:", getattr(result, "tool_calls", None))
        if len(result.tool_calls) == 0:
            return 'email_sender'
        return 'more_tools'

    def email_sender(self, state: AgentState):
        print('Sending email')
        email_llm = ChatOpenAI(model='gpt-4o', temperature=0.1)  # Instantiate another LLM
        email_message = [SystemMessage(content=EMAILS_SYSTEM_PROMPT), HumanMessage(content=state['messages'][-1].content)]
        email_response = email_llm.invoke(email_message)
        print('Email content:', email_response.content)

        message = Mail(from_email=os.environ['FROM_EMAIL'], to_emails=os.environ['TO_EMAIL'], subject=os.environ['EMAIL_SUBJECT'],
                       html_content=email_response.content)
        try:
            sg = SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))
            response = sg.send(message)
            print(response.status_code)
            print(response.body)
            print(response.headers)
        except Exception as e:
            print(str(e))

    def call_tools_llm(self, state: AgentState):
        messages = state['messages']
        messages = [SystemMessage(content=TOOLS_SYSTEM_PROMPT)] + messages
        # Groq works with message objects directly
        message = self._tools_llm.invoke(messages)
        return {'messages': [message]}

    def invoke_tools(self, state: AgentState):
        # Extract details from user query for tool inputs
        user_message = state['messages'][0].content.lower()
        # Improved extraction logic for departure, arrival cities and dates
        import re
        from datetime import datetime, timedelta
        today = datetime.now()

        # Extract departure and arrival cities - improved regex patterns
        # Pattern 1: "from X to Y" - stop before dates or other keywords
        from_to_match = re.search(r'from\s+([a-zA-Z\s]+?)\s+to\s+([a-zA-Z\s]+?)(?:\s+from\s+\d|\s+on\s+\d|\s+find|\s*$)', user_message)
        if not from_to_match:
            # Pattern 2: "X to Y" (without "from") - stop before dates or other keywords
            from_to_match = re.search(r'([a-zA-Z\s]{2,}?)\s+to\s+([a-zA-Z\s]{2,}?)(?:\s+from\s+\d|\s+on\s+\d|\s+find|\s*$)', user_message)
        
        if from_to_match:
            departure_city = from_to_match.group(1).strip()
            arrival_city = from_to_match.group(2).strip()
            # Clean up: remove extra spaces, common typos and variations
            departure_city = re.sub(r'\s+', ' ', departure_city).strip()
            arrival_city = re.sub(r'\s+', ' ', arrival_city).strip()
            arrival_city = arrival_city.replace('mardrid', 'madrid').replace('mardid', 'madrid')
            
            # Remove common words that might have been captured
            departure_city = re.sub(r'\b(plan|trip|want|need|going|travel)\b', '', departure_city, flags=re.IGNORECASE).strip()
            arrival_city = re.sub(r'\b(plan|trip|want|need|going|travel)\b', '', arrival_city, flags=re.IGNORECASE).strip()
            
            print(f"Extracted cities - Departure: '{departure_city}', Arrival: '{arrival_city}'")
        else:
            print(f"WARNING: Could not extract cities from query: '{user_message}'")
            print("Please use format: 'from [city] to [city]' or '[city] to [city]'")
            error_msg = "❌ Could not identify departure and arrival cities. Please use format: 'from [city] to [city]' or '[city] to [city]'"
            return {'messages': [ToolMessage(tool_call_id='flights_finder', name='flights_finder', content=error_msg)]}

        # Extract dates
        dates_match = re.search(r'from\s*(\d{1,2}(?:st|nd|rd|th)?)\s*([a-zA-Z]+)\s*to\s*(\d{1,2}(?:st|nd|rd|th)?)\s*([a-zA-Z]+)\s*(\d{4})', user_message)
        if dates_match:
            day1, month1, day2, month2, year = dates_match.groups()
            
            # Standardize month abbreviations (e.g., 'oct' to 'Oct')
            month1_abbr = month1.title()[:3] # 'oct' -> 'Oct'
            month2_abbr = month2.title()[:3] # 'nov' -> 'Nov'

            outbound_date_str = f"{day1} {month1_abbr} {year}"
            return_date_str = f"{day2} {month2_abbr} {year}"
            
            print(f'Extracted date parts: day1={day1}, month1={month1}, day2={day2}, month2={month2}, year={year}')
            print(f'Constructed outbound_date_str for parsing: {outbound_date_str}')
            print(f'Constructed return_date_str for parsing: {return_date_str}')

            # Convert to datetime objects
            try:
                outbound_date = datetime.strptime(outbound_date_str.replace("st", "").replace("nd", "").replace("rd", "").replace("th", ""), '%d %b %Y')
                return_date = datetime.strptime(return_date_str.replace("st", "").replace("nd", "").replace("rd", "").replace("th", ""), '%d %b %Y')
            except ValueError as e:
                print(f'ValueError during date parsing: {e}')
                # Fallback if specific date parsing fails
                outbound_date = today + timedelta(days=1)
                return_date = outbound_date + timedelta(days=3) # Default to 3 days if date parsing fails
                outbound_date_str = outbound_date.strftime('%Y-%m-%d') # Assign default here
                return_date_str = return_date.strftime('%Y-%m-%d') # Assign default here
        else:
            print('Date regex did not match. Using default dates.')
            # Fallback to current date + offset if no dates are found
            outbound_date = today + timedelta(days=1)
            return_date = outbound_date + timedelta(days=3)
            outbound_date_str = outbound_date.strftime('%Y-%m-%d') # Assign default here
            return_date_str = return_date.strftime('%Y-%m-%d') # Assign default here

        check_in_str = outbound_date.strftime('%Y-%m-%d')
        check_out_str = return_date.strftime('%Y-%m-%d')

        # Check if cities were extracted
        if not departure_city or not arrival_city:
            error_msg = "Could not extract departure and arrival cities from your query. Please use format: 'from [city] to [city]'"
            print(error_msg)
            return {'messages': [ToolMessage(tool_call_id='flights_finder', name='flights_finder', content=error_msg)]}
        
        # Lookup airport codes
        departure_airport_code = self._tools['airport_code_lookup'].invoke(input={'q': departure_city})
        if not departure_airport_code or "N/A" in str(departure_airport_code) or "Error:" in str(departure_airport_code):
            error_msg = f"❌ Could not find airport code for departure city: {departure_city}. Please check the city name spelling."
            print(error_msg)
            return {'messages': [ToolMessage(tool_call_id='flights_finder', name='flights_finder', content=error_msg)]}
        
        arrival_airport_code = self._tools['airport_code_lookup'].invoke(input={'q': arrival_city})
        if not arrival_airport_code or "N/A" in str(arrival_airport_code) or "Error:" in str(arrival_airport_code):
            error_msg = f"❌ Could not find airport code for arrival city: {arrival_city}. Please check the city name spelling."
            print(error_msg)
            return {'messages': [ToolMessage(tool_call_id='flights_finder', name='flights_finder', content=error_msg)]}
        
        print(f"Airport codes - Departure: {departure_airport_code}, Arrival: {arrival_airport_code}")

        # Extract hotel class
        hotel_class_match = re.search(r'(\d+)\s*star hotel', user_message)
        hotel_class = hotel_class_match.group(1) if hotel_class_match else None

        flights_args = {
            'departure_airport': departure_airport_code,
            'arrival_airport': arrival_airport_code,
            'outbound_date': check_in_str,
            'return_date': check_out_str,
            'adults': 1,
            'children': 0,
            'infants_in_seat': 0,
            'infants_on_lap': 0
        }
        hotels_args = {
            'q': arrival_city,
            'check_in_date': check_in_str,
            'check_out_date': check_out_str,
            'adults': 1,
            'children': 0,
            'rooms': 1,
            'sort_by': 8,
            'hotel_class': hotel_class
        }
        print(f'Calling flights_finder with: {flights_args}')
        print(f'Calling hotels_finder with: {hotels_args}')
        flights_result = self._tools['flights_finder'].invoke({'params': flights_args})
        print(f'flights_finder output: {flights_result}')
        hotels_result = self._tools['hotels_finder'].invoke({'params': hotels_args})
        print(f'hotels_finder output: {hotels_result}')
        
        # Create the basic travel itinerary
        basic_itinerary = self.format_travel_itinerary(flights_result, hotels_result)
        
        # Create detailed daily itinerary
        daily_itinerary = self.create_daily_itinerary(
            departure_city, 
            arrival_city, 
            check_in_str, 
            check_out_str, 
            hotels_result
        )
        
        # Combine both itineraries
        full_itinerary = basic_itinerary + daily_itinerary
        
        print('Formatted itinerary:', full_itinerary)
        results = [
            ToolMessage(tool_call_id='flights_finder', name='flights_finder', content=full_itinerary)
        ]
        print('Final tool results:', results)
        print('Back to the model!')
        return {'messages': results}
