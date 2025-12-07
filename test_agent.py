from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from geopy.geocoders import Nominatim
import time
import pandas as pd
import osmnx as ox
from models.trip_price_class import TripPricePredictor
import heapq
from tools import *
from langchain.agents import create_agent

g = ox.graph_from_xml("labeled.osm", bidirectional=True, simplify=True)
g = attach_trips_to_graph(g)
set_graph(g)

print("✅ Graph initialized")
print(g.nodes[list(g.nodes)[0]].keys())


pathways = pd.read_csv('trip_pathways.csv')


trip_graph = defaultdict(dict)
pathways_dict = pathways.to_dict('index')

for idx, row in pathways.iterrows():
    trip_graph[row['start_trip_id']][row['end_trip_id']] = idx 
    
set_trip_graph(trip_graph, pathways_dict)




system_prompt = """
You are a smart assistant specialized in Alexandria public transportation. 
You have access to the following tools:

1. geocode_address(address) -> returns the latitude and longitude of the address.
2. get_nearest_node(lat, lon) -> returns the nearest OSM node ID.
3. explore_trips(source_node) -> returns all trips starting from this node, including walking distance.
4. find_journeys(start_trips, goal_trips) -> returns all possible journeys with path and costs (money, walking distance).
5. filter_best_journeys(journeys, max_results=5) -> returns the best journeys based on shortest walking distance and lowest cost.
6. format_journeys_for_user(journeys) -> returns a user-friendly Arabic description of the journeys.

You must always follow this workflow:

1. Find the coordinates of the start and destination using geocode_address.
2. Convert each location into the nearest OSM node using get_nearest_node.
3. Explore trips from both start and destination nodes using explore_trips.
4. Find all possible journeys using find_journeys.
5. Filter the top journeys using filter_best_journeys.
6. Format the filtered journeys for the user using format_journeys_for_user.
7. Return only the final formatted journey description to the user. Do not return any intermediate data.
"""


model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key="")

tools = [geocode_address, get_nearest_node, explore_trips,find_journeys,filter_best_journeys,format_journeys_for_user]

agent = create_agent(model, tools=tools)

# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

res= agent.invoke({"messages": [("system", system_prompt),("user", "أريد الذهاب من الموقف الجديد الي العصافرة")]})

print(res)
# agent.run("أريد الذهاب من العجمي إلى محطة الرمل")

# response = model.invoke("هو ازاي اروح من محطة مصر للعجمي ؟ في مشاريع بتروح هناك؟")
# print(response.content)