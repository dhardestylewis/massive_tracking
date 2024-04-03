import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
import geopandas as gpd
from math import radians, cos, sin, asin, sqrt
from numba import jit
import os.path
import pickle
import cProfile
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation
import folium
from folium.plugins import HeatMap
import requests

def download_file(url, local_filename):
    """Download a file from a URL to a local file."""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

def load_and_preprocess_data(data_file, cell_locations, sample_size=0.2):
    """ Combines loading and preprocessing"""
    cdr_data = pd.read_csv(data_file, sep='\t', header=None, 
                           names=['cell_id', 'timestamp', 'user_id', 'sms_out', 'sms_in', 'call_out', 'call_in', 'internet'])

    if sample_size and sample_size < 1:
        cdr_data = cdr_data.sample(frac=sample_size)

    # Merge cdr_data with cell_locations to get latitude and longitude
    cdr_data = cdr_data.merge(cell_locations, left_on='cell_id', right_index=True)
    
#    distances = calculate_distances_vectorized(cell_locations)
    cdr_data = preprocess_cdr_data(cdr_data, cell_locations)#, distances)
    return cdr_data

def identify_homes_and_segment_trips(cdr_data,grid):
    """ Combines two logically related steps"""
    unique_users, home_locations = identify_users_and_home_locations(cdr_data)

    # Ensure the data types are correct after merging
    cdr_data['latitude'] = cdr_data['latitude'].astype(float)
    cdr_data['longitude'] = cdr_data['longitude'].astype(float)

    # Minimal Conversion for Spatial Segmentation
    if not isinstance(cdr_data, gpd.GeoDataFrame): 
        from shapely.geometry import Point 
        cdr_data['geometry'] = cdr_data.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
        cdr_data = gpd.GeoDataFrame(cdr_data, geometry='geometry', crs='epsg:32632')  # Replace with the correct CRS

    # Perform the spatial segmentation
    trips = segment_trips_spatially(cdr_data,grid)

    return unique_users, home_locations, trips

def analyze_and_visualize(cdr_data, unique_users, home_locations, trips, cell_locations):
    """ Encapsulates the analysis steps"""
    print(f"Number of trips: {len(trips)}")
    if len(trips) > 0:
        print(f"First trip has {len(trips[0])} records")
        for trip in trips[:5]:
            print(trip)
    else:
        print("No trips found.")

    if os.path.exists('../mobility_metrics.pickle'):
        with open('../mobility_metrics.pickle', 'rb') as f:
            mobility_metrics = pickle.load(f)
    else:
        mobility_metrics = calculate_mobility_metrics(trips, cell_locations)
        with open('../mobility_metrics.pickle', 'wb') as f:
            pickle.dump(mobility_metrics, f)
    print(mobility_metrics[['user_id', 'start_time', 'end_time', 'duration']].head())

    hourly_activity = analyze_mobility_patterns(cdr_data, mobility_metrics)
    print(hourly_activity) 
    # Add further analysis and visualizations here 

def haversine_distance(coord1, coord2):
    """
    Calculate the great-circle distance between two points 
    on the Earth surface given their latitude and longitude in decimal degrees.
    """
    # NEW (CORRECT) - Accepts coordinates as NumPy array elements  
    lat1, lon1 = map(radians, coord1) 
    lat2, lon2 = map(radians, coord2) 

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of Earth in kilometers
    return c * r

#def calculate_distances_vectorized(cell_locations):
#    coords = cell_locations[['latitude', 'longitude']].to_numpy()
#    print(coords.shape)  #  <--- Add this line
#    print(coords[:5])  # Print the first 5 rows of coords
#    
#    # Broadcasting for efficient pairwise calculations
#    lat1, lon1 = coords[:, 0, None], coords[:, 1, None]  # Extract directly
#    lat2, lon2 = coords[None, :, 0], coords[None, :, 1]
#
#    dlat = lat2 - lat1 
#    dlon = lon2 - lon1
#    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
#    c = 2 * np.arcsin(np.sqrt(a))
#    r = 6371  # Radius of Earth in kilometers
#    distances = r * c
#
#    return distances

#def calculate_distance(cell_id_1, cell_id_2):
#    if cell_id_1 == 0 or cell_id_2 == 0:
#        return 0  # Or a suitable default distance
#
#    row_1, col_1 = divmod(cell_id_1, 100)
#    row_2, col_2 = divmod(cell_id_2, 100)
#    distance = np.sqrt((row_1 - row_2)**2 + (col_1 - col_2)**2)
#    return distance

def calculate_distance(coord1, coord2):
    """
    Calculate the great-circle distance between two points
    on the Earth surface given their latitude and longitude in decimal degrees.
    """
    lat1, lon1 = map(radians, coord1)
    lat2, lon2 = map(radians, coord2)

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r

def preprocess_cdr_data(cdr_data, cell_locations):#,distances):
    """
    Clean and preprocess the raw CDR data.
    """
    # Remove erroneous CDRs (e.g., impossible timestamps, locations)
    start_date = pd.Timestamp('2013-12-03 00:00:00')
    end_date = pd.Timestamp('2013-12-03 23:59:59')
    date_range = pd.date_range(start=start_date, end=end_date, freq='ms')
    # Format consistency
    cdr_data['timestamp'] = pd.to_datetime(cdr_data['timestamp'], unit='ms')
    cdr_data = cdr_data[cdr_data['timestamp'].isin(date_range)]
    cdr_data = cdr_data[cdr_data['cell_id'] >= 0]
    cdr_data = cdr_data[cdr_data['cell_id'] != 0]  # Assuming 0 is an invalid or placeholder cell ID
    cdr_data['cell_id'] = cdr_data['cell_id'].astype(int)

    # Filter out stationary records (long durations at the same location)
    cdr_data = cdr_data.sort_values(['user_id', 'timestamp'])
    cdr_data['prev_cell_id'] = cdr_data.groupby('user_id')['cell_id'].shift(1)
    cdr_data['cell_id'] = pd.to_numeric(cdr_data['cell_id'], errors='coerce').fillna(0).astype(int)
    cdr_data['prev_cell_id'] = pd.to_numeric(cdr_data['prev_cell_id'], errors='coerce').fillna(0).astype(int)
    cdr_data = cdr_data[cdr_data['cell_id'] != cdr_data['prev_cell_id']]

    # Use vectorized operations for speed
#    cdr_data['distance_from_prev'] = cdr_data.apply(lambda row: calculate_distance(row['cell_id'], row['prev_cell_id']), axis=1)
    cdr_data['distance_from_prev'] = cdr_data.apply(lambda row: calculate_distance(
        (cell_locations.loc[row['cell_id']]['latitude'], cell_locations.loc[row['cell_id']]['longitude']) if row['cell_id'] in cell_locations.index else (0, 0),
        (cell_locations.loc[row['prev_cell_id']]['latitude'], cell_locations.loc[row['prev_cell_id']]['longitude']) if row['prev_cell_id'] in cell_locations.index else (0, 0)
    ), axis=1)

#    cdr_data['distance_from_prev'] = cdr_data.apply(calculate_distance_safe, axis=1)
    min_distance_threshold = 100  # Meters
    cdr_data = cdr_data[cdr_data['distance_from_prev'] >= min_distance_threshold]
    print(cdr_data[['timestamp', 'cell_id', 'prev_cell_id']].head())
    cdr_data.drop(['prev_cell_id', 'distance_from_prev'], axis=1, inplace=True)

    return cdr_data

def calculate_mobility_metrics(trips, cell_locations):
    """
    Calculate various mobility metrics from the segmented trips.
    """
    mobility_metrics = []

    for trip in trips:
        if not trip or pd.isna(trip[0]['cell_id']) or trip[0]['cell_id'] == 0:
            continue  # Skip trips with problematic first records
        if 'user_id' not in trip[0] or pd.isna(trip[0]['user_id']):
            continue  # Skip trips with missing or invalid user_id 

        user_id = trip[0]['user_id']  # Use iloc for safety
        origin = trip[0]['cell_id']
        destination = trip[-1]['cell_id']
        start_time = trip[0]['timestamp']
        end_time = trip[-1]['timestamp']
        duration = (end_time - start_time).total_seconds() / 60  # Duration in minutes

        # Calculate trip length
        trip_length = calculate_trip_length((cell_locations.loc[origin]['latitude'], cell_locations.loc[origin]['longitude']),
                                            (cell_locations.loc[destination]['latitude'], cell_locations.loc[destination]['longitude']))

        if duration == 0: 
            average_speed = 0  # Or a suitable default if short trips are meaningful
        else: 
            average_speed = trip_length / duration * 60  # km/h
    
        # Calculate radius of gyration
        radius_of_gyration = calculate_radius_of_gyration(trip, cell_locations)

        # Calculate entropy
        entropy = calculate_entropy(trip)

        mobility_metrics.append({
            'user_id': user_id,
            'origin': origin,
            'destination': destination,
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'trip_length': trip_length,
            # Add other calculated metrics
        })

    return pd.DataFrame(mobility_metrics)

def analyze_temporal_spatial_patterns(cdr_data, mobility_metrics, hotspots, frequent_locations, travel_corridors):
    """
    Examine variations in mobility patterns across different times and locations.
    """
    # Identify peak hours or days for different spatial units
    hourly_activity_by_cell = cdr_data.groupby(['cell_id', cdr_data['timestamp'].dt.hour])['timestamp'].count().unstack(fill_value=0)
    peak_hours_by_cell = hourly_activity_by_cell.idxmax(axis=1)

    # Visualize the peak hours for each cell
    plt.figure(figsize=(12, 8))
    sns.heatmap(hourly_activity_by_cell, cmap='YlOrRd', mask=hourly_activity_by_cell.isnull())
    plt.title('Peak Hours by Cell')
    plt.xlabel('Hour')
    plt.ylabel('Cell ID')
    plt.show()

    # Analyze how mobility patterns vary between hotspots, frequent locations, and travel corridors
    hotspot_trips = mobility_metrics[mobility_metrics['origin'].isin(hotspots['cell_id']) | mobility_metrics['destination'].isin(hotspots['cell_id'])]
    frequent_location_trips = mobility_metrics[mobility_metrics['origin'].isin(frequent_locations['cell_id']) | mobility_metrics['destination'].isin(frequent_locations['cell_id'])]
    corridor_trips = mobility_metrics[mobility_metrics[['origin', 'destination']].apply(tuple, axis=1).isin(travel_corridors[['origin', 'destination']].apply(tuple, axis=1))]

    # Compare the average trip duration and distance for each category
    print('Average Trip Duration:')
    print(f'Hotspots: {hotspot_trips["duration"].mean():.2f} minutes')
    print(f'Frequent Locations: {frequent_location_trips["duration"].mean():.2f} minutes')
    print(f'Travel Corridors: {corridor_trips["duration"].mean():.2f} minutes')

    print('\nAverage Trip Distance:')
    print(f'Hotspots: {hotspot_trips["trip_length"].mean():.2f} km')
    print(f'Frequent Locations: {frequent_location_trips["trip_length"].mean():.2f} km')
    print(f'Travel Corridors: {corridor_trips["trip_length"].mean():.2f} km')

    # Visualize the temporal-spatial dynamics
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=mobility_metrics['start_time'].dt.hour, y=mobility_metrics['origin'], hue=mobility_metrics['destination'], data=mobility_metrics)
    plt.title('Temporal-Spatial Dynamics of Mobility')
    plt.xlabel('Start Time (hour)')
    plt.ylabel('Origin Cell ID')
    plt.show()

    return {
        'peak_hours_by_cell': peak_hours_by_cell,
        'hotspot_trips': hotspot_trips,
        'frequent_location_trips': frequent_location_trips,
        'corridor_trips': corridor_trips
    }

def analyze_mobility_patterns(cdr_data, mobility_metrics):
    """
    Analyze mobility patterns based on the processed data and calculated metrics.
    """
    # Temporal analysis
    hourly_activity = cdr_data.groupby(cdr_data['timestamp'].dt.hour).size()
    # Analyze hourly variations in activity

    # Identify hotspots, frequent locations, travel corridors
    hotspots = identify_hotspots(cdr_data)
    frequent_locations = identify_frequent_locations(cdr_data)

    # Identify travel corridors
    travel_corridors = identify_travel_corridors(mobility_metrics)

    # Combine temporal and spatial analysis
    # Examine variations in mobility patterns across different times and locations
    if os.path.exists('../temporal_spatial_analysis.pickle'):
        with open('../temporal_spatial_analysis.pickle', 'rb') as f:
            temporal_spatial_analysis = pickle.load(f)
    else:
        temporal_spatial_analysis = analyze_temporal_spatial_patterns(cdr_data, mobility_metrics, hotspots, frequent_locations, travel_corridors)
        with open('../temporal_spatial_analysis.pickle', 'wb') as f:
            pickle.dump(temporal_spatial_analysis, f)

    # Other analyses (e.g., activity patterns, etc)
#    activity_patterns = analyze_activity_patterns(cdr_data)

    return hourly_activity  # Return any relevant analysis results

#def precalculate_distance_matrix(cell_locations):
#    coords = cell_locations[['latitude', 'longitude']].to_numpy()
#    num_cells = len(coords)
#    distance_matrix = np.zeros((num_cells, num_cells))
#
#    for i in range(num_cells):
#        for j in range(i, num_cells):  # Exploit symmetry
#            distance_matrix[i, j] = distance_matrix[j, i] = haversine_distance(coords[i], coords[j])
#
#    cell_id_to_index = {cell_id: i for i, cell_id in enumerate(cell_locations.index) if not pd.isna(cell_id)}
#    return distance_matrix, cell_id_to_index

#@jit(nopython=True)  # Tell Numba to compile this function
#def calculate_distance(cell_id1, cell_id2):
#    if pd.isna(cell_id1) or pd.isna(cell_id2) or cell_id1 == 0 or cell_id2 == 0:
#        return None  # Or a suitable default distance
#
#    index1 = cell_id_to_index[cell_id1]
#    index2 = cell_id_to_index[cell_id2]
#    return distance_matrix[index1, index2]
#    """
#    Calculate the distance between two cell towers based on their IDs.
#    """
#    try:
#        # If either cell ID is not found, print a warning and return None
#        if cell_id1 not in cell_locations.index or cell_id2 not in cell_locations.index:
#            print(f"Warning: Cell ID {cell_id1} or {cell_id2} not found")
#            return None
#        coord1 = cell_locations.loc[cell_id1] 
#        coord2 = cell_locations.loc[cell_id2]
#        # Calculate distance using the Haversine formula or another suitable method:
#        distance_km = haversine_distance(coord1, coord2)  
#        return distance_km
#    except KeyError:
#        print(f"Warning: Cell ID {cell_id1} or {cell_id2} not found")
#        return None  # Or a suitable default value

def calculate_trip_length(origin, destination):
    """
    Calculate the trip length based on the origin and destination cell IDs.
    Assumes you have a method to estimate the route between two cell towers.
    """
    # Placeholder: Estimate a simple straight-line path
    if isinstance(origin, (int, str)) and isinstance(destination, (int, str)):
        # Origin and destination are cell IDs
        origin_coord = (cell_locations.loc[origin]['latitude'], cell_locations.loc[origin]['longitude'])
        destination_coord = (cell_locations.loc[destination]['latitude'], cell_locations.loc[destination]['longitude'])
    else:
        # Origin and destination are coordinates
        origin_coord = origin
        destination_coord = destination

    return calculate_distance(origin, destination)

def calculate_center_of_mass(unique_locations, cell_locations):
    latitudes = [cell_locations.loc[loc]['latitude'] for loc in unique_locations]
    longitudes = [cell_locations.loc[loc]['longitude'] for loc in unique_locations]
    return (np.mean(latitudes), np.mean(longitudes))

def calculate_radius_of_gyration(trip, cell_locations):
    """
    Calculate the radius of gyration for a given trip.
    """
    cell_ids = [int(record['cell_id']) for record in trip]
    unique_locations = set(cell_ids)
    center_of_mass = calculate_center_of_mass(unique_locations, cell_locations)

    distances = [calculate_distance((cell_locations.loc[loc]['latitude'], cell_locations.loc[loc]['longitude']),
                                   center_of_mass) for loc in unique_locations]
    return np.sqrt(np.mean([d**2 for d in distances]))

def calculate_entropy(trip):
    """
    Calculate the entropy for a given trip (a measure of its predictability).
    """
    cell_ids = [record['cell_id'] for record in trip]
    cell_id_counts = Counter(cell_ids)
    probabilities = [count / len(cell_ids) for count in cell_id_counts.values()]
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def aggregate_to_spatial_scale(cdr_data, spatial_scale):
    """Aggregate CDR data to a spatial scale (ex: neighborhoods)"""
    # Requires a mapping from cell tower IDs to neighborhoods
    cdr_data['neighborhood'] = cdr_data['cell_id'].map(cell_id_to_neighborhood)  

    return cdr_data.groupby(['user_id', 'neighborhood'])['timestamp'].count().reset_index()

def identify_hotspots(cdr_data):
    """Identify hotspots based on aggregated data"""
    cell_record_counts = cdr_data.groupby('cell_id')['timestamp'].count()
    hotspots_threshold = cell_record_counts.quantile(0.95)
    hotspot_cell_ids = cell_record_counts[cell_record_counts >= hotspots_threshold].index
    return cdr_data[cdr_data['cell_id'].isin(hotspot_cell_ids)]

def identify_frequent_locations(cdr_data):
    """Identify frequently visited locations"""
    cell_record_counts = cdr_data.groupby('cell_id')['timestamp'].count()
    freq_threshold = cell_record_counts.quantile(0.9)
    frequent_cell_ids = cell_record_counts[cell_record_counts >= freq_threshold].index
    return cdr_data[cdr_data['cell_id'].isin(frequent_cell_ids)]

def identify_travel_corridors(mobility_metrics):
    """Identify common travel corridors"""
    corridor_threshold = 10  # Minimum trips between an origin-destination pair
    return mobility_metrics.groupby(['origin', 'destination']).size().reset_index().rename(columns={0: 'count'})[lambda x: x['count'] >= corridor_threshold]

def identify_users_and_home_locations(cdr_data):
    """
    Determine unique users and estimate their home locations.
    """
    unique_users = cdr_data['user_id'].unique()
    print(f"Number of unique users: {len(unique_users)}")

    # Estimate home location based on recurrent overnight activity
    overnight_records = cdr_data.loc[cdr_data['timestamp'].dt.hour.between(22, 8), ['user_id', 'cell_id']]
    print(f"Number of overnight records: {len(overnight_records)}")

    if len(overnight_records) > 0:
        home_locations = overnight_records.groupby('user_id')['cell_id'].apply(lambda x: Counter(x).most_common(1)[0][0]).reset_index()
        home_locations.columns = ['user_id', 'home_location']
        print(f"Number of home locations identified: {len(home_locations)}")
    else:
        print("No overnight records found. Estimating home locations using most frequent cell_id.")
        home_locations = cdr_data.groupby('user_id')['cell_id'].apply(lambda x: Counter(x).most_common(1)[0][0]).reset_index()
        home_locations.columns = ['user_id', 'home_location']

    return unique_users, home_locations

def segment_trips_spatially(cdr_data, grid):
    """Segment CDR data into trips based on spatial units (e.g., grid cells)"""

    # Add a 'grid_cell_id' column to the CDR data
    cdr_data = gpd.sjoin(cdr_data, grid, how='left', predicate='within')  
    print(f"Number of records after spatial join: {len(cdr_data)}")

    trips = []
    current_trip = []
    prev_grid_cell_id = None

    for idx, row in cdr_data.iterrows():
        if prev_grid_cell_id is None or row['cell_id'] != prev_grid_cell_id:
            if current_trip:
                trips.append(current_trip)
                current_trip = []

        current_trip.append(row)
        prev_grid_cell_id = row['cell_id']

    if current_trip:
        trips.append(current_trip)

    # Count trips with more than one record
    trips_with_more_than_one_record = sum(len(trip) > 1 for trip in trips)
    print(f"Number of trips with more than one record: {trips_with_more_than_one_record}")

    return trips

def segment_trips(cdr_data, dwell_time_threshold=.03):
    """
    Segment the CDR data into individual trips based on dwell time.
    """
    cdr_data = cdr_data.sort_values(['user_id', 'timestamp'])
    cdr_data['dwell_time'] = cdr_data.groupby('user_id')['timestamp'].diff().dt.total_seconds() / 60

    print(f"Minimum dwell time: {cdr_data['dwell_time'].min()}")
    print(f"Maximum dwell time: {cdr_data['dwell_time'].max()}")
    print(f"Mean dwell time: {cdr_data['dwell_time'].mean()}")
    print(f"Median dwell time: {cdr_data['dwell_time'].median()}")
    print(f"Descriptive statistics for dwell times (in minutes):\n {cdr_data['dwell_time'].describe()}")

    # Visualization
    sns.displot(cdr_data, x='dwell_time', kind='hist')
    plt.xlabel('Dwell Time (minutes)')
    plt.title('Distribution of Dwell Times')
    plt.show()

    trips = []
    current_trip = []
    prev_cell_id = None

    for idx, row in cdr_data.iterrows():
        if prev_cell_id is None or row['cell_id'] != prev_cell_id or row['dwell_time'] > dwell_time_threshold:
            if current_trip:
                trips.append(current_trip)
                current_trip = []

        current_trip.append(row)
        prev_cell_id = row['cell_id']

    if current_trip:
        trips.append(current_trip)

    for trip in trips[:5]:  # Assuming trips is a list of segmented trips
        print(f"Trip Start: {trip[0]['timestamp']}, End: {trip[-1]['timestamp']}")

    # Optionally, print details of the first few trips with more than one record for inspection
    detailed_count = 0
    for trip in trips:
        if len(trip) > 1 and detailed_count < 5:  # Adjust the number to display as needed
            print(f"Trip with {len(trip)} records: Start at {trip[0]['timestamp']}, End at {trip[-1]['timestamp']}")
            detailed_count += 1

    return trips

def analyze_internet_traffic(cdr_data, cell_locations):
    """
    Analyze internet traffic activity based on the preprocessed CDR data.
    """
    # Filter for internet traffic
    internet_traffic = cdr_data[cdr_data['internet'] > 0]

    # Aggregate internet traffic by square ID and time interval
    # Aggregate internet traffic by square ID and time interval
    internet_traffic_by_square = internet_traffic.merge(cell_locations[['cell_id', 'latitude', 'longitude']], on='cell_id')
    internet_traffic_by_square = internet_traffic.groupby(['cell_id', internet_traffic['timestamp'].dt.floor('H')]).size().reset_index()
    internet_traffic_by_square.columns = ['cell_id', 'timestamp', 'internet_traffic_volume']

    # Descriptive statistics of internet activity per square ID
    internet_activity_stats = internet_traffic_by_square.groupby('cell_id')['internet_traffic_volume'].describe()

    # Internet activity on hourly basis
    hourly_internet_activity = internet_traffic_by_square.groupby(internet_traffic_by_square['timestamp'].dt.hour)['internet_traffic_volume'].sum()

    # Save the results
    internet_traffic_by_square.to_csv('../results/internet_traffic_volume.csv', index=False)
    internet_activity_stats.to_csv('../results/internet_activity_stats.csv')

    # Visualize the hourly internet activity
    plt.figure(figsize=(12, 6))
    hourly_internet_activity.plot(kind='bar')
    plt.title('Hourly Internet Activity')
    plt.xlabel('Hour')
    plt.ylabel('Internet Traffic Volume')
    plt.savefig('../results/hourly_internet_activity.png')
    plt.close()

    # Create the spatial animation
    m = folium.Map(location=[45.4642, 9.1900], zoom_start=12)
    heatmap = HeatMap(data=internet_traffic_by_square[['latitude', 'longitude', 'internet_traffic_volume']].values,                     name='Internet Traffic',
                     radius=10,
                     min_opacity=0.2,
                     max_opacity=0.8,
                     gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1.0: 'red'})
    heatmap.add_to(m)

    def update(frame):
        hour = frame
        data = internet_traffic_by_square[internet_traffic_by_square['timestamp'].dt.hour == hour]
        heatmap.data = data[['latitude', 'longitude', 'internet_traffic_volume']].values
        heatmap.add_to(m)
        m.get_root().html.add_child(folium.Element(f"<h3>Hour {hour}</h3>"))
        return [heatmap]

    ani = animation.FuncAnimation(m, update, frames=range(24), interval=500)
    ani.save('../results/hourly_internet_activity.gif', writer='imagemagick')
    m.save('../results/hourly_internet_activity_map.html')

    return internet_traffic_by_square, internet_activity_stats, hourly_internet_activity

def main():
#    cProfile.run('main()', sort='time')

    file_url = "https://dataverse.harvard.edu/api/access/datafile/:persistentId/?persistentId=doi:10.7910/DVN/EGZHFV/1MIYTI"
    data_file = '../data/sms-call-internet-mi-2013-12-03.txt'
    download_file(file_url, data_file)
    # Load grid and reproject
    grid = gpd.read_file('../data/milano-grid.geojson')
    grid = grid.to_crs(epsg=32632) # Or whatever the correct EPSG code is
    grid['centroid'] = grid.centroid

    if os.path.exists('../cell_locations.pickle'):
        with open('../cell_locations.pickle', 'rb') as f:
            cell_locations = pickle.load(f)
    else:
        # Load and process the cell locations
        cell_locations = grid[['cellId', 'centroid']].set_index('cellId')
        cell_locations['latitude'] = cell_locations['centroid'].y
        cell_locations['longitude'] = cell_locations['centroid'].x  
        cell_locations = cell_locations.drop('centroid', axis=1)
        # Make sure the cellId is of type int and set as the index
        cell_locations = cell_locations.reset_index()
        cell_locations['cellId'] = cell_locations['cellId'].astype(int)
        cell_locations = cell_locations.set_index('cellId')
        with open('../cell_locations.pickle', 'wb') as f:
            pickle.dump(cell_locations, f)

    def validate_cdr_data(cdr_data):
        required_columns = ['cell_id', 'timestamp', 'user_id']
        if not all(col in cdr_data.columns for col in required_columns):
            raise ValueError("CDR data is missing required columns")
        # Check distribution of cell_id values
        print(f"Unique cell_id values: {cdr_data['cell_id'].nunique()}")
        print(f"Minimum cell_id: {cdr_data['cell_id'].min()}")
        print(f"Maximum cell_id: {cdr_data['cell_id'].max()}")
    
        # Check timestamp column
        print(f"Earliest timestamp: {cdr_data['timestamp'].min()}")
        print(f"Latest timestamp: {cdr_data['timestamp'].max()}")
    
        # Check user_id column
        print(f"Unique user_id values: {cdr_data['user_id'].nunique()}")

    if os.path.exists('../preprocessed_data.pickle'):
        with open('../preprocessed_data.pickle', 'rb') as f:
            cdr_data = pickle.load(f)
    else:
        cdr_data = load_and_preprocess_data(data_file, cell_locations)
        validate_cdr_data(cdr_data)  # Check data integrity
        with open('../preprocessed_data.pickle', 'wb') as f:
            pickle.dump(cdr_data, f)

    internet_traffic_by_square, internet_activity_stats, hourly_internet_activity = analyze_internet_traffic(cdr_data, cell_locations)

    if os.path.exists('../homes_and_trips.pickle'):
        with open('../homes_and_trips.pickle', 'rb') as f:
            unique_users, home_locations, trips = pickle.load(f)
    else:
        unique_users, home_locations, trips = identify_homes_and_segment_trips(cdr_data,grid)
        with open('../homes_and_trips.pickle', 'wb') as f:
            pickle.dump((unique_users, home_locations, trips), f)

    analyze_and_visualize(cdr_data, unique_users, home_locations, trips, cell_locations)

if __name__ == "__main__":
    main()

