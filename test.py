import pandas as pd
import numpy as np
import requests
from geopy.distance import great_circle
from datetime import datetime
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix

class DeliveryOptimizer:
    def __init__(self, store_location):
        self.store_location = store_location
        self.osrm_base_url = "http://localhost:5000/route/v1/driving/"
        
        # Define vehicle type priority order and max shipment capacity
        self.vehicle_priority = ['3W', '4W-EV', '4W']
        self.max_shipments_per_vehicle = {
            '3W': 5,
            '4W-EV': 8, 
            '4W': 25
        }

    def calculate_mst_distance(self, points):
        """
        Calculate MST distance for a set of points
        """
        n_points = len(points)
        distances = np.zeros((n_points, n_points))
        
        for i in range(n_points):
            for j in range(i + 1, n_points):
                distance = great_circle(points[i], points[j]).km
                distances[i][j] = distance
                distances[j][i] = distance
        
        # Create sparse matrix and calculate MST
        sparse_distances = csr_matrix(distances)
        mst = minimum_spanning_tree(sparse_distances)
        
        return mst.sum()

    def calculate_osrm_route(self, coordinates):
        """
        Calculate route using local OSRM server
        Returns duration (minutes), distance (km), and full route details
        """
        try:
            coords_string = ";".join([f"{lon},{lat}" for lat, lon in coordinates])
            url = f"{self.osrm_base_url}{coords_string}?overview=full&annotations=true"
            
            response = requests.get(url)
            if response.status_code != 200:
                print(f"OSRM request failed: {response.status_code}")
                return self.fallback_route_calculation(coordinates)
            
            route_data = response.json()
            duration = route_data['routes'][0]['duration'] / 60
            distance = route_data['routes'][0]['distance'] / 1000
            route_geometry = route_data['routes'][0]['geometry']
            
            return duration, distance, route_geometry
        except Exception as e:
            print(f"Error in OSRM routing: {e}")
            return self.fallback_route_calculation(coordinates)

    def find_optimal_cluster(self, delivery_points, max_radius):
        """
        Find optimal cluster of points that satisfies MST constraint
        """
        if not delivery_points:
            return []
            
        all_points = [self.store_location] + delivery_points
        current_points = delivery_points.copy()
        
        while current_points:
            mst_distance = self.calculate_mst_distance([self.store_location] + current_points)
            if mst_distance <= max_radius:
                return current_points
            current_points.pop()
            
        return []

    def optimize_deliveries(self, delivery_df, vehicle_df):
        """
        Enhanced delivery optimization using MST-based clustering
        """
        # Initialize vehicle priorities
        vehicle_df['Priority'] = vehicle_df['Vehicle Type'].map({
            vtype: idx for idx, vtype in enumerate(self.vehicle_priority)
        })
        vehicle_df = vehicle_df.sort_values(['Priority', 'Shipments_Capacity'], ascending=[True, False])
        
        optimized_routes = []
        trip_id = 1
        
        # Process each time slot
        for time_slot, time_slot_df in delivery_df.groupby('Delivery Timeslot'):
            print(f"\nProcessing Time Slot: {time_slot}")
            start_time, end_time = self.parse_time_slot(time_slot)
            max_trip_duration = (datetime.combine(datetime.today(), end_time) - 
                               datetime.combine(datetime.today(), start_time)).total_seconds() / 60
            
            unassigned_deliveries = time_slot_df.copy()
            
            # Process vehicles in priority order
            for _, vehicle in vehicle_df.iterrows():
                vehicle_type = vehicle['Vehicle Type']
                vehicle_capacity = self.max_shipments_per_vehicle[vehicle_type]
                vehicle_max_radius = vehicle['Max Trip Radius (in KM)']
                available_vehicles = vehicle['Number']
                
                while not unassigned_deliveries.empty and available_vehicles > 0:
                    delivery_points = [(row['Latitude'], row['Longitude']) 
                                     for _, row in unassigned_deliveries.iterrows()]
                    
                    # Find optimal cluster based on MST constraint
                    optimal_points = self.find_optimal_cluster(
                        delivery_points[:vehicle_capacity], 
                        vehicle_max_radius
                    )
                    
                    if not optimal_points:
                        break
                        
                    # Get corresponding deliveries
                    current_group = unassigned_deliveries.iloc[:len(optimal_points)]
                    
                    # Calculate final route details using OSRM
                    route_coords = [self.store_location] + optimal_points + [self.store_location]
                    duration, distance, route_geometry = self.calculate_osrm_route(route_coords)
                    
                    if duration <= max_trip_duration:
                        # Create route information
                        route_info = {
                            'TRIP_ID': f'T{trip_id:03d}',
                            'Vehicle_Type': vehicle_type,
                            'Time_Slot': time_slot,
                            'Total_Shipments': len(current_group),
                            'Total_Distance': distance,
                            'Trip_Duration': duration,
                            'MST_Distance': self.calculate_mst_distance([self.store_location] + optimal_points),
                            'Capacity_Utilization': (len(current_group) / vehicle_capacity) * 100,
                            'Shipment_Details': current_group.to_dict('records'),
                            'Route_Geometry': route_geometry,
                            'Optimized_Route': route_coords,
                            'Max_Trip_Duration': max_trip_duration,
                            'Vehicle_Max_Radius': vehicle_max_radius
                        }
                        
                        optimized_routes.append(route_info)
                        trip_id += 1
                        available_vehicles -= 1
                        
                        # Remove assigned deliveries
                        unassigned_deliveries = unassigned_deliveries.drop(current_group.index)
                        
                        # Print route summary
                        print(f"\nCreated Route {route_info['TRIP_ID']}:")
                        print(f"Vehicle Type: {vehicle_type}")
                        print(f"Deliveries: {len(current_group)}")
                        print(f"Distance: {distance:.2f} km")
                        print(f"MST Distance: {route_info['MST_Distance']:.2f} km")
                        print(f"Duration: {duration:.2f} minutes")
                        print(f"Capacity Utilization: {route_info['Capacity_Utilization']:.1f}%")
                    else:
                        # If duration exceeds limit, try with fewer deliveries
                        unassigned_deliveries = unassigned_deliveries.iloc[1:]
        
        return optimized_routes

    def parse_time_slot(self, time_slot):
        """Parse time slot string to start and end datetime objects"""
        start_time, end_time = time_slot.split('-')
        start_dt = datetime.strptime(start_time, '%H:%M:%S').time()
        end_dt = datetime.strptime(end_time, '%H:%M:%S').time()
        return start_dt, end_dt

    def fallback_route_calculation(self, coordinates):
        """
        Fallback method if OSRM routing fails
        """
        total_distance = sum(
            great_circle(coordinates[i], coordinates[i + 1]).km 
            for i in range(len(coordinates) - 1)
        )
        estimated_duration = (total_distance / 30) * 60
        return estimated_duration, total_distance, None

def prepare_output_data(optimized_routes):
    """
    Prepare data in the required format for Excel output
    """
    output_rows = []
    
    for route in optimized_routes:
        # Calculate utilization metrics
        time_uti = route['Trip_Duration'] / route['Max_Trip_Duration']
        cov_uti = route['Total_Distance'] / route['Vehicle_Max_Radius']
        
        # Process each shipment in the route
        for shipment in route['Shipment_Details']:
            row = {
                'TRIP ID': route['TRIP_ID'],
                'Shipment ID': shipment['Shipment ID'],
                'Latitude': shipment['Latitude'],
                'Longitude': shipment['Longitude'],
                'TIME SLOT': route['Time_Slot'],
                'Shipments': route['Total_Shipments'],
                'MST_DIST': route['MST_Distance'],
                'TRIP_TIME': route['Trip_Duration'],
                'Vehical_Type': route['Vehicle_Type'],  # Note: Keeping the typo as per requirement
                'CAPACITY_UTI': route['Capacity_Utilization'] / 100,  # Convert percentage to decimal
                'TIME_UTI': time_uti,
                'COV_UTI': cov_uti
            }
            output_rows.append(row)
    
    return pd.DataFrame(output_rows)

def save_results(optimized_routes, filename='Optimized_Routes_MSTFinal.xlsx'):
    """
    Save results to Excel in the required format
    """
    # Prepare data in the required format
    output_df = prepare_output_data(optimized_routes)
    
    # Sort by TRIP ID and Shipment ID
    output_df = output_df.sort_values(['TRIP ID', 'Shipment ID'])
    
    # Save to Excel
    with pd.ExcelWriter(filename) as writer:
        # Save main results
        output_df.to_excel(writer, sheet_name='Routes', index=False)
        
        # Add summary sheet
        vehicle_summary = output_df.groupby('Vehical_Type').agg({
            'Shipments': ['sum', 'mean'],
            'MST_DIST': ['mean', 'max'],
            'CAPACITY_UTI': 'mean',
            'TIME_UTI': 'mean',
            'COV_UTI': 'mean'
        }).round(3)
        
        vehicle_summary.to_excel(writer, sheet_name='Vehicle_Summary')

def main():
    # Coordinates of the store (Mumbai example)
    store_location = (19.075987, 72.877656)
    
    try:
        # Load delivery and vehicle data
        delivery_df = pd.read_excel("Shipments_tiimeslot.xlsx")
        vehicle_df = pd.read_excel("vehicle_data.xlsx")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Validate input data
    required_columns = ['Shipment ID', 'Latitude', 'Longitude', 'Delivery Timeslot']
    vehicle_columns = ['Vehicle Type', 'Number', 'Shipments_Capacity', 'Max Trip Radius (in KM)']
    
    if not all(col in delivery_df.columns for col in required_columns):
        raise ValueError(f"Missing delivery data columns: {[col for col in required_columns if col not in delivery_df.columns]}")
    if not all(col in vehicle_df.columns for col in vehicle_columns):
        raise ValueError(f"Missing vehicle data columns: {[col for col in vehicle_columns if col not in vehicle_df.columns]}")
    
    # Clean and prepare data
    delivery_df['Latitude'] = pd.to_numeric(delivery_df['Latitude'], errors='coerce')
    delivery_df['Longitude'] = pd.to_numeric(delivery_df['Longitude'], errors='coerce')
    delivery_df = delivery_df.dropna(subset=['Latitude', 'Longitude', 'Delivery Timeslot'])
    
    # Create and run optimizer
    optimizer = DeliveryOptimizer(store_location)
    optimized_routes = optimizer.optimize_deliveries(delivery_df, vehicle_df)
    
    # Save results in the new format
    save_results(optimized_routes)
    
    print("\nOptimization complete. Detailed results saved to Optimized_Routes_MSTTest.xlsx")

if __name__ == "__main__":
    main()