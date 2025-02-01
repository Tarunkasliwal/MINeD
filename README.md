# SmartRoute Optimizer - Delivery Planning Solution

SmartRoute Optimizer is an advanced home delivery solution designed to optimize grocery and essential item deliveries. The system intelligently creates delivery trips, sequences shipments, and assigns vehicles to achieve key operational goals like maximizing vehicle utilization, minimizing trips, and reducing overall travel distances.

---

## Features

1. **Maximize Vehicle Capacity Utilization**  
   Efficiently uses vehicle space to minimize the number of vehicles required for delivery.

2. **Minimize the Number of Trips**  
   Optimizes shipment allocation to reduce the total number of delivery trips.

3. **Minimize the Overall Distance**  
   Plans routes to reduce the distance traveled by vehicles.

4. **Delivery Time Constraints**  
   Ensures trips are planned within acceptable time limits to meet user-specified delivery timeslots.

5. **Automated Vehicle Type Allocation**  
   Prioritizes sustainable vehicles (e.g., 3W and 4W-EV) for environmental efficiency.

6. **Real-Time Traffic Awareness (Optional)**  
   Supports dynamic routing with real-time traffic data using OSRM.

---

## Technologies Used

- **Python**: Core development language.  
- **Pandas**: Data preprocessing and manipulation.  
- **Open Source Routing Machine (OSRM)**: Accurate distance and duration estimation.  
- **Minimum Spanning Tree (MST)**: Approximate route calculation for computational efficiency.  
- **Excel Integration**: Data input/output for shipment and vehicle information.  

---

## Project Workflow

1. **Data Input & Preprocessing**  
   - Load shipment and vehicle data from Excel.  
   - Validate and clean the input data.

2. **Vehicle Selection & Prioritization**  
   - Assign shipments based on vehicle capacity.  
   - Prioritize 3W and 4W-EV for sustainability.  

3. **Route Optimization**  
   - Use MST for approximate route calculations.  
   - Leverage OSRM for accurate distance and duration estimation.  

4. **Clustering & Allocation**  
   - Cluster deliveries using MST constraints.  
   - Dynamically allocate shipments to available vehicles.  

5. **Final Route Generation & Export**  
   - Generate optimized delivery routes.  
   - Export results to Excel for tracking and reporting.  

---

## Why Choose SmartRoute Optimizer?

- **Efficiency**: Optimizes deliveries to minimize trips, distance, and costs.  
- **Sustainability**: Prioritizes environmentally friendly vehicles.  
- **OSRM Advantage**: Offers faster and more accurate routing compared to alternatives like Mapbox.  
- **Flexibility**: Adapts to custom time slots and real-time traffic (if enabled).  
- **Scalability**: Designed to handle a growing number of deliveries and routes.  

---

## Limitations

1. Real-time traffic support is optional and depends on third-party API availability.  
2. Computation time may increase for extremely large datasets with detailed OSRM routing.  
3. Requires well-structured input data for accurate optimization.  
4. Edge cases such as vehicle breakdowns or delivery failures are not fully handled.  

---

## Getting Started

### Prerequisites
- Python 3.9+
- Required libraries: `pandas`, `numpy`, `openpyxl`, `requests`, `osrm`

### Installation
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/smartroute-optimizer.git
   cd smartroute-optimizer
   
2. Install dependencies:  
      pip install -r requirements.txt
   
3. Configure the input Excel files for shipments and vehicles.

### Running the Code
python main.py

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have suggestions for improvement.

---

## Acknowledgments

- OSRM for its robust and accurate routing capabilities.  
- Open-source tools and the Python community for making this project possible.  

---
