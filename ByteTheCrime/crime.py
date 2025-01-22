import streamlit as st
import pydeck as pdk
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os
import random   
from io import BytesIO
import os
import random
import requests
from io import BytesIO
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats
import plotly.express as px


# Page Configuration
st.set_page_config(page_title="Byte the Crime", layout="centered")

# Initialize Session State
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "current_page" not in st.session_state:
    st.session_state.current_page = "login"
if "model_ready" not in st.session_state:
    st.session_state.model_ready = False
if "location_input" not in st.session_state:
    st.session_state.location_input = {}
if "recent_searches" not in st.session_state:
    st.session_state.recent_searches = []
def fetch_satellite_image(lon, lat, zoom=15, width=299, height=299, access_token=None):
    """Fetch satellite image from Mapbox Static API and return the image and source URL."""
    # Replace this with your valid access token
    if access_token is None:
        access_token = "pk.eyJ1IjoiZ3RyazI1IiwiYSI6ImNtNHB6cnd0MDAxNWMydW9jbTY0dXJ3cWEifQ.Vbd-lll_gjwgkGuujvqsvw"  # Replace with a valid token

    try:
        # Proper coordinate order: longitude, latitude
        url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{lon},{lat},{zoom},0/{width}x{height}?access_token={access_token}"
        response = requests.get(url)

        if response.status_code == 200:
            image = Image.open(BytesIO(response.content)).convert("RGB")

            return image, url
        elif response.status_code == 401:
            st.error("Unauthorized: Invalid or missing Mapbox Access Token. Verify your API key.")
        else:
            st.error(f"Failed to fetch image. Status code: {response.status_code}")
    except Exception as e:
        st.error(f"Error fetching satellite image: {str(e)}")
    return None, None
# Modified prediction function to use new imports
def predict_image(image_file, model, auxiliary_data=None):
    """Process image and auxiliary data, and make prediction using Keras model"""
    try:
        # Check if input is already an image (Pillow Image object)
        if isinstance(image_file, Image.Image):  # If image_file is already a Pillow Image
            img = image_file
        else:  # Otherwise, open the image
            img = Image.open(image_file).convert('RGB')

        # Resize and preprocess the image
        img = img.resize((299, 299))  # Resize to 128x128
        img_array = img_to_array(img)  # Convert image to array
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize and expand dims

        # Prepare auxiliary input (default to zeros if none provided)
        if auxiliary_data is None:
            auxiliary_data = np.zeros((1, 2))  # Default auxiliary input with shape (1, 2)
        else:
            auxiliary_data = np.array(auxiliary_data).reshape(1, -1)  # Reshape auxiliary input

        # Make prediction with both inputs
        prediction = model.predict([img_array, auxiliary_data], verbose=0)
        return float(prediction[0][0])

    except Exception as e:
        st.error(f"Error processing image: {e}")
        return 0.0

# Model loading function
@st.cache_resource
def load_keras_model(model_path):
    """Load the Keras model with proper error handling"""
    try:
        if not os.path.exists(model_path):
            st.error(f"Model file not found at {model_path}")
            return None
            
        # Load the Keras model
        model = load_model(model_path, compile=False)
        return model
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


# Load models at startup
# Load models at startup
try:
    MODEL_PATHS = {
        "Morning (6AM-12PM)": "D:\\CNN_MODEL_2.keras",
        "Afternoon (12PM-6PM)": "D:\\CNN_MODEL_3.keras",
        "Evening (6PM-12AM)": "D:\\CNN_MODEL_4.keras",
        "Night (12AM-6AM)": "D:\\CNN_MODEL_1.keras",
    }

    MODELS = {}

    for time_period, model_path in MODEL_PATHS.items():
        model = load_keras_model(model_path)
        
        if model is not None:
            MODELS[time_period] = model
        else:
            st.warning(f"Failed to load model for {time_period}")

    st.session_state.model_ready = len(MODELS) > 0

except Exception as e:
    st.error(f"Error during model initialization: {str(e)}")
    MODELS = {}
    st.session_state.model_ready = False

def predict_crime_probability(model, image_file):
    """Main prediction function"""
    try:
        probability = predict_image(image_file, model)
        return probability
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 0.0

def parse_coordinates(coord_str):
    """Parse coordinates from string input"""
    try:
        lon, lat = map(float, coord_str.strip().split(','))
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon
        else:
            raise ValueError("Coordinates out of valid range")
    except Exception as e:
        raise ValueError(f"Invalid coordinate format: {e}")

def create_heatmap_visualization(points):
    """Create a heatmap visualization with gradient colors based on risk levels"""
    df = pd.DataFrame(points)
    
    return pdk.Layer(
        "HeatmapLayer",
        data=df,
        get_position=["lon", "lat"],
        get_weight="probability",
        radiusPixels=50,  # Increased for better visibility
        intensity=1.5,     # Increased for better visibility
        threshold=0.05,    # Lower threshold for better visibility
        colorRange=[
            [144, 238, 144],  # Light green
            [255, 255, 0],    # Yellow
            [255, 0, 0]       # Red
        ]
    )

# Valid users for login
VALID_USERS = [
    {"username": "admin", "password": "admin"}
]
# CSS for styling
st.markdown("""
<style>
    .stApp {
        background-color: #f4f4f4;
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
    }

    .login-header {
        margin-bottom: 20px;
    }

    .login-header h1 {
        margin-bottom: 10px;
        color: #333;
    }

    .login-header p {
        color: #666;
        margin-bottom: 20px;
    }

    .stTextInput > div > div > input {
        width: 100%;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 4px;
    }

    .stButton > button {
        width: 100%;
        padding: 10px;
        background-color: #007bff !important;
        color: white !important;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        margin-top: 10px;
    }

    .forgot-password {
        display: block;
        margin-top: 15px;
        color: #007bff;
        text-decoration: none;
    }
</style>
""", unsafe_allow_html=True)

# Session state
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "login"
if "model_ready" not in st.session_state:
    st.session_state["model_ready"] = True


# Login Page
def login_page():
    st.markdown('<div class="login-page">', unsafe_allow_html=True)
    
    # Login Header
    st.markdown('<h1>Byte the Crime</h1>', unsafe_allow_html=True)
    st.markdown('<p>Track and manage crime statistics effectively.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    
    # Login Form
    with st.form(key='login_form'):
        email = st.text_input("Email or Phone", placeholder="Enter email or phone")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        login_button = st.form_submit_button("Log In")
        
        if login_button:
            # Validate credentials
            user = next((u for u in VALID_USERS if u['username'] == email and u['password'] == password), None)
            
            if user:
                st.session_state["authenticated"] = True
                st.session_state["current_page"] = "home"
                st.session_state["username"] = email
            else:
                st.error("Incorrect Credentials")
    
    # Forgot Password and Create Account Links
    st.markdown('<a href="#" class="forgot-password">Forgot Password?</a>', unsafe_allow_html=True)
    st.markdown('<hr>', unsafe_allow_html=True)
    
    if st.button("Create New Account"):
        st.warning("Account creation functionality to be implemented.")

    st.markdown('</div>', unsafe_allow_html=True)

# Home Page
def home_page():
    st.title("Crime Prediction Analysis")
    st.write("Enter location details to predict crime hotspots.")
    
    # Initialize locations in session state if not exists
def home_page():
    st.title("Crime Prediction Analysis")
    st.write("Enter location details to predict crime hotspots.")
    
    # Initialize locations in session state if not exists
    if "locations" not in st.session_state:
        st.session_state["locations"] = [""]

    # Add/Remove location buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Add Location"):
            st.session_state["locations"].append("")
    with col2:
        if st.button("Remove Last Location") and len(st.session_state["locations"]) > 1:
            st.session_state["locations"].pop()

    # Dynamically render input fields for each coordinate
    with st.form("location_form"):
        st.subheader("Enter Coordinates (Longitude, Latitude)")
        
        # Time period selection
        time_period = st.selectbox(
            "Select Time Period",
            ["Morning (6AM-12PM)", "Afternoon (12PM-6PM)", "Evening (6PM-12AM)", "Night (12AM-6AM)"]
        )
        
        # Coordinate inputs
        for i in range(len(st.session_state["locations"])):
            st.session_state["locations"][i] = st.text_input(
                f"Location {i+1} Coordinates",
                value=st.session_state["locations"][i],
                placeholder="-71.07292311707977, 42.33615566349997",
                key=f"location_{i}"
            )

        submit = st.form_submit_button("Analyze Locations")

        if submit:
            valid_coordinates = []
            for loc in st.session_state["locations"]:
                if loc.strip():  # Only process non-empty inputs
                    try:
                        lon, lat = map(float, loc.strip().split(","))
                        if -90 <= lat <= 90 and -180 <= lon <= 180:
                            valid_coordinates.append({"lon": lon, "lat": lat})
                        else:
                            st.error(f"Coordinates {loc} are out of valid range.")
                    except Exception:
                        st.error(f"Invalid format for coordinates: {loc}")

            if valid_coordinates:
                predictions = []
                for coord in valid_coordinates:
                    lon, lat = coord["lon"], coord["lat"]
                    image, url = fetch_satellite_image(lon, lat)

                    if image:
                        st.image(image, caption=f"Satellite Image for ({lat}, {lon})", use_container_width=True)
                        
                        model = MODELS.get(time_period)
                        if model:
                            try:
                                # Process image for prediction
                                img_array = img_to_array(image)
                                img_array = img_array.reshape((1, 299, 299, 3)) / 255.0
                                
                                # Create dummy coordinates input
                                aux_input = np.zeros((1, 2), dtype=np.float32)
                                
                                # Make prediction with both inputs
                                prediction = model.predict([img_array, aux_input], verbose=0)
                                prediction_value = abs(float(prediction[0][0]))

                                predictions.append({
                                    "lat": lat,
                                    "lon": lon,
                                    "probability": prediction_value
                                })
                            except Exception as e:
                                st.error(f"Error making prediction: {str(e)}")
                        else:
                            st.error(f"No model available for {time_period}")
                    else:
                        st.error(f"Failed to fetch satellite image for ({lat}, {lon})")

                if predictions:
                    st.session_state["location_input"] = {
                        "coordinates": predictions,
                        "time_period": time_period
                    }
                    st.session_state.current_page = "output"
                    st.rerun()
            else:
                st.error("No valid coordinates provided. Please check your input.")

def output_page():
    st.title("Crime Hotspot Analysis")

    if not st.session_state.get("model_ready", False):
        st.error("Model is not loaded. Please restart the application.")
        return

    data = st.session_state.get("location_input", {})
    if not data or "coordinates" not in data:
        st.error("No input data found.")
        return

    try:
        all_points = []  # Initialize list to store predictions for heatmap

        # Default time period from session or user selection
        current_time = data.get("time_period", "Morning (6AM-12PM)")
        time_period = st.selectbox(
            "Select Time Period for Prediction",
            ["Morning (6AM-12PM)", "Afternoon (12PM-6PM)", "Evening (6PM-12AM)", "Night (12AM-6AM)"],
            index=["Morning (6AM-12PM)", "Afternoon (12PM-6PM)", "Evening (6PM-12AM)", "Night (12AM-6AM)"].index(current_time)
        )

        # Recalculate prediction if time period changes
        if time_period != current_time:
            updated_predictions = []
            for coord in data["coordinates"]:
                lon, lat = coord["lon"], coord["lat"]
                image, _ = fetch_satellite_image(lon, lat)
                
                if image:
                    model = MODELS.get(time_period)
                    if model:
                        # Process image for prediction
                        img_array = img_to_array(image)
                        img_array = img_array.reshape((1, 299, 299, 3)) / 255.0
                        
                        # Create dummy coordinates input
                        aux_input = np.zeros((1, 2), dtype=np.float32)
                        
                        # Make prediction with both inputs
                        prediction = model.predict([img_array, aux_input], verbose=0)
                        prediction_value = abs(float(prediction[0][0]))
                        
                        updated_predictions.append({
                            "lat": lat,
                            "lon": lon,
                            "probability": prediction_value
                        })
            
            data["coordinates"] = updated_predictions
            data["time_period"] = time_period
            st.session_state.location_input = data
            st.rerun()

        # Process all coordinates
        for coord in data["coordinates"]:
            all_points.append(coord)

        # Display overall risk assessment
        st.header("Overall Risk Assessment")
        if all_points:
            risk_values = [point["probability"] for point in all_points]
            avg_risk = np.mean(risk_values)
            std_dev = np.std(risk_values)

            def get_risk_info(risk_value):
                if risk_value < 0.2:
                    return {
                        "level": "Very Low",
                        "color": "#00FF00",
                        "bg_color": "rgba(0, 255, 0, 0.1)"
                    }
                elif risk_value < 0.4:
                    return {
                        "level": "Low",
                        "color": "#90EE90",
                        "bg_color": "rgba(144, 238, 144, 0.1)"
                    }
                elif risk_value < 0.6:
                    return {
                        "level": "Moderate",
                        "color": "#FFFF00",
                        "bg_color": "rgba(255, 255, 0, 0.1)"
                    }
                elif risk_value < 0.8:
                    return {
                        "level": "High",
                        "color": "#FFA500",
                        "bg_color": "rgba(255, 165, 0, 0.1)"
                    }
                else:
                    return {
                        "level": "Very High",
                        "color": "#FF0000",
                        "bg_color": "rgba(255, 0, 0, 0.1)"
                    }

            risk_info = get_risk_info(avg_risk)
            
            st.markdown(f"""
                <div style='
                    padding: 20px;
                    border-radius: 10px;
                    background-color: {risk_info['bg_color']};
                    border: 2px solid {risk_info['color']};
                    margin-bottom: 20px;
                '>
                    <h3 style='color: black; margin: 0;'>
                        Overall Density Score: {avg_risk:.3f}
                    </h3>
                    <p style='margin: 10px 0 0 0; color: black;'>
                        Based on {len(risk_values)} location{'' if len(risk_values) == 1 else 's'}<br>
                        Risk Level: {risk_info['level']}<br>
                        Click 'View Analytics' for detailed analysis
                    </p>
                </div>
            """, unsafe_allow_html=True)

        # Generate heatmap for multiple coordinates
        if len(all_points) > 1:
            df = pd.DataFrame(all_points)
            
            # Create heatmap layer
            heatmap_layer = pdk.Layer(
                "HeatmapLayer",
                data=df,
                get_position=["lon", "lat"],
                get_weight="probability",
                radiusPixels=60,
                intensity=1.0,
                threshold=0.0,
                colorRange=[
                    [0, 255, 0, 50],     # Green
                    [144, 238, 144, 100], # Light green
                    [255, 255, 0, 150],   # Yellow
                    [255, 165, 0, 200],   # Orange
                    [255, 0, 0, 255]      # Red
                ]
            )

            # Map visualization
            st.pydeck_chart(pdk.Deck(
                map_style="mapbox://styles/mapbox/satellite-v9",
                initial_view_state=pdk.ViewState(
                    latitude=df["lat"].mean(),
                    longitude=df["lon"].mean(),
                    zoom=11,
                    pitch=45,
                ),
                layers=[heatmap_layer]
            ))

        elif len(all_points) == 1:  # Single coordinate
            point = all_points[0]
            image, _ = fetch_satellite_image(point["lon"], point["lat"])
            if image:
                st.image(image, caption=f"Satellite Image for ({point['lat']}, {point['lon']})", use_container_width=True)
                
                # Get risk info for single point
                point_risk_info = get_risk_info(point['probability'])
                st.markdown(f"""
                    <div style='
                        padding: 10px;
                        border-radius: 5px;
                        background-color: {point_risk_info['bg_color']};
                        border: 1px solid {point_risk_info['color']};
                    '>
                        Density Score: {point['probability']:.3f} ({point_risk_info['level']})
                    </div>
                """, unsafe_allow_html=True)

        # Show location details
        st.subheader("Prediction Details")
        for idx, point in enumerate(all_points, 1):
            with st.expander(f"Location {idx}"):
                risk_info = get_risk_info(point['probability'])
                st.write(f"Coordinates: ({point['lat']:.4f}, {point['lon']:.4f})")
                st.markdown(f"""
                    <div style='
                        padding: 5px;
                        border-radius: 5px;
                        background-color: {risk_info['bg_color']};
                        border: 1px solid {risk_info['color']};
                    '>
                        Density Score: {point['probability']:.3f}<br>
                        Risk Level: {risk_info['level']}
                    </div>
                """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("View Analytics", use_container_width=True):
            st.session_state.current_page = "analytics"
            st.rerun()
    with col2:
        if st.button("Back to Home", use_container_width=True):
            st.session_state.current_page = "home"
            st.rerun()
# Helper function for risk level (should be accessible to both output and analytics pages)
def get_risk_level(score):
    if score < 0.2:
        return ("Very Low", "Historically averages lower number of incidents monthly")
    elif score < 0.4:
        return ("Low", "Historically averages below average number of incidents monthly")
    elif score < 0.6:
        return ("Moderate", "Historically averages moderate number of incidents monthly")
    elif score < 0.8:
        return ("High", "Historically averages above average number of incidents monthly")
    else:
        return ("Very High", "Historically averages high number of incidents monthly")

def update_analytics_page():
    st.title("Crime Analysis Details")
    if "location_input" not in st.session_state:
        st.error("No data to analyze")
        return
        
    data = st.session_state.location_input
    locations = data["coordinates"]
    time_period = data["time_period"]
    
    # Load density bin data from CSV
    density_df = pd.read_csv("res.csv")
    
    # Calculate overall metrics using bandwidth estimation
    probabilities = [coord["probability"] for coord in locations]
    bandwidth_score = 0.004  # Bandwidth parameter for spatial pattern estimation
    
    # Helper functions remain the same...
    def process_density_scores(scores, bandwidth=0.004):
        adjusted_scores = []
        for score in scores:
            neighbor_influence = score * bandwidth
            adjusted_score = score + neighbor_influence
            adjusted_scores.append(adjusted_score)
        return adjusted_scores, neighbor_influence

    def analyze_density_pattern(score):
        for _, row in density_df.iterrows():
            try:
                bin_values = row["density_bin"].replace("(", "").replace("]", "").split(",")
                min_val = float(bin_values[0].strip())
                max_val = float(bin_values[1].strip())
                
                if min_val <= score <= max_val:
                    crime_probs = {}
                    total_inverse = 0
                    counts = []
                    
                    for col in row.index:
                        if col != "density_bin":
                            try:
                                count = float(col)
                                inverse_prob = 1 / (count + 1)
                                counts.append((count, inverse_prob))
                                total_inverse += inverse_prob
                            except ValueError:
                                continue
                    
                    for count, inverse_prob in counts:
                        normalized_prob = inverse_prob / total_inverse
                        crime_probs[count] = normalized_prob
                    
                    return crime_probs, min_val, max_val
            except ValueError as e:
                st.error(f"Error processing density bin: {e}")
        return None, None, None

    # Display overall analysis section remains the same...
    st.header("1. Spatial Pattern Understanding")
    
    with st.expander("📚 How This Analysis Works", expanded=True):
        st.write("""
        **Key Concepts:**
        1. **Density Patterns**: Analysis is based on observed density patterns, not predictions
        2. **Spatial Relationships**: Each location is influenced by its neighboring areas
        3. **Bandwidth Effect**: Uses a 0.004 bandwidth to account for spatial relationships
        
        **What This Means:**
        - Higher crime counts have naturally lower probabilities of occurrence
        - Patterns are based on actual observations, not hypothetical predictions
        - Each location's analysis considers its surrounding area
        """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        adjusted_scores, neighbor_effect = process_density_scores(probabilities)
        avg_pattern = np.mean(adjusted_scores)
        st.metric("Pattern Intensity", f"{avg_pattern:.4f}")
        st.caption("Based on spatial bandwidth estimation")
        
    with col2:
        st.metric("Bandwidth Parameter", f"{bandwidth_score}")
        st.caption("Spatial relationship factor")
        
    with col3:
        st.metric("Time Period", time_period)
        st.caption("Analysis timeframe")

    # Individual location analysis with unique keys for plotly charts
    st.header("2. Location-Specific Analysis")
    
    for idx, location in enumerate(locations, 1):
        with st.expander(f"Location {idx} Pattern Analysis"):
            loc_score = location["probability"]
            adjusted_scores, neighbor_influence = process_density_scores([loc_score])
            adjusted_score = adjusted_scores[0]
            
            st.write(f"**Coordinates**: ({location['lat']:.4f}, {location['lon']:.4f})")
            
            st.subheader("Spatial Influence Analysis")
            cols = st.columns(2)
            with cols[0]:
                st.metric("Base Pattern", f"{loc_score:.4f}")
                st.caption("Original density score")
            with cols[1]:
                st.metric("Neighbor Influence", f"{neighbor_influence:.4f}")
                st.caption("Effect of surrounding areas")
            
            st.write("**Spatial Relationship Visualization**")
            df = pd.DataFrame({
                'Distance': ['Core Location', 'Neighbor Influence'],
                'Value': [loc_score, neighbor_influence]
            })
            
            # Add unique key for the first plotly chart
            fig = px.bar(df, x='Distance', y='Value',
                        title="Pattern Composition",
                        labels={'Value': 'Intensity'},
                        color='Distance',
                        color_discrete_map={
                            'Core Location': 'rgb(99,110,250)',
                            'Neighbor Influence': 'rgb(239,85,59)'
                        })
            st.plotly_chart(fig, use_container_width=True, key=f"pattern_comp_{idx}")
            
            crime_probs, min_density, max_density = analyze_density_pattern(loc_score)
            
            if crime_probs:
                st.subheader("Density Pattern Distribution")
                st.write(f"Density Range: {min_density:.4f} - {max_density:.4f}")
                
                crime_df = pd.DataFrame({
                    "Crime Count": [f"{count:.1f}" for count in crime_probs.keys()],
                    "Probability": crime_probs.values()
                })
                
                crime_df["Crime Count Numeric"] = crime_df["Crime Count"].astype(float)
                crime_df = crime_df.sort_values("Crime Count Numeric")
                
                # Add unique key for the second plotly chart
                fig = px.line(
                    crime_df,
                    x="Crime Count Numeric",
                    y="Probability",
                    title="Crime Count Probability Distribution",
                    labels={
                        "Crime Count Numeric": "Number of Incidents",
                        "Probability": "Probability of Occurrence"
                    },
                )
                
                fig.update_traces(mode='lines+markers')
                fig.update_layout(showlegend=False)
                
                st.plotly_chart(fig, use_container_width=True, key=f"prob_dist_{idx}")
                
                display_df = crime_df[["Crime Count", "Probability"]].copy()
                display_df["Probability"] = display_df["Probability"].apply(lambda x: f"{x:.2%}")
                st.dataframe(display_df, hide_index=True, key=f"prob_table_{idx}")
                
                st.write("**Pattern Interpretation**")
                st.write(f"""
                - This location shows a base pattern intensity of {loc_score:.4f}
                - Neighboring areas contribute an additional {neighbor_influence:.4f} to the pattern
                - The pattern falls within a density range of {min_density:.4f} to {max_density:.4f}
                - Lower incident counts show higher probabilities, following observed patterns
                - The analysis reflects actual density patterns, not predictions
                """)
            else:
                st.warning("Could not analyze pattern for this location.")

    # Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to Results", use_container_width=True):
            st.session_state.current_page = "output"
            st.rerun()
    with col2:
        if st.button("New Analysis", use_container_width=True):
            st.session_state.current_page = "home"
            st.rerun()
# Main Application
def main():
    if st.session_state.authenticated:
        if st.session_state.current_page == "home":
            home_page()
        elif st.session_state.current_page == "output":
            output_page()
        elif st.session_state.current_page == "analytics":
            update_analytics_page()
    else:
        login_page()

if __name__ == "__main__":
    main()