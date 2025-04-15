<<<<<<< HEAD
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import pandas as pd
from PIL import Image
import os
from geopy.distance import geodesic
from geopy.geocoders import Nominatim

# === Set your file paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PARTS_CSV = os.path.join(BASE_DIR, "C:/Users/sivab/OneDrive/Desktop/project/car parts.csv")
SHOPS_CSV = os.path.join(BASE_DIR, "C:/Users/sivab/OneDrive/Desktop/project/coimbatore_spare_parts_shops.csv")
MODEL_PATH = os.path.join(BASE_DIR,"C:/Users/sivab/OneDrive/Desktop/project/model.pth")

# === Load parts CSV ===
if not os.path.exists(PARTS_CSV):
    st.error(f"Parts dataset not found at {PARTS_CSV}")
    st.stop()

df = pd.read_csv(PARTS_CSV)


class_to_idx = {name: idx for idx, name in enumerate(df["labels"].unique())}
idx_to_class = {v: k for k, v in class_to_idx.items()}

# === Load model ===
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_to_idx))

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}")
    st.stop()

checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()

# === Image Transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Load shop CSV ===
if not os.path.exists(SHOPS_CSV):
    st.error(f"Shop dataset not found at {SHOPS_CSV}")
    st.stop()

shops_df = pd.read_csv(SHOPS_CSV)

# Check for required columns
if "latitude" not in shops_df.columns or "longitude" not in shops_df.columns:
    st.error("❌ Missing 'Latitude' or 'Longitude' columns in the shop dataset. Please check the CSV headers.")
    st.stop()

# === Streamlit UI ===
st.title("🔍 Car Part Identifier & Nearby Shop Finder")

uploaded_file = st.file_uploader("📸 Upload a car part image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=100)

    # Predict
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        predicted_part = idx_to_class[predicted.item()]

    st.success(f"🧩 Predicted Car Part: **{predicted_part}**")

    # Get location
    user_location = st.text_input("📍 Enter your location (e.g., Coimbatore, Peelamedu)")

    if user_location:
        geolocator = Nominatim(user_agent="car-part-app")
        location = geolocator.geocode(user_location)

        if location:
            user_coords = (location.latitude, location.longitude)
            st.map(pd.DataFrame([{"lat": location.latitude, "lon": location.longitude}]))

            st.markdown("### 🛠️ Nearby Repair Shops (within 80 km)")
            nearby_shops = []

            for _, shop in shops_df.iterrows():
                shop_coords = (shop["latitude"], shop["longitude"])  # Check column names
                distance_km = geodesic(user_coords, shop_coords).km
                if distance_km <= 80:
                    travel_time_min = distance_km / 40 * 60
                    nearby_shops.append({
                        "name": shop["name"],
                        "location": shop_coords,
                        "distance": distance_km,
                        "time": int(travel_time_min)
                    })

            if nearby_shops:
                for shop in sorted(nearby_shops, key=lambda x: x["distance"]):
                    st.markdown(f"""
                    - **{shop['name']}**
                      - 📍 Location: {shop['location']}
                      - 🛣️ Distance: `{shop['distance']:.2f} km`
                      - ⏱️ Estimated Time: `{shop['time']} minutes`
                    """)
            else:
                st.info("No nearby shops found within 80 km.")
        else:
            st.error("❌ Couldn't find your location. Try using a more specific name.")
=======
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import pandas as pd
from PIL import Image
import os
from geopy.distance import geodesic
from geopy.geocoders import Nominatim

# === Set your file paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PARTS_CSV = os.path.join(BASE_DIR, "C:/Users/sivab/OneDrive/Desktop/project/car parts.csv")
SHOPS_CSV = os.path.join(BASE_DIR, "C:/Users/sivab/OneDrive/Desktop/project/coimbatore_spare_parts_shops.csv")
MODEL_PATH = os.path.join(BASE_DIR,"C:/Users/sivab/OneDrive/Desktop/project/model.pth")

# === Load parts CSV ===
if not os.path.exists(PARTS_CSV):
    st.error(f"Parts dataset not found at {PARTS_CSV}")
    st.stop()

df = pd.read_csv(PARTS_CSV)


class_to_idx = {name: idx for idx, name in enumerate(df["labels"].unique())}
idx_to_class = {v: k for k, v in class_to_idx.items()}

# === Load model ===
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_to_idx))

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}")
    st.stop()

checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()

# === Image Transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Load shop CSV ===
if not os.path.exists(SHOPS_CSV):
    st.error(f"Shop dataset not found at {SHOPS_CSV}")
    st.stop()

shops_df = pd.read_csv(SHOPS_CSV)

# Check for required columns
if "latitude" not in shops_df.columns or "longitude" not in shops_df.columns:
    st.error("❌ Missing 'Latitude' or 'Longitude' columns in the shop dataset. Please check the CSV headers.")
    st.stop()

# === Streamlit UI ===
st.title("🔍 Car Part Identifier & Nearby Shop Finder")

uploaded_file = st.file_uploader("📸 Upload a car part image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=100)

    # Predict
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        predicted_part = idx_to_class[predicted.item()]

    st.success(f"🧩 Predicted Car Part: **{predicted_part}**")

    # Get location
    user_location = st.text_input("📍 Enter your location (e.g., Coimbatore, Peelamedu)")

    if user_location:
        geolocator = Nominatim(user_agent="car-part-app")
        location = geolocator.geocode(user_location)

        if location:
            user_coords = (location.latitude, location.longitude)
            st.map(pd.DataFrame([{"lat": location.latitude, "lon": location.longitude}]))

            st.markdown("### 🛠️ Nearby Repair Shops (within 80 km)")
            nearby_shops = []

            for _, shop in shops_df.iterrows():
                shop_coords = (shop["latitude"], shop["longitude"])  # Check column names
                distance_km = geodesic(user_coords, shop_coords).km
                if distance_km <= 80:
                    travel_time_min = distance_km / 40 * 60
                    nearby_shops.append({
                        "name": shop["name"],
                        "location": shop_coords,
                        "distance": distance_km,
                        "time": int(travel_time_min)
                    })

            if nearby_shops:
                for shop in sorted(nearby_shops, key=lambda x: x["distance"]):
                    st.markdown(f"""
                    - **{shop['name']}**
                      - 📍 Location: {shop['location']}
                      - 🛣️ Distance: `{shop['distance']:.2f} km`
                      - ⏱️ Estimated Time: `{shop['time']} minutes`
                    """)
            else:
                st.info("No nearby shops found within 80 km.")
        else:
            st.error("❌ Couldn't find your location. Try using a more specific name.")
>>>>>>> f03c02f0cd366e62466121a591648f4cc635b426
