


🛠️ Car Part Identifier & Nearby Repair Shop Finder

This application allows users to upload an image of a car part, which is then classified using a ResNet18-based deep learning model.
Once identified, the app suggests nearby car repair and spare parts shops within an 80 km radius, using the user's real-time location via Nominatim.

✨ Features

📷 Car Part Detection: Upload a car part image → classified using a trained ResNet18 model.

🏪 Nearby Shops Finder: Suggests nearby car repair/spare part shops within 80 km radius.

📍 Real-Time Geolocation: Uses Nominatim + geopy for location search.

🌐 Interactive UI: Built with Streamlit for ease of use.

🛠️ Tech Stack

Deep Learning: PyTorch (ResNet18)

Frontend: Streamlit

Geolocation: Nominatim API, Geopy

Data:

Car parts dataset (/train folder with labeled images)

CSV file of shops (name, latitude, longitude, category)
