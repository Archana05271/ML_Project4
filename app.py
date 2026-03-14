import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import folium
from streamlit_folium import st_folium
import plotly.express as px

# --- Streamlit Page Setup ---
st.set_page_config(page_title="CrimeWatch", layout="wide")
st.title("🚨 CrimeWatch – Crime Hotspot Detection")

# --- Upload CSV ---
uploaded_file = st.file_uploader("Upload CSV file (must include 'Latitude' and 'Longitude')", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    # --- Check for Latitude & Longitude ---
    if 'Latitude' not in data.columns or 'Longitude' not in data.columns:
        st.error("Dataset must include 'Latitude' and 'Longitude' columns.")
    else:
        # --- Features for clustering ---
        features = ['Latitude','Longitude']
        X = data[features]
        X_scaled = StandardScaler().fit_transform(X)

        # --- PCA for 2D visualization ---
        X_pca = PCA(n_components=2).fit_transform(X_scaled)

        # --- DBSCAN Clustering ---
        cluster_model = DBSCAN(eps=0.5, min_samples=2)
        clusters = cluster_model.fit_predict(X_pca)
        data['Cluster'] = clusters

        st.subheader("Cluster Counts")
        st.write(data['Cluster'].value_counts())

        # --- 2D Cluster Plot ---
        fig = px.scatter(
            x=X_pca[:,0], y=X_pca[:,1],
            color=data['Cluster'].astype(str),
            labels={'x':'PCA1','y':'PCA2','color':'Cluster'},
            title="Clusters Visualized using PCA"
        )
        st.plotly_chart(fig)

        # --- Map Visualization ---
        map_center = [data['Latitude'].mean(), data['Longitude'].mean()]
        crime_map = folium.Map(location=map_center, zoom_start=5)
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cadetblue', 'pink']

        for idx, row in data.iterrows():
            cluster = row['Cluster']
            color = colors[int(cluster) % len(colors)] if cluster != -1 else 'black'
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=6,
                color=color,
                fill=True,
                popup=f"City: {row.get('City','N/A')} | Crime Type: {row.get('Crime_Type','N/A')}"
            ).add_to(crime_map)

        st.subheader("Crime Hotspot Map")
        st_folium(crime_map, width=700, height=500)

        # --- Download CSV ---
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV with Clusters",
            data=csv,
            file_name='crime_clusters.csv',
            mime='text/csv'
        )

else:
    st.info("Please upload a CSV file with 'Latitude' and 'Longitude'.")