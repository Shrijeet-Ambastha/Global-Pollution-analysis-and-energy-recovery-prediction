import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

st.title("Global Pollution Analysis & Energy Recovery Prediction")

# Load dataset automatically
df = pd.read_csv("Global_Pollution_Analysis.csv")

# =========================
# PHASE 1: PREPROCESSING
# =========================
df.fillna(df.mean(numeric_only=True), inplace=True)

le = LabelEncoder()
df['Country'] = le.fit_transform(df['Country'])

df['Energy_per_Capita'] = (
    df['Energy_Consumption_Per_Capita (in MWh)'] /
    df['Population (in millions)']
)

df['Yearly_Pollution_Index'] = df[
    ['Air_Pollution_Index', 'Water_Pollution_Index', 'Soil_Pollution_Index']
].mean(axis=1)

st.subheader("Dataset Preview")
st.dataframe(df.head())

# =========================
# CLUSTERING
# =========================
feature_cols = [
    'Air_Pollution_Index',
    'Water_Pollution_Index',
    'Soil_Pollution_Index',
    'Energy_Consumption_Per_Capita (in MWh)',
    'Energy_Recovered (in GWh)'
]

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[feature_cols])

st.header("Clustering Analysis")

# Elbow Method
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    inertias.append(kmeans.inertia_)

fig1, ax1 = plt.subplots()
ax1.plot(range(1, 11), inertias, marker='o')
ax1.set_xlabel("Number of Clusters")
ax1.set_ylabel("Inertia")
ax1.set_title("Elbow Method")
st.pyplot(fig1)

# KMeans
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(scaled_features)

df['KMeans_Cluster'] = cluster_labels

# PCA Visualization
pca = PCA(n_components=2)
components = pca.fit_transform(scaled_features)

fig2, ax2 = plt.subplots()
scatter = ax2.scatter(
    components[:, 0],
    components[:, 1],
    c=cluster_labels,
    cmap='viridis'
)
ax2.set_xlabel("Principal Component 1")
ax2.set_ylabel("Principal Component 2")
ax2.set_title("K-Means Clustering Visualization")
plt.colorbar(scatter)
st.pyplot(fig2)

# Hierarchical Clustering
linkage_matrix = linkage(scaled_features, method='ward')

fig3, ax3 = plt.subplots(figsize=(10, 6))
dendrogram(linkage_matrix)
ax3.set_title("Hierarchical Clustering Dendrogram")
ax3.set_xlabel("Countries")
ax3.set_ylabel("Distance")
st.pyplot(fig3)

agg = AgglomerativeClustering(n_clusters=3)
agg_labels = agg.fit_predict(scaled_features)

df['Hierarchical_Cluster'] = agg_labels

st.subheader("Cluster Results")
st.dataframe(df[['Country', 'KMeans_Cluster', 'Hierarchical_Cluster']].head(10))

# =========================
# REGRESSION PREDICTION
# =========================
st.header("Energy Recovery Prediction")

X = df[
    [
        'Air_Pollution_Index',
        'Water_Pollution_Index',
        'Soil_Pollution_Index',
        'CO2_Emissions (in MT)',
        'Industrial_Waste (in tons)',
        'Energy_Consumption_Per_Capita (in MWh)'
    ]
]

y = df['Energy_Recovered (in GWh)']

scaler_nn = MinMaxScaler()
X_scaled = scaler_nn.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

st.subheader("Linear Regression Results")
st.write("R² Score:", r2_score(y_test, lr_pred))
st.write("MAE:", mean_absolute_error(y_test, lr_pred))
st.write("MSE:", mean_squared_error(y_test, lr_pred))