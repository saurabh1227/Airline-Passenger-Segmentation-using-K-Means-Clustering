# Airline Passenger Segmentation using K-Means Clustering

Groups airline passengers into segments with similar service-rating patterns, and shows the result in a Streamlit web app.

**Author:** Saurabh Kumar

---

## a. Problem Statement

An airline collects ratings from its passengers for services such as seat comfort, inflight wifi, cleanliness and online booking. Looking at 130,000 individual passengers is not practical, and one average rating hides the fact that different passengers care about different things.

The goal of this project is to use **K-Means clustering** (unsupervised learning) to:

1. Divide passengers into a small number of segments based on how they rate the 14 services.
2. Describe each segment: which services it rates high or low, and how satisfied its passengers are.
3. Provide a Streamlit app where a user can segment a CSV of passengers, or place a single passenger into a segment.

The airline can use the segments to see which services need work for which group of passengers.

## b. Dataset Description

| Item | Details |
|---|---|
| Name | Invistico Airline passenger satisfaction data (`Invistico_Airline.csv`) |
| Size | 129,880 passengers x 23 columns |
| Features used for clustering (`X`) | The 14 service ratings, from `Seat comfort` to `Online boarding`. Each rating is 0 to 5, where 0 means the service was not applicable |
| Columns not used for clustering | `satisfaction`, `Gender`, `Customer Type`, `Age`, `Type of Travel`, `Class`, `Flight Distance`, delays. They are used only to describe the clusters |
| Missing values | Only `Arrival Delay in Minutes` (393 rows). It is not used for clustering |
| Satisfaction | 71,087 satisfied and 58,793 dissatisfied passengers |

The 14 ratings are: Seat comfort, Departure/Arrival time convenient, Food and drink, Gate location, Inflight wifi service, Inflight entertainment, Online support, Ease of Online booking, On-board service, Leg room service, Baggage handling, Checkin service, Cleanliness, Online boarding.

## c. Approach

1. **Select X:** the 14 rating columns.
2. **Scale with StandardScaler (z-score):** `z = (x - mean) / std`. K-Means is based on distances, so every feature must be on a comparable scale. After scaling, each rating has mean 0 and standard deviation 1.
3. **Choose K:** run K-Means for K = 1 to 10, plot the inertia (elbow plot) and detect the elbow. The silhouette score is also calculated (on a 10,000-passenger sample).
4. **Fit the final K-Means model** (`n_init=10`, `random_state=42`).
5. **Visualise** the clusters in 2D with PCA.
6. **Profile the clusters** with cluster-center bar charts, average ratings, age, satisfaction by travel class and by gender, and customer type.
7. **Save** the scaler, K-Means model, PCA and profile tables, and use them in the Streamlit app.

### Min-Max scaling vs Z-score scaling

| | Min-Max | Z-score (StandardScaler) |
|---|---|---|
| Formula | `(x - min) / (max - min)` | `(x - mean) / std` |
| Result | between 0 and 1 | mean 0, std 1 |
| Example: marks `[45, 78, 82, 70, 90, 88, 72, 87, 60, 85]` | 45 becomes 0, 90 becomes 1, 78 becomes 0.733 | mean is 75.7, 45 becomes -2.26, 78 becomes 0.17 |

This project uses z-score scaling. Because all 14 ratings are on the same 0 to 5 scale, Min-Max scaling gives very similar clusters here (Adjusted Rand Index 0.92 between the two results).

## d. Results

### Choosing K

| K | Inertia | Silhouette |
|---|---|---|
| 1 | 1,818,320 | - |
| 2 | 1,457,532 | 0.1955 |
| 3 | 1,285,509 | 0.1639 |
| **4** | **1,171,821** | **0.1623** |
| 5 | 1,114,426 | 0.1475 |
| 6 | 1,066,892 | 0.1421 |

The elbow of the inertia curve is at **K = 4**, so 4 clusters are used. The silhouette scores are low (about 0.15 to 0.20), which is common for rating data: the groups overlap and have no sharp boundary. K = 2 has the highest silhouette, but two groups give too little detail.

### The 4 segments

Numbers below come from the notebook run on this dataset with `random_state=42`. Cluster numbers can differ between runs, so each segment is described by its ratings.

| Segment | Passengers | Satisfied | What defines it |
|---|---|---|---|
| Happy with everything | 41,379 (31.9%) | 80.9% | Rates nearly every service high (mostly 3.7 to 4.3) |
| Good service, poor comfort | 29,650 (22.8%) | 73.1% | Rates online services, on-board service, baggage and cleanliness high (about 4) but seat comfort, schedule, food and gate location low (about 1.6 to 1.9) |
| Average, weak on-board basics | 30,156 (23.2%) | 33.5% | Medium ratings, with the lowest scores for on-board service, leg room, baggage and cleanliness (about 2.2 to 2.5) |
| Weak digital experience | 28,695 (22.1%) | 20.3% | Rates wifi, online support, online booking and online boarding lowest (about 1.7 to 2.0) |

### Observations

- Satisfaction was **not** used to build the clusters, yet it separates them clearly (20% to 81% satisfied). This shows that the clusters are meaningful.
- The two satisfied-leaning segments are about 57 to 60% Business class. The two weakest segments have about 30 to 40% Business class.
- The "weak digital experience" segment has the lowest share of loyal customers (67%, against 82 to 89% in the other segments) and the lowest satisfaction, so fixing wifi and online services looks like the most valuable improvement.
- Business class passengers are more often satisfied than Eco passengers in every segment (for example 91% vs 67% in the happiest segment, and 31% vs 16% in the weak digital segment).
- Women are more often satisfied than men in every segment (for example 86% vs 74% in the happiest segment, and 30% vs 14% in the weak digital segment).
- The weak digital segment is the youngest (average age about 35, against about 41 in the other three).
- The segments describe patterns in the data. They do not prove that a low rating is the cause of dissatisfaction.

## e. Streamlit App Features

- **Segment a CSV:** use the built-in sample data or upload a CSV with the 14 rating columns. Shows cluster sizes, a 2D PCA picture, the average rating and satisfaction per cluster, and lets you download the data with a cluster column.
- **Single passenger:** set the 14 ratings with sliders and see which cluster the passenger belongs to, with a comparison to the cluster average.
- **Model details:** the K selection table, elbow and silhouette plots, the cluster profile and a heatmap of the cluster centers.

## f. Project Structure

```
.
├── Airline_Passenger_Segmentation_KMeans.ipynb   # Analysis, training and app creation
├── app.py                                        # Streamlit app
├── utils.py                                      # Shared helper code (features, elbow, traits)
├── requirements.txt                              # Python packages
├── sample_data.csv                               # 2,000 passengers for the app demo
├── model/                                        # Saved scaler, K-Means, PCA and profile tables
│   ├── scaler.joblib
│   ├── kmeans.joblib
│   ├── pca.joblib
│   ├── meta.json
│   ├── cluster_profile.csv
│   └── k_selection.csv
└── README.md
```

The dataset `Invistico_Airline.csv` is needed only to run the notebook.

## g. How to Run (Windows)

**First time**

1. Put all project files and `Invistico_Airline.csv` in one simple folder without spaces, for example `C:\airline-segmentation`.
2. Open Command Prompt **inside that folder**: open the folder in File Explorer, click the address bar, type `cmd` and press Enter.
3. Install the packages:
   ```
   python -m pip install -r requirements.txt
   ```
4. Run the notebook from the top (`jupyter notebook`, then open `Airline_Passenger_Segmentation_KMeans.ipynb` and run all cells). It creates the `model/` folder, `sample_data.csv` and `app.py`.
5. Start the app:
   ```
   python -m streamlit run app.py
   ```
6. The app opens in the browser. If it does not, open `http://localhost:8501`.

**Next time:** only repeat steps 2 and 5.

**To stop the app:** press `Ctrl + C` in the Command Prompt window.

### Troubleshooting

| Error | What to do |
|---|---|
| `File does not exist: app.py` | You are not in the project folder. Use `cd` to go into it and check with `dir`. |
| `'streamlit' is not recognized` | Run it as `python -m streamlit run app.py`. |
| `No module named ...` | The packages are installed in a different Python. Run `python -m pip install -r requirements.txt` with the same Python you use to start the app. |
| `FileNotFoundError: model/kmeans.joblib` | Run the notebook first. It creates the `model/` folder. |
| Model fails to load | Install the same `scikit-learn` version the model was trained with and pin it in `requirements.txt`. |

## h. Limitations and Future Work

- K-Means assumes round, similar-sized groups. Ratings are whole numbers from 0 to 5, so groups overlap (low silhouette score).
- A rating of 0 ("not applicable") is treated as a normal value. Handling it separately could change the segments.
- Other clustering methods (for example Gaussian Mixture or hierarchical clustering) could be compared.
- Age, class, travel type and delays could be added as features, which needs encoding and different scaling.
