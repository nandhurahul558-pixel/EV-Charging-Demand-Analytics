from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import math


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "gradient_boosting_ev_demand_model.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "ev_charging_data.csv"
)


# ============================================================
# LOAD MODEL
# ============================================================

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print("Model loading error:", e)
    model = None


# ============================================================
# LOAD DATASET
# ============================================================

try:
    df = pd.read_csv(DATA_PATH)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values(
        ["station_id", "timestamp"]
    ).reset_index(drop=True)

    print("Dataset loaded successfully.")
    print("Dataset shape:", df.shape)

except Exception as e:
    print("Dataset loading error:", e)
    df = pd.DataFrame()


# ============================================================
# STATION INFORMATION
# ============================================================

STATIONS = {
    "ST001": {
        "location": "Kochi",
        "chargers": 8
    },
    "ST002": {
        "location": "Trivandrum",
        "chargers": 12
    },
    "ST003": {
        "location": "Kottayam",
        "chargers": 10
    },
    "ST004": {
        "location": "Kozhikode",
        "chargers": 15
    },
    "ST005": {
        "location": "Thrissur",
        "chargers": 6
    },
    "ST006": {
        "location": "Kannur",
        "chargers": 10
    },
    "ST007": {
        "location": "Alappuzha",
        "chargers": 14
    },
    "ST008": {
        "location": "Kollam",
        "chargers": 8
    }
}


# ============================================================
# MODEL PERFORMANCE
# ============================================================

MODEL_NAME = "Gradient Boosting Regressor"

TEST_MAE = 34.04
TEST_RMSE = 47.53
TEST_R2 = 0.4910

TRAINING_SAMPLES = 139008


# ============================================================
# CAPACITY PLANNING ASSUMPTIONS
# ============================================================

CHARGER_CAPACITY_KWH = 20.0

TARGET_UTILIZATION = 0.80

EFFECTIVE_CHARGER_CAPACITY = (
    CHARGER_CAPACITY_KWH * TARGET_UTILIZATION
)


# ============================================================
# DATASET ANALYTICS
# ============================================================

def generate_analytics():

    if df.empty:
        return {
            "total_records": 0,
            "station_count": 0,
            "average_demand": 0,
            "max_demand": 0,
            "peak_hours": [],
            "station_comparison": []
        }

    total_records = len(df)

    station_count = df["station_id"].nunique()

    average_demand = df["energy_consumed_kwh"].mean()

    max_demand = df["energy_consumed_kwh"].max()


    # --------------------------------------------------------
    # Peak hours
    # --------------------------------------------------------

    hourly_demand = (
        df.groupby("hour")["energy_consumed_kwh"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
    )

    peak_hours = []

    for hour, demand in hourly_demand.items():

        peak_hours.append({
            "hour": f"{int(hour):02d}:00",
            "demand": round(float(demand), 2)
        })


    # --------------------------------------------------------
    # Station comparison
    # --------------------------------------------------------

    station_avg = (
        df.groupby("station_id")["energy_consumed_kwh"]
        .mean()
        .sort_values(ascending=False)
    )

    station_comparison = []

    rank = 1

    for station_id, avg_demand in station_avg.items():

        station_info = STATIONS.get(
            station_id,
            {
                "location": "Unknown",
                "chargers": 0
            }
        )

        station_comparison.append({
            "rank": rank,
            "station_id": station_id,
            "location": station_info["location"],
            "chargers": station_info["chargers"],
            "average_demand": round(
                float(avg_demand),
                2
            )
        })

        rank += 1


    return {
        "total_records": total_records,
        "station_count": station_count,
        "average_demand": round(
            float(average_demand),
            2
        ),
        "max_demand": round(
            float(max_demand),
            2
        ),
        "peak_hours": peak_hours,
        "station_comparison": station_comparison
    }


# ============================================================
# DEMAND CLASSIFICATION
# ============================================================

def classify_demand(demand):

    if demand < 50:
        return "Low Demand"

    elif demand < 100:
        return "Moderate Demand"

    elif demand < 200:
        return "High Demand"

    else:
        return "Very High Demand"


# ============================================================
# AUTOMATIC HISTORICAL FEATURES
# ============================================================

def get_historical_features(
    station_id,
    selected_timestamp
):

    station_data = df[
        df["station_id"] == station_id
    ].copy()

    if station_data.empty:

        return {
            "previous_hour_demand": 0,
            "previous_day_demand": 0,
            "previous_week_demand": 0,
            "rolling_24h_mean": 0,
            "rolling_7d_mean": 0,
            "source": "No historical data available"
        }


    station_data = station_data.sort_values(
        "timestamp"
    )


    # --------------------------------------------------------
    # Exact timestamp
    # --------------------------------------------------------

    exact_row = station_data[
        station_data["timestamp"] == selected_timestamp
    ]


    # --------------------------------------------------------
    # If selected timestamp exists in dataset
    # --------------------------------------------------------

    if not exact_row.empty:

        row = exact_row.iloc[0]

        return {
            "previous_hour_demand": round(
                float(row["previous_hour_demand"]),
                2
            ),

            "previous_day_demand": round(
                float(row["previous_day_demand"]),
                2
            ),

            "previous_week_demand": round(
                float(row["previous_week_demand"]),
                2
            ),

            "rolling_24h_mean": round(
                float(row["rolling_24h_mean"]),
                2
            ),

            "rolling_7d_mean": round(
                float(row["rolling_7d_mean"]),
                2
            ),

            "source": "Historical dataset"
        }


    # --------------------------------------------------------
    # Future timestamp
    #
    # If the selected date is outside the dataset,
    # estimate historical features using station history.
    # --------------------------------------------------------

    target_hour = selected_timestamp.hour

    target_day_of_week = selected_timestamp.dayofweek


    # Same station + same hour
    same_hour = station_data[
        station_data["hour"] == target_hour
    ]


    # Same station + same weekday + same hour
    same_pattern = station_data[
        (
            station_data["hour"] == target_hour
        )
        &
        (
            station_data["day_of_week"]
            == target_day_of_week
        )
    ]


    if same_pattern.empty:
        same_pattern = same_hour


    if same_pattern.empty:
        same_pattern = station_data


    historical_values = (
        same_pattern["energy_consumed_kwh"]
    )


    previous_hour = historical_values.mean()

    previous_day = historical_values.mean()

    previous_week = historical_values.mean()

    rolling_24h = historical_values.mean()

    rolling_7d = historical_values.mean()


    return {
        "previous_hour_demand": round(
            float(previous_hour),
            2
        ),

        "previous_day_demand": round(
            float(previous_day),
            2
        ),

        "previous_week_demand": round(
            float(previous_week),
            2
        ),

        "rolling_24h_mean": round(
            float(rolling_24h),
            2
        ),

        "rolling_7d_mean": round(
            float(rolling_7d),
            2
        ),

        "source": "Historical pattern estimate"
    }


# ============================================================
# CAPACITY PLANNING
# ============================================================

def calculate_capacity_plan(
    predicted_demand,
    current_chargers
):

    required_chargers = math.ceil(
        predicted_demand
        / EFFECTIVE_CHARGER_CAPACITY
    )


    additional_chargers = max(
        0,
        required_chargers - current_chargers
    )


    spare_chargers = max(
        0,
        current_chargers - required_chargers
    )


    current_capacity = (
        current_chargers
        * EFFECTIVE_CHARGER_CAPACITY
    )


    estimated_utilization = (
        predicted_demand
        / current_capacity
    ) * 100


    if additional_chargers > 0:

        status = "Capacity Expansion Recommended"

        message = (
            f"Consider planning for approximately "
            f"{additional_chargers} additional charger(s) "
            f"under the current planning assumptions."
        )

    elif estimated_utilization >= 70:

        status = "Capacity Adequate"

        message = (
            "Current charger capacity can support "
            "the predicted demand, but utilization "
            "is relatively high."
        )

    else:

        status = "Capacity Available"

        message = (
            "Current charger capacity appears sufficient "
            "for the predicted demand."
        )


    return {
        "required_chargers": required_chargers,

        "current_chargers": current_chargers,

        "additional_chargers": additional_chargers,

        "spare_chargers": spare_chargers,

        "current_capacity": round(
            current_capacity,
            2
        ),

        "estimated_utilization": round(
            estimated_utilization,
            1
        ),

        "status": status,

        "message": message
    }


# ============================================================
# GET WEATHER / STATION DEFAULTS
# ============================================================

@app.route("/station-info")
def station_info():

    station_id = request.args.get(
        "station_id"
    )


    if station_id not in STATIONS:

        return jsonify({
            "error": "Invalid station"
        }), 400


    return jsonify({
        "station_id": station_id,

        "location": STATIONS[
            station_id
        ]["location"],

        "chargers": STATIONS[
            station_id
        ]["chargers"]
    })


# ============================================================
# GET AUTOMATIC HISTORICAL FEATURES
# ============================================================

@app.route("/historical-features")
def historical_features():

    station_id = request.args.get(
        "station_id"
    )

    date_value = request.args.get(
        "date"
    )

    hour_value = request.args.get(
        "hour"
    )


    if station_id not in STATIONS:

        return jsonify({
            "error": "Invalid station"
        }), 400


    if not date_value or hour_value is None:

        return jsonify({
            "error": "Date and hour are required"
        }), 400


    try:

        selected_timestamp = pd.Timestamp(
            date_value
        ) + pd.Timedelta(
            hours=int(hour_value)
        )

    except Exception:

        return jsonify({
            "error": "Invalid date or hour"
        }), 400


    features = get_historical_features(
        station_id,
        selected_timestamp
    )


    return jsonify(features)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    analytics = generate_analytics()

    return render_template(
        "index.html",
        stations=STATIONS,
        analytics=analytics,
        prediction=None,
        demand_class=None,
        capacity=None,
        historical_features=None,
        historical_source=None,
        error=None
    )


# ============================================================
# PREDICTION
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # ----------------------------------------------------
        # Form values
        # ----------------------------------------------------

        station_id = request.form[
            "station_id"
        ]

        date_value = request.form[
            "date"
        ]

        hour = int(
            request.form["hour"]
        )

        is_holiday = int(
            request.form["is_holiday"]
        )

        temperature = float(
            request.form["temperature"]
        )

        humidity = float(
            request.form["humidity"]
        )

        rainfall = float(
            request.form["rainfall"]
        )


        # ----------------------------------------------------
        # Validate station
        # ----------------------------------------------------

        if station_id not in STATIONS:

            raise ValueError(
                "Invalid charging station."
            )


        station_info = STATIONS[
            station_id
        ]

        location = station_info[
            "location"
        ]

        charger_count = station_info[
            "chargers"
        ]


        # ----------------------------------------------------
        # Date/time
        # ----------------------------------------------------

        selected_timestamp = (
            pd.Timestamp(date_value)
            + pd.Timedelta(
                hours=hour
            )
        )


        month = selected_timestamp.month

        day_of_week = (
            selected_timestamp.dayofweek
        )

        is_weekend = (
            1
            if day_of_week >= 5
            else 0
        )


        # ----------------------------------------------------
        # Automatically retrieve historical features
        # ----------------------------------------------------

        historical = get_historical_features(
            station_id,
            selected_timestamp
        )


        previous_hour_demand = historical[
            "previous_hour_demand"
        ]

        previous_day_demand = historical[
            "previous_day_demand"
        ]

        previous_week_demand = historical[
            "previous_week_demand"
        ]

        rolling_24h_mean = historical[
            "rolling_24h_mean"
        ]

        rolling_7d_mean = historical[
            "rolling_7d_mean"
        ]


        # ----------------------------------------------------
        # Cyclic time features
        # ----------------------------------------------------

        hour_sin = np.sin(
            2 * np.pi * hour / 24
        )

        hour_cos = np.cos(
            2 * np.pi * hour / 24
        )

        day_sin = np.sin(
            2 * np.pi * day_of_week / 7
        )

        day_cos = np.cos(
            2 * np.pi * day_of_week / 7
        )


        # ----------------------------------------------------
        # Create prediction data
        # ----------------------------------------------------

        input_data = pd.DataFrame({

            "station_id": [station_id],

            "location": [location],

            "charger_count": [
                charger_count
            ],

            "hour": [hour],

            "day_of_week": [
                day_of_week
            ],

            "month": [month],

            "is_weekend": [
                is_weekend
            ],

            "is_holiday": [
                is_holiday
            ],

            "temperature": [
                temperature
            ],

            "humidity": [
                humidity
            ],

            "rainfall": [
                rainfall
            ],

            "previous_hour_demand": [
                previous_hour_demand
            ],

            "previous_day_demand": [
                previous_day_demand
            ],

            "previous_week_demand": [
                previous_week_demand
            ],

            "rolling_24h_mean": [
                rolling_24h_mean
            ],

            "rolling_7d_mean": [
                rolling_7d_mean
            ],

            "hour_sin": [
                hour_sin
            ],

            "hour_cos": [
                hour_cos
            ],

            "day_sin": [
                day_sin
            ],

            "day_cos": [
                day_cos
            ]
        })


        # ----------------------------------------------------
        # MODEL FEATURE COMPATIBILITY
        # ----------------------------------------------------
        #
        # If the saved model contains feature_names_in_,
        # use exactly those features.
        #
        # This prevents feature-count/order mismatch.
        # ----------------------------------------------------

        if hasattr(
            model,
            "feature_names_in_"
        ):

            expected_features = list(
                model.feature_names_in_
            )

            missing_features = [
                col
                for col in expected_features
                if col not in input_data.columns
            ]


            if missing_features:

                raise ValueError(
                    "The saved model expects "
                    f"these missing features: "
                    f"{missing_features}"
                )


            input_data = input_data[
                expected_features
            ]


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(
            input_data
        )[0]


        prediction = max(
            0,
            float(prediction)
        )


        # ----------------------------------------------------
        # Demand classification
        # ----------------------------------------------------

        demand_class = classify_demand(
            prediction
        )


        # ----------------------------------------------------
        # Capacity planning
        # ----------------------------------------------------

        capacity = calculate_capacity_plan(
            prediction,
            charger_count
        )


        # ----------------------------------------------------
        # Analytics
        # ----------------------------------------------------

        analytics = generate_analytics()


        # ----------------------------------------------------
        # Render dashboard
        # ----------------------------------------------------

        return render_template(

            "index.html",

            stations=STATIONS,

            analytics=analytics,

            prediction=round(
                prediction,
                2
            ),

            demand_class=demand_class,

            capacity=capacity,

            historical_features=historical,

            historical_source=historical[
                "source"
            ],

            selected_station=station_id,

            selected_date=date_value,

            selected_hour=hour,

            selected_holiday=is_holiday,

            selected_temperature=temperature,

            selected_humidity=humidity,

            selected_rainfall=rainfall,

            error=None
        )


    except Exception as e:

        print(
            "Prediction error:",
            e
        )

        analytics = generate_analytics()

        return render_template(

            "index.html",

            stations=STATIONS,

            analytics=analytics,

            prediction=None,

            demand_class=None,

            capacity=None,

            historical_features=None,

            historical_source=None,

            error=str(e)
        )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )