import os
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

ROUTES = {
    "KUL-BKI": {"flight": "MH2610", "dest": "BKI (Kota Kinabalu)", "dist_km": 1628, "lat": 5.9372, "lon": 116.0510, "tail": "9M-MNA"},
    "KUL-KCH": {"flight": "MH2504", "dest": "KCH (Kuching)", "dist_km": 975, "lat": 1.4847, "lon": 110.3470, "tail": "9M-MNB"},
    "KUL-PEN": {"flight": "MH1132", "dest": "PEN (Penang)", "dist_km": 324, "lat": 5.2971, "lon": 100.2769, "tail": "9M-MXA"},
    "KUL-SIN": {"flight": "MH601", "dest": "SIN (Singapore)", "dist_km": 296, "lat": 1.3644, "lon": 103.9915, "tail": "9M-MXB"},
    "KUL-LGK": {"flight": "MH1430", "dest": "LGK (Langkawi)", "dist_km": 418, "lat": 6.3297, "lon": 99.7287, "tail": "9M-MXC"},
    "KUL-BKK": {"flight": "MH784", "dest": "BKK (Bangkok)", "dist_km": 1220, "lat": 13.6900, "lon": 100.7501, "tail": "9M-MND"},
    "KUL-CGK": {"flight": "MH711", "dest": "CGK (Jakarta)", "dist_km": 1130, "lat": -6.1256, "lon": 106.6559, "tail": "9M-MNE"},
    "KUL-HND": {"flight": "MH70", "dest": "HND (Tokyo Haneda)", "dist_km": 5340, "lat": 35.5494, "lon": 139.7798, "tail": "9M-MAG"}
}

ROUTE_ENCODE = {k: i for i, k in enumerate(ROUTES.keys())}

FEATURE_ORDER = [
    "route_encoded",
    "departure_hour",
    "inbound_delay_mins",
    "turnaround_buffer_mins",
    "weather_severity_idx",
    "airport_congestion_idx",
    "pax_load_factor",
    "crew_duty_risk",
    "route_distance_km"
]

FEATURE_LABELS = {
    "route_encoded": "Route Corridor",
    "departure_hour": "Scheduled Departure Hour",
    "inbound_delay_mins": "Inbound Aircraft Rotation Delay (Mins)",
    "turnaround_buffer_mins": "Ground Turnaround Buffer (Mins)",
    "weather_severity_idx": "Weather & Crosswind Severity (0-10)",
    "airport_congestion_idx": "KLIA Slot & Runway Congestion",
    "pax_load_factor": "Passenger Load Factor (%)",
    "crew_duty_risk": "Crew Legal Duty-Hour Risk Flag",
    "route_distance_km": "Sector Distance (km)"
}


class FlightDisruptionEngine:
    def __init__(self):
        self.clf = joblib.load(os.path.join(MODELS_DIR, "xgb_delay_classifier.pkl"))
        self.reg = joblib.load(os.path.join(MODELS_DIR, "xgb_delay_regressor.pkl"))
        self.explainer = joblib.load(os.path.join(MODELS_DIR, "shap_flight_explainer.pkl"))

    def analyze_flight(self, flight_input: dict, recovery_mins_saved: int = 0) -> dict:
        route_code = flight_input["route_code"]
        route_info = ROUTES[route_code]

        encoded_row = {
            "route_encoded": ROUTE_ENCODE[route_code],
            "departure_hour": flight_input["departure_hour"],
            "inbound_delay_mins": max(0, flight_input["inbound_delay_mins"] - recovery_mins_saved),
            "turnaround_buffer_mins": flight_input["turnaround_buffer_mins"] + recovery_mins_saved,
            "weather_severity_idx": flight_input["weather_severity_idx"],
            "airport_congestion_idx": flight_input["airport_congestion_idx"],
            "pax_load_factor": flight_input["pax_load_factor"],
            "crew_duty_risk": flight_input["crew_duty_risk"],
            "route_distance_km": route_info["dist_km"]
        }

        df_X = pd.DataFrame([encoded_row])[FEATURE_ORDER]

        # Stage 1: Probability of OTP Delay (>= 15 mins)
        delay_prob = float(self.clf.predict_proba(df_X)[0, 1])

        # Stage 2: Expected Delay Duration (Minutes)
        raw_pred_mins = float(self.reg.predict(df_X)[0])
        expected_delay_mins = max(0, int(round(raw_pred_mins)))

        # Risk Category & Prescriptive AOCC Recommendations
        if expected_delay_mins < 15 and delay_prob < 0.35:
            status = "ON TIME (LOW RISK)"
            color = "#22c55e"
            actions = [
                "✅ Standard turnaround protocol — no intervention required.",
                "✅ Maintain scheduled gate assignment and standard fueling."
            ]
        elif expected_delay_mins < 40:
            status = "MODERATE DISRUPTION RISK"
            color = "#eab308"
            actions = [
                "⚡ Deploy Rapid Turnaround Cleaning & Catering Team (cuts ground time by ~12 mins).",
                "🛄 Flag priority baggage transfer for tight-connection passengers at KLIA.",
                "🛫 Request priority pushback slot from KLIA Apron Control."
            ]
        else:
            status = "CRITICAL DISRUPTION (KNOCK-ON ALERT)"
            color = "#ef4444"
            actions = [
                "🚨 Activate Fast-Track Turnaround & Dual-Door Boarding (-15 mins ground time).",
                "🔄 Evaluate Standby Aircraft Tail-Swap at KLIA Hub to break rotation cascade.",
                "👨‍✈️ Alert Crew Scheduling: Verify pilot/cabin Flight Duty Period (FDP) timeout margins.",
                "🤝 Hold high-value connecting passengers or auto-rebook onto next partner flight."
            ]

        # Stage 3: SHAP Root-Cause Attribution (in Minutes of Delay Impact)
        shap_vals = self.explainer.shap_values(df_X)
        shap_row = shap_vals[0] if len(np.shape(shap_vals)) == 2 else shap_vals

        contrib_df = pd.DataFrame({
            "feature": FEATURE_ORDER,
            "label": [FEATURE_LABELS[f] for f in FEATURE_ORDER],
            "shap_mins": np.round(shap_row, 1)
        }).sort_values(by="shap_mins", key=abs, ascending=False)

        top_drivers = []
        for _, r in contrib_df.head(4).iterrows():
            sign = "+" if r["shap_mins"] >= 0 else ""
            icon = "🔺" if r["shap_mins"] > 0 else "🟢"
            top_drivers.append(f"{icon} **{r['label']}** ({float(r['shap_mins']):+.1f} mins impact)")

        # Stage 4: Calculate 3-Leg Aircraft Tail Rotation "Domino Cascade"
        leg1_delay = flight_input["inbound_delay_mins"]
        leg2_delay = expected_delay_mins
        # Next return sector absorbs ~15 mins of ground buffer if turnaround is optimized
        leg3_delay = max(0, int(round(leg2_delay * 0.72 - 10)))

        rotation_chain = pd.DataFrame([
            {"Leg": "Leg 1 (Inbound to KUL)", "Flight": f"{route_info['flight']}A", "Delay_Mins": leg1_delay},
            {"Leg": f"Leg 2 (Active: {route_code})", "Flight": route_info["flight"], "Delay_Mins": leg2_delay},
            {"Leg": "Leg 3 (Return Sector)", "Flight": f"{route_info['flight']}R", "Delay_Mins": leg3_delay}
        ])

        return {
            "flight_no": route_info["flight"],
            "tail_no": route_info["tail"],
            "dest_name": route_info["dest"],
            "delay_prob": round(delay_prob, 4),
            "expected_delay_mins": expected_delay_mins,
            "status": status,
            "color": color,
            "actions": actions,
            "top_drivers": top_drivers,
            "contrib_df": contrib_df,
            "rotation_chain": rotation_chain
        }


if __name__ == "__main__":
    engine = FlightDisruptionEngine()
    sample = {
        "route_code": "KUL-BKI",
        "departure_hour": 18,
        "inbound_delay_mins": 42,
        "turnaround_buffer_mins": 45,
        "weather_severity_idx": 6.5,
        "airport_congestion_idx": 0.82,
        "pax_load_factor": 0.92,
        "crew_duty_risk": 1
    }
    res = engine.analyze_flight(sample)
    print(f"Flight {res['flight_no']} ({res['tail_no']}) -> {res['dest_name']}")
    print(f"Delay Risk: {res['delay_prob']*100:.1f}% | Expected Delay: {res['expected_delay_mins']} mins")
    print("Main Factors:")
    for d in res["top_drivers"]:
        print(" ", d)
    print("Recommended Actions:")
    for a in res["actions"]:
        print(" ", a)