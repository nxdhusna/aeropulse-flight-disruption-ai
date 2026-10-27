import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.disruption_engine import FlightDisruptionEngine, ROUTES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "klia_flight_disruptions_powerbi.csv")

st.set_page_config(
    page_title="AeroPulse AI | Flight Disruption & Turnaround Control",
    page_icon="✈️",
    layout="wide"
)


@st.cache_resource
def load_engine():
    return FlightDisruptionEngine()


engine = load_engine()

st.title("✈️ AeroPulse AI: Intelligent Flight Disruption & Turnaround Recovery Platform")
st.markdown(
    "**KLIA Hub Airport Operations Control Center (AOCC)** | Powered by **Dual-Stage XGBoost (`0.9851 AUC` Classifier + `±3.84m MAE` Regressor)** & **SHAP Explainable AI**."
)

tab1, tab2 = st.tabs([
    "📡 Live AOCC Flight Radar & Turnaround Simulator",
    "📊 Hub OTP Analytics & Power BI Export"
])

with tab1:
    with st.sidebar:
        st.header("🛫 Flight & Sector Selection")
        route_code = st.selectbox(
            "Select Active KLIA Route",
            list(ROUTES.keys()),
            format_func=lambda k: f"{ROUTES[k]['flight']} | {k} ({ROUTES[k]['dest']})"
        )
        departure_hour = st.slider("Scheduled Departure Hour (24h)", 6, 23, 18)

        st.divider()
        st.header("⛈️ Live Operational Conditions")
        inbound_delay_mins = st.slider("Inbound Aircraft Delay (Mins)", 0, 120, 42)
        turnaround_buffer_mins = st.slider("Scheduled Ground Turnaround Buffer (Mins)", 35, 75, 45)
        weather_severity_idx = st.slider("Weather / Crosswind Severity (0 = Clear, 10 = Storm)", 0.0, 10.0, 6.5, 0.5)
        airport_congestion_idx = st.slider("KLIA Slot & Runway Congestion Index", 0.10, 0.99, 0.82, 0.02)
        pax_load_factor = st.slider("Passenger Load Factor", 0.55, 0.99, 0.90, 0.01)
        crew_duty_risk = st.selectbox("Crew Flight Duty Period (FDP) Timeout Risk?", [0, 1], format_func=lambda x: "Yes (Near Limit)" if x == 1 else "No (Normal)")

        st.divider()
        st.header("🛠️ AOCC Turnaround Recovery Simulator")
        st.caption("Simulate operational interventions to absorb inbound delay:")
        act_rapid_clean = st.checkbox("⚡ Deploy Rapid Turnaround Crew (-12m)")
        act_dual_board = st.checkbox("🚪 Enable Dual-Bridge Boarding (-8m)")
        act_priority_taxi = st.checkbox("🛫 Priority Gate & Pushback Slot (-5m)")

    recovery_saved = (12 if act_rapid_clean else 0) + (8 if act_dual_board else 0) + (5 if act_priority_taxi else 0)

    flight_input = {
        "route_code": route_code,
        "departure_hour": departure_hour,
        "inbound_delay_mins": inbound_delay_mins,
        "turnaround_buffer_mins": turnaround_buffer_mins,
        "weather_severity_idx": weather_severity_idx,
        "airport_congestion_idx": airport_congestion_idx,
        "pax_load_factor": pax_load_factor,
        "crew_duty_risk": crew_duty_risk
    }

    # Baseline (Without intervention) vs Recovered (With AOCC intervention)
    base_res = engine.analyze_flight(flight_input, recovery_mins_saved=0)
    res = engine.analyze_flight(flight_input, recovery_mins_saved=recovery_saved)
    mins_mitigated = base_res["expected_delay_mins"] - res["expected_delay_mins"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Flight & Tail Number",
        f"{res['flight_no']} ({res['tail_no']})",
        route_code
    )
    k2.metric(
        "OTP Delay Probability (>15m)",
        f"{res['delay_prob'] * 100:.1f}%",
        delta=f"{(res['delay_prob'] - base_res['delay_prob']) * 100:.1f}%" if recovery_saved > 0 else None,
        delta_color="inverse"
    )
    k3.metric(
        "Expected Departure Delay",
        f"{res['expected_delay_mins']} mins",
        delta=f"-{mins_mitigated} mins saved" if mins_mitigated > 0 else "Baseline",
        delta_color="normal" if mins_mitigated > 0 else "off"
    )
    k4.metric("Operational Status", res["status"])

    st.divider()
    col_map, col_diag = st.columns([1.25, 1])

    with col_map:
        st.subheader("🗺️ Live KLIA (KUL) Hub Geospatial Route Radar")
        kul_lat, kul_lon = 2.7456, 101.7072
        fig_map = go.Figure()

        # Plot all KLIA Hub routes in background, highlighting selected route
        for r_code, r_meta in ROUTES.items():
            is_selected = (r_code == route_code)
            line_color = res["color"] if is_selected else "rgba(148, 163, 184, 0.35)"
            line_width = 4 if is_selected else 1.5

            fig_map.add_trace(go.Scattergeo(
                lon=[kul_lon, r_meta["lon"]],
                lat=[kul_lat, r_meta["lat"]],
                mode="lines+markers+text",
                line=dict(width=line_width, color=line_color),
                marker=dict(size=10 if is_selected else 6, color=line_color),
                text=["KUL (KLIA Hub)", r_meta["dest"] if is_selected else r_code.split("-")[1]],
                textposition="top right",
                name=f"{r_meta['flight']} ({r_code})"
            ))

        fig_map.update_geos(
            projection_type="natural earth",
            center=dict(lat=8.5, lon=112.0),
            lataxis_range=[-10, 38],
            lonaxis_range=[94, 145],
            showland=True, landcolor="#1e293b",
            showocean=True, oceancolor="#0f172a",
            showcountries=True, countrycolor="#334155"
        )
        fig_map.update_layout(
            height=430,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False
        )
        st.plotly_chart(fig_map, use_container_width=True)

        st.subheader("🛠️ Prescriptive AOCC Operational Actions")
        for action in res["actions"]:
            st.markdown(f"- {action}")

    with col_diag:
        st.subheader("🔍 Primary Delay Drivers (SHAP Attribution in Minutes)")
        contrib = res["contrib_df"].sort_values(by="shap_mins", ascending=True)
        contrib["Impact Type"] = contrib["shap_mins"].apply(
            lambda x: "Adds Delay (+Mins)" if x > 0 else "Reduces Delay (-Mins)"
        )
        fig_shap = px.bar(
            contrib,
            x="shap_mins",
            y="label",
            color="Impact Type",
            orientation="h",
            color_discrete_map={"Adds Delay (+Mins)": "#ef4444", "Reduces Delay (-Mins)": "#22c55e"},
            labels={"shap_mins": "SHAP Delay Contribution (Minutes)", "label": "Operational Factor"}
        )
        fig_shap.update_layout(height=340, legend_title_text="", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_shap, use_container_width=True)

        st.subheader(f"⛓️ Aircraft Tail Rotation Cascade ({res['tail_no']})")
        st.caption("Shows how inbound delay cascades into downstream flights using the same aircraft.")
        fig_rot = px.bar(
            res["rotation_chain"],
            x="Leg",
            y="Delay_Mins",
            text="Delay_Mins",
            color="Delay_Mins",
            color_continuous_scale=["#22c55e", "#eab308", "#ef4444"],
            range_color=[0, 60],
            labels={"Delay_Mins": "Projected Delay (Mins)"}
        )
        fig_rot.update_traces(texttemplate="%{text} mins", textposition="outside")
        fig_rot.update_layout(height=250, margin=dict(l=10, r=10, t=20, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig_rot, use_container_width=True)

with tab2:
    st.subheader("📊 KLIA Hub On-Time Performance (OTP) & Power BI Data Pipeline")
    if os.path.exists(DATA_PATH):
        df_pbi = pd.read_csv(DATA_PATH)
        c1, c2, c3 = st.columns(3)
        otp_rate = (1 - df_pbi["is_delayed"].mean()) * 100
        c1.metric("Historical Hub OTP (<15m Delay)", f"{otp_rate:.1f}%")
        c2.metric("Average Disruption Duration", f"{df_pbi[df_pbi['is_delayed']==1]['delay_minutes'].mean():.1f} mins")
        c3.metric("Total Flight Telemetry Records", f"{len(df_pbi):,}")

        route_summary = df_pbi.groupby("route_code").agg(
            Avg_Delay_Mins=("delay_minutes", "mean"),
            Disruption_Rate=("is_delayed", lambda x: round(x.mean() * 100, 1))
        ).reset_index()

        fig_route = px.bar(
            route_summary, x="route_code", y="Disruption_Rate",
            color="Avg_Delay_Mins", text="Disruption_Rate",
            title="Disruption Frequency (%) & Avg Delay by KLIA Route Corridor",
            labels={"route_code": "Route Corridor", "Disruption_Rate": "OTP Breach Rate (%)"}
        )
        st.plotly_chart(fig_route, use_container_width=True)

        csv_bytes = df_pbi.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export Enriched Telemetry Table for Power BI (.CSV)",
            csv_bytes,
            "klia_flight_disruptions_powerbi.csv",
            "text/csv"
        )