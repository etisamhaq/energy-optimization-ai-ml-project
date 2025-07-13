"""
Streamlit dashboard for Energy Optimization ML.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.predict import EnergyPredictor
from src.optimization.optimizer import EnergyOptimizer, OptimizationType
from src.data.data_loader import EnergyDataLoader

# Page configuration
st.set_page_config(
    page_title="Energy Optimization Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .recommendation-card {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1976D2;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = EnergyPredictor()
if 'optimizer' not in st.session_state:
    st.session_state.optimizer = EnergyOptimizer()
if 'data_loader' not in st.session_state:
    st.session_state.data_loader = EnergyDataLoader()

# Header
st.markdown("<h1 class='main-header'>⚡ Energy Optimization Dashboard</h1>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # Building selection
    building_ids = [f"B{str(i).zfill(3)}" for i in range(1, 11)]
    selected_building = st.selectbox("Select Building", building_ids)
    
    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=7)
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now() + timedelta(days=7)
        )
    
    # Model selection
    model_options = ['ensemble', 'xgboost', 'lightgbm', 'random_forest', 'prophet']
    selected_model = st.selectbox("Prediction Model", model_options)
    
    # Electricity rate
    electricity_rate = st.number_input(
        "Electricity Rate ($/kWh)",
        min_value=0.05,
        max_value=0.50,
        value=0.12,
        step=0.01
    )
    
    # Run analysis button
    run_analysis = st.button("Run Analysis", type="primary", use_container_width=True)

# Main content area
if run_analysis:
    with st.spinner("Loading predictions and generating insights..."):
        try:
            # Get building metadata
            building_metadata = st.session_state.data_loader.load_building_metadata([selected_building])
            metadata_dict = building_metadata.iloc[0].to_dict()
            
            # Make predictions
            predictions_df, daily_totals = st.session_state.predictor.predict_range(
                selected_building,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                selected_model
            )
            
            # Generate optimization report
            st.session_state.optimizer.electricity_rate = electricity_rate
            optimization_report = st.session_state.optimizer.generate_optimization_report(
                predictions_df,
                metadata_dict
            )
            
            # Store in session state for use in other sections
            st.session_state.predictions_df = predictions_df
            st.session_state.daily_totals = daily_totals
            st.session_state.optimization_report = optimization_report
            st.session_state.metadata_dict = metadata_dict
            
            st.success("Analysis complete!")
            
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")

# Display results if available
if 'predictions_df' in st.session_state:
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "📈 Predictions", 
        "💡 Optimization", 
        "🏢 Building Info",
        "📋 Reports"
    ])
    
    with tab1:
        # Overview metrics
        st.subheader("Energy Consumption Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_kwh = st.session_state.predictions_df['predicted_consumption_kwh'].sum()
        avg_daily = st.session_state.daily_totals['daily_total_kwh'].mean()
        peak_hour = st.session_state.predictions_df['predicted_consumption_kwh'].max()
        total_cost = total_kwh * electricity_rate
        
        with col1:
            st.metric(
                "Total Predicted (kWh)",
                f"{total_kwh:,.0f}",
                f"${total_cost:,.2f}"
            )
        
        with col2:
            st.metric(
                "Average Daily (kWh)",
                f"{avg_daily:,.0f}",
                f"${avg_daily * electricity_rate:,.2f}/day"
            )
        
        with col3:
            st.metric(
                "Peak Hour (kWh)",
                f"{peak_hour:,.0f}"
            )
        
        with col4:
            savings_potential = st.session_state.optimization_report['summary']['total_kwh_savings_potential']
            st.metric(
                "Savings Potential",
                f"{savings_potential:,.0f} kWh",
                f"{st.session_state.optimization_report['summary']['savings_percentage']:.1f}%"
            )
        
        # Consumption patterns chart
        st.subheader("Consumption Patterns")
        
        # Hourly pattern
        hourly_avg = st.session_state.predictions_df.groupby(
            st.session_state.predictions_df['timestamp'].dt.hour
        )['predicted_consumption_kwh'].mean()
        
        fig_hourly = px.line(
            x=hourly_avg.index,
            y=hourly_avg.values,
            title="Average Hourly Consumption Pattern",
            labels={'x': 'Hour of Day', 'y': 'Consumption (kWh)'}
        )
        fig_hourly.update_traces(line_color='#1E88E5', line_width=3)
        st.plotly_chart(fig_hourly, use_container_width=True)
        
        # Daily totals
        fig_daily = px.bar(
            st.session_state.daily_totals,
            x='date',
            y='daily_total_kwh',
            title="Daily Energy Consumption",
            labels={'daily_total_kwh': 'Total Consumption (kWh)', 'date': 'Date'}
        )
        fig_daily.update_traces(marker_color='#43A047')
        st.plotly_chart(fig_daily, use_container_width=True)
    
    with tab2:
        # Detailed predictions
        st.subheader("Energy Consumption Predictions")
        
        # Time series plot
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=st.session_state.predictions_df['timestamp'],
            y=st.session_state.predictions_df['predicted_consumption_kwh'],
            mode='lines',
            name='Predicted Consumption',
            line=dict(color='#1976D2', width=2)
        ))
        
        fig_ts.update_layout(
            title="Hourly Energy Consumption Forecast",
            xaxis_title="Time",
            yaxis_title="Consumption (kWh)",
            hovermode='x unified'
        )
        st.plotly_chart(fig_ts, use_container_width=True)
        
        # Heatmap of consumption
        pivot_data = st.session_state.predictions_df.pivot_table(
            values='predicted_consumption_kwh',
            index=st.session_state.predictions_df['timestamp'].dt.hour,
            columns=st.session_state.predictions_df['timestamp'].dt.date,
            aggfunc='mean'
        )
        
        fig_heatmap = px.imshow(
            pivot_data,
            title="Energy Consumption Heatmap",
            labels=dict(x="Date", y="Hour of Day", color="Consumption (kWh)"),
            color_continuous_scale="RdYlBu_r"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Download predictions
        csv = st.session_state.predictions_df.to_csv(index=False)
        st.download_button(
            label="Download Predictions CSV",
            data=csv,
            file_name=f"predictions_{selected_building}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with tab3:
        # Optimization recommendations
        st.subheader("Optimization Recommendations")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        summary = st.session_state.optimization_report['summary']
        
        with col1:
            st.metric(
                "Total Recommendations",
                summary['total_recommendations']
            )
        
        with col2:
            st.metric(
                "Potential Annual Savings",
                f"${summary['total_cost_savings_annual']:,.0f}"
            )
        
        with col3:
            st.metric(
                "Average Payback",
                f"{summary['average_payback_months']:.1f} months"
            )
        
        # Recommendations by type
        st.subheader("Recommendations by Category")
        
        rec_by_type = st.session_state.optimization_report['recommendations_by_type']
        if rec_by_type:
            type_df = pd.DataFrame.from_dict(rec_by_type, orient='index')
            
            fig_rec_types = px.bar(
                type_df,
                y=type_df.index,
                x='total_cost_savings',
                orientation='h',
                title="Annual Savings by Recommendation Type",
                labels={'total_cost_savings': 'Annual Savings ($)', 'y': 'Type'}
            )
            st.plotly_chart(fig_rec_types, use_container_width=True)
        
        # Detailed recommendations
        st.subheader("Detailed Recommendations")
        
        for rec in st.session_state.optimization_report['recommendations']:
            with st.expander(f"{rec['title']} - Priority {rec['priority']}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Energy Savings", f"{rec['impact_kwh']:,.0f} kWh")
                
                with col2:
                    st.metric("Cost Savings", f"${rec['cost_savings']:,.2f}/month")
                
                with col3:
                    st.metric("Payback Period", f"{rec['payback_period_months']:.1f} months")
                
                st.write(f"**Description:** {rec['description']}")
                st.write(f"**Confidence:** {rec['confidence']*100:.0f}%")
                
                st.write("**Implementation Steps:**")
                for i, step in enumerate(rec['implementation_steps'], 1):
                    st.write(f"{i}. {step}")
                
                if rec['prerequisites']:
                    st.write("**Prerequisites:**")
                    for prereq in rec['prerequisites']:
                        st.write(f"- {prereq}")
    
    with tab4:
        # Building information
        st.subheader("Building Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Building Details**")
            for key, value in st.session_state.metadata_dict.items():
                if key != 'building_id':
                    formatted_key = key.replace('_', ' ').title()
                    st.write(f"- {formatted_key}: {value}")
        
        with col2:
            # Building efficiency gauge
            efficiency_score = st.session_state.metadata_dict.get('energy_star_rating', 50)
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=efficiency_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Energy Star Rating"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 75], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 75
                    }
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
    
    with tab5:
        # Reports
        st.subheader("Generate Reports")
        
        report_type = st.selectbox(
            "Select Report Type",
            ["Executive Summary", "Detailed Analysis", "Optimization Plan"]
        )
        
        if st.button("Generate Report"):
            if report_type == "Executive Summary":
                report_content = f"""
# Energy Optimization Report - Executive Summary

**Building:** {selected_building}  
**Analysis Period:** {start_date} to {end_date}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Key Findings

- **Total Predicted Consumption:** {total_kwh:,.0f} kWh
- **Estimated Cost:** ${total_cost:,.2f}
- **Potential Savings:** {savings_potential:,.0f} kWh ({st.session_state.optimization_report['summary']['savings_percentage']:.1f}%)
- **Number of Recommendations:** {summary['total_recommendations']}

## Top Recommendations

"""
                # Add top 3 recommendations
                for i, rec in enumerate(st.session_state.optimization_report['recommendations'][:3], 1):
                    report_content += f"\n{i}. **{rec['title']}**\n"
                    report_content += f"   - Savings: {rec['impact_kwh']:,.0f} kWh/year\n"
                    report_content += f"   - ROI: {rec['payback_period_months']:.1f} months\n"
                
                st.markdown(report_content)
                
                # Download button
                st.download_button(
                    label="Download Report",
                    data=report_content,
                    file_name=f"energy_report_{selected_building}_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown"
                )

# Footer
st.markdown("---")
st.markdown("Built with ❤️ for sustainable energy management")

# Run the app
if __name__ == "__main__":
    st.write("Dashboard loaded successfully!")