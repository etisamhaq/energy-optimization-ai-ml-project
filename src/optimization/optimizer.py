"""
Energy optimization engine for providing actionable recommendations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class OptimizationType(Enum):
    """Types of optimization recommendations."""
    PEAK_LOAD_SHIFTING = "peak_load_shifting"
    HVAC_OPTIMIZATION = "hvac_optimization"
    SCHEDULE_OPTIMIZATION = "schedule_optimization"
    EQUIPMENT_UPGRADE = "equipment_upgrade"
    BEHAVIORAL_CHANGE = "behavioral_change"
    RENEWABLE_INTEGRATION = "renewable_integration"


@dataclass
class OptimizationRecommendation:
    """Data class for optimization recommendations."""
    recommendation_id: str
    type: OptimizationType
    title: str
    description: str
    impact_kwh: float
    cost_savings: float
    implementation_cost: float
    payback_period_months: float
    priority: int
    confidence: float
    implementation_steps: List[str]
    prerequisites: List[str]
    
    @property
    def roi(self) -> float:
        """Calculate return on investment."""
        if self.implementation_cost == 0:
            return float('inf')
        annual_savings = self.cost_savings * 12
        return (annual_savings / self.implementation_cost) * 100


class EnergyOptimizer:
    """Energy optimization engine."""
    
    def __init__(self, electricity_rate: float = 0.12):
        """
        Initialize optimizer.
        
        Args:
            electricity_rate: Cost per kWh in dollars
        """
        self.electricity_rate = electricity_rate
        self.peak_hours = list(range(7, 19))  # 7 AM to 7 PM
        self.off_peak_hours = list(range(0, 7)) + list(range(19, 24))
    
    def analyze_consumption_patterns(self, consumption_data: pd.DataFrame, 
                                   building_metadata: Dict) -> Dict:
        """Analyze energy consumption patterns."""
        analysis = {
            'peak_usage': {},
            'off_peak_usage': {},
            'daily_patterns': {},
            'weekly_patterns': {},
            'seasonal_patterns': {},
            'anomalies': []
        }
        
        # Ensure timestamp is datetime
        consumption_data['timestamp'] = pd.to_datetime(consumption_data['timestamp'])
        consumption_data['hour'] = consumption_data['timestamp'].dt.hour
        consumption_data['day_of_week'] = consumption_data['timestamp'].dt.dayofweek
        consumption_data['month'] = consumption_data['timestamp'].dt.month
        
        # Peak vs off-peak analysis
        peak_mask = consumption_data['hour'].isin(self.peak_hours)
        analysis['peak_usage'] = {
            'total_kwh': consumption_data[peak_mask]['predicted_consumption_kwh'].sum(),
            'avg_kwh': consumption_data[peak_mask]['predicted_consumption_kwh'].mean(),
            'max_kwh': consumption_data[peak_mask]['predicted_consumption_kwh'].max(),
            'percentage': (consumption_data[peak_mask]['predicted_consumption_kwh'].sum() / 
                         consumption_data['predicted_consumption_kwh'].sum()) * 100
        }
        
        analysis['off_peak_usage'] = {
            'total_kwh': consumption_data[~peak_mask]['predicted_consumption_kwh'].sum(),
            'avg_kwh': consumption_data[~peak_mask]['predicted_consumption_kwh'].mean(),
            'percentage': (consumption_data[~peak_mask]['predicted_consumption_kwh'].sum() / 
                         consumption_data['predicted_consumption_kwh'].sum()) * 100
        }
        
        # Daily patterns
        hourly_avg = consumption_data.groupby('hour')['predicted_consumption_kwh'].mean()
        analysis['daily_patterns'] = {
            'peak_hour': int(hourly_avg.idxmax()),
            'peak_hour_consumption': hourly_avg.max(),
            'lowest_hour': int(hourly_avg.idxmin()),
            'lowest_hour_consumption': hourly_avg.min(),
            'hourly_averages': hourly_avg.to_dict()
        }
        
        # Weekly patterns
        daily_avg = consumption_data.groupby('day_of_week')['predicted_consumption_kwh'].sum()
        analysis['weekly_patterns'] = {
            'highest_day': int(daily_avg.idxmax()),
            'highest_day_consumption': daily_avg.max(),
            'lowest_day': int(daily_avg.idxmin()),
            'lowest_day_consumption': daily_avg.min(),
            'weekend_vs_weekday_ratio': (daily_avg[5:].mean() / daily_avg[:5].mean())
        }
        
        # Detect anomalies (consumption > 2 std dev from mean)
        mean_consumption = consumption_data['predicted_consumption_kwh'].mean()
        std_consumption = consumption_data['predicted_consumption_kwh'].std()
        threshold = mean_consumption + 2 * std_consumption
        
        anomalies = consumption_data[consumption_data['predicted_consumption_kwh'] > threshold]
        if not anomalies.empty:
            analysis['anomalies'] = anomalies[['timestamp', 'predicted_consumption_kwh']].to_dict('records')
        
        return analysis
    
    def generate_peak_load_shifting_recommendations(self, analysis: Dict, 
                                                  building_metadata: Dict) -> List[OptimizationRecommendation]:
        """Generate recommendations for peak load shifting."""
        recommendations = []
        
        # Calculate potential savings from shifting peak loads
        peak_percentage = analysis['peak_usage']['percentage']
        if peak_percentage > 60:
            # Significant peak usage - recommend load shifting
            shiftable_load = analysis['peak_usage']['total_kwh'] * 0.2  # Assume 20% can be shifted
            cost_savings = shiftable_load * self.electricity_rate * 0.3  # 30% rate difference
            
            rec = OptimizationRecommendation(
                recommendation_id="PLS001",
                type=OptimizationType.PEAK_LOAD_SHIFTING,
                title="Implement Peak Load Shifting Strategy",
                description=f"Shift {shiftable_load:.0f} kWh of load from peak hours to off-peak hours. "
                          f"Currently {peak_percentage:.1f}% of consumption occurs during peak hours.",
                impact_kwh=shiftable_load,
                cost_savings=cost_savings,
                implementation_cost=5000,  # Smart controls
                payback_period_months=5000 / (cost_savings * 12) if cost_savings > 0 else float('inf'),
                priority=1,
                confidence=0.85,
                implementation_steps=[
                    "Install smart building management system",
                    "Program automated load shifting schedules",
                    "Train staff on peak load management",
                    "Monitor and optimize shifting patterns"
                ],
                prerequisites=["Smart meter installation", "Flexible equipment schedules"]
            )
            recommendations.append(rec)
        
        # Time-of-use optimization
        if analysis['daily_patterns']['peak_hour_consumption'] > analysis['daily_patterns']['lowest_hour_consumption'] * 3:
            rec = OptimizationRecommendation(
                recommendation_id="PLS002",
                type=OptimizationType.PEAK_LOAD_SHIFTING,
                title="Time-of-Use Rate Optimization",
                description="Enroll in time-of-use electricity rates and adjust consumption patterns",
                impact_kwh=analysis['peak_usage']['total_kwh'] * 0.1,
                cost_savings=analysis['peak_usage']['total_kwh'] * 0.1 * self.electricity_rate * 0.25,
                implementation_cost=500,
                payback_period_months=2,
                priority=2,
                confidence=0.9,
                implementation_steps=[
                    "Contact utility for time-of-use rate enrollment",
                    "Install real-time energy monitoring",
                    "Create consumption scheduling guidelines",
                    "Educate occupants on rate structure"
                ],
                prerequisites=["Utility offers TOU rates"]
            )
            recommendations.append(rec)
        
        return recommendations
    
    def generate_hvac_optimization_recommendations(self, analysis: Dict, 
                                                  building_metadata: Dict,
                                                  weather_data: Optional[pd.DataFrame] = None) -> List[OptimizationRecommendation]:
        """Generate HVAC optimization recommendations."""
        recommendations = []
        
        # Temperature setpoint optimization
        hvac_consumption_estimate = analysis['peak_usage']['total_kwh'] * 0.4  # HVAC typically 40% of load
        
        rec = OptimizationRecommendation(
            recommendation_id="HVAC001",
            type=OptimizationType.HVAC_OPTIMIZATION,
            title="Optimize HVAC Temperature Setpoints",
            description="Adjust temperature setpoints by 2°F during peak hours to reduce HVAC load",
            impact_kwh=hvac_consumption_estimate * 0.15,
            cost_savings=hvac_consumption_estimate * 0.15 * self.electricity_rate,
            implementation_cost=200,
            payback_period_months=1,
            priority=1,
            confidence=0.95,
            implementation_steps=[
                "Program smart thermostats with optimized setpoints",
                "Create seasonal setpoint schedules",
                "Monitor occupant comfort levels",
                "Fine-tune based on feedback"
            ],
            prerequisites=["Programmable thermostats"]
        )
        recommendations.append(rec)
        
        # Weekend/holiday scheduling
        if analysis['weekly_patterns']['weekend_vs_weekday_ratio'] > 0.7:
            weekend_savings = (analysis['weekly_patterns']['weekend_vs_weekday_ratio'] - 0.5) * \
                            analysis['off_peak_usage']['total_kwh'] * 0.2
            
            rec = OptimizationRecommendation(
                recommendation_id="HVAC002",
                type=OptimizationType.HVAC_OPTIMIZATION,
                title="Implement Weekend HVAC Scheduling",
                description="Reduce HVAC operation during weekends when building occupancy is lower",
                impact_kwh=weekend_savings,
                cost_savings=weekend_savings * self.electricity_rate,
                implementation_cost=100,
                payback_period_months=0.5,
                priority=2,
                confidence=0.9,
                implementation_steps=[
                    "Program weekend setback schedules",
                    "Verify occupancy patterns",
                    "Create override procedures for special events",
                    "Monitor energy savings"
                ],
                prerequisites=["Programmable HVAC controls"]
            )
            recommendations.append(rec)
        
        return recommendations
    
    def generate_schedule_optimization_recommendations(self, analysis: Dict, 
                                                     building_metadata: Dict) -> List[OptimizationRecommendation]:
        """Generate operational schedule optimization recommendations."""
        recommendations = []
        
        # Early morning startup optimization
        early_hours = [5, 6, 7]
        early_consumption = sum(analysis['daily_patterns']['hourly_averages'].get(h, 0) for h in early_hours)
        
        if early_consumption > analysis['daily_patterns']['lowest_hour_consumption'] * 6:
            rec = OptimizationRecommendation(
                recommendation_id="SCH001",
                type=OptimizationType.SCHEDULE_OPTIMIZATION,
                title="Optimize Morning Startup Sequence",
                description="Stagger equipment startup to reduce morning demand spike",
                impact_kwh=early_consumption * 0.2,
                cost_savings=early_consumption * 0.2 * self.electricity_rate * 1.2,
                implementation_cost=1000,
                payback_period_months=3,
                priority=2,
                confidence=0.8,
                implementation_steps=[
                    "Analyze equipment startup sequences",
                    "Create staggered startup schedule",
                    "Program automated controls",
                    "Train maintenance staff"
                ],
                prerequisites=["Equipment with programmable controls"]
            )
            recommendations.append(rec)
        
        return recommendations
    
    def generate_equipment_upgrade_recommendations(self, analysis: Dict, 
                                                 building_metadata: Dict) -> List[OptimizationRecommendation]:
        """Generate equipment upgrade recommendations."""
        recommendations = []
        
        # LED lighting upgrade
        lighting_estimate = analysis['peak_usage']['total_kwh'] * 0.2  # Lighting typically 20%
        building_age = building_metadata.get('building_age', 10)
        
        if building_age > 10:
            rec = OptimizationRecommendation(
                recommendation_id="EQP001",
                type=OptimizationType.EQUIPMENT_UPGRADE,
                title="LED Lighting Retrofit",
                description="Replace traditional lighting with LED fixtures for 50% energy reduction",
                impact_kwh=lighting_estimate * 0.5,
                cost_savings=lighting_estimate * 0.5 * self.electricity_rate,
                implementation_cost=building_metadata.get('floor_area_sqm', 5000) * 5,
                payback_period_months=24,
                priority=3,
                confidence=0.95,
                implementation_steps=[
                    "Conduct lighting audit",
                    "Select appropriate LED replacements",
                    "Phase installation by area",
                    "Dispose of old fixtures properly"
                ],
                prerequisites=["Building has non-LED lighting"]
            )
            recommendations.append(rec)
        
        # High-efficiency HVAC
        if building_metadata.get('energy_star_rating', 50) < 75:
            hvac_upgrade_savings = analysis['peak_usage']['total_kwh'] * 0.4 * 0.25
            
            rec = OptimizationRecommendation(
                recommendation_id="EQP002",
                type=OptimizationType.EQUIPMENT_UPGRADE,
                title="High-Efficiency HVAC System Upgrade",
                description="Replace aging HVAC equipment with high-efficiency models",
                impact_kwh=hvac_upgrade_savings,
                cost_savings=hvac_upgrade_savings * self.electricity_rate,
                implementation_cost=building_metadata.get('floor_area_sqm', 5000) * 20,
                payback_period_months=60,
                priority=4,
                confidence=0.85,
                implementation_steps=[
                    "Perform HVAC system assessment",
                    "Size new equipment properly",
                    "Select ENERGY STAR certified models",
                    "Schedule phased replacement"
                ],
                prerequisites=["HVAC system > 15 years old"]
            )
            recommendations.append(rec)
        
        return recommendations
    
    def generate_behavioral_recommendations(self, analysis: Dict, 
                                          building_metadata: Dict) -> List[OptimizationRecommendation]:
        """Generate behavioral change recommendations."""
        recommendations = []
        
        # Occupant awareness program
        behavioral_savings = analysis['peak_usage']['total_kwh'] * 0.05
        
        rec = OptimizationRecommendation(
            recommendation_id="BEH001",
            type=OptimizationType.BEHAVIORAL_CHANGE,
            title="Energy Awareness Campaign",
            description="Implement occupant education program to reduce energy waste",
            impact_kwh=behavioral_savings,
            cost_savings=behavioral_savings * self.electricity_rate,
            implementation_cost=2000,
            payback_period_months=4,
            priority=2,
            confidence=0.7,
            implementation_steps=[
                "Create energy awareness materials",
                "Conduct training sessions",
                "Install real-time energy displays",
                "Implement recognition program"
            ],
            prerequisites=["Management support"]
        )
        recommendations.append(rec)
        
        return recommendations
    
    def generate_renewable_recommendations(self, analysis: Dict, 
                                         building_metadata: Dict) -> List[OptimizationRecommendation]:
        """Generate renewable energy recommendations."""
        recommendations = []
        
        # Solar PV installation
        if building_metadata.get('building_type') in ['office', 'retail', 'industrial']:
            daily_consumption = sum(analysis['daily_patterns']['hourly_averages'].values())
            solar_potential = daily_consumption * 0.3  # 30% offset potential
            
            rec = OptimizationRecommendation(
                recommendation_id="REN001",
                type=OptimizationType.RENEWABLE_INTEGRATION,
                title="Rooftop Solar PV Installation",
                description="Install solar panels to offset 30% of electricity consumption",
                impact_kwh=solar_potential * 365,
                cost_savings=solar_potential * 365 * self.electricity_rate,
                implementation_cost=solar_potential * 1000,  # $1000/kW estimate
                payback_period_months=84,  # 7 years
                priority=5,
                confidence=0.8,
                implementation_steps=[
                    "Conduct solar feasibility study",
                    "Obtain permits and approvals",
                    "Select qualified installer",
                    "Apply for incentives and tax credits"
                ],
                prerequisites=["Suitable roof space", "Structural capacity"]
            )
            recommendations.append(rec)
        
        return recommendations
    
    def generate_optimization_report(self, consumption_data: pd.DataFrame,
                                   building_metadata: Dict,
                                   weather_data: Optional[pd.DataFrame] = None) -> Dict:
        """Generate comprehensive optimization report."""
        # Analyze consumption patterns
        analysis = self.analyze_consumption_patterns(consumption_data, building_metadata)
        
        # Generate recommendations by category
        all_recommendations = []
        
        all_recommendations.extend(
            self.generate_peak_load_shifting_recommendations(analysis, building_metadata)
        )
        all_recommendations.extend(
            self.generate_hvac_optimization_recommendations(analysis, building_metadata, weather_data)
        )
        all_recommendations.extend(
            self.generate_schedule_optimization_recommendations(analysis, building_metadata)
        )
        all_recommendations.extend(
            self.generate_equipment_upgrade_recommendations(analysis, building_metadata)
        )
        all_recommendations.extend(
            self.generate_behavioral_recommendations(analysis, building_metadata)
        )
        all_recommendations.extend(
            self.generate_renewable_recommendations(analysis, building_metadata)
        )
        
        # Sort by priority and ROI
        all_recommendations.sort(key=lambda x: (x.priority, -x.roi))
        
        # Calculate total potential savings
        total_kwh_savings = sum(rec.impact_kwh for rec in all_recommendations)
        total_cost_savings = sum(rec.cost_savings for rec in all_recommendations)
        total_implementation_cost = sum(rec.implementation_cost for rec in all_recommendations)
        
        # Create report
        report = {
            'analysis': analysis,
            'recommendations': [vars(rec) for rec in all_recommendations],
            'summary': {
                'total_recommendations': len(all_recommendations),
                'total_kwh_savings_potential': total_kwh_savings,
                'total_cost_savings_annual': total_cost_savings * 12,
                'total_implementation_cost': total_implementation_cost,
                'average_payback_months': np.mean([rec.payback_period_months for rec in all_recommendations 
                                                  if rec.payback_period_months < float('inf')]),
                'savings_percentage': (total_kwh_savings / consumption_data['predicted_consumption_kwh'].sum()) * 100
            },
            'recommendations_by_type': {}
        }
        
        # Group by type
        for opt_type in OptimizationType:
            type_recs = [rec for rec in all_recommendations if rec.type == opt_type]
            if type_recs:
                report['recommendations_by_type'][opt_type.value] = {
                    'count': len(type_recs),
                    'total_savings_kwh': sum(rec.impact_kwh for rec in type_recs),
                    'total_cost_savings': sum(rec.cost_savings for rec in type_recs) * 12
                }
        
        return report