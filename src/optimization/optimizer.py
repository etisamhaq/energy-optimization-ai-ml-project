"""
Energy optimization engine for providing actionable recommendations.

Savings are estimated from the analysed consumption with rule-of-thumb factors.
Every factor and cost lives in DEFAULT_ASSUMPTIONS so it can be reviewed and
overridden per deployment; payback and ROI are calculated from those inputs.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


# Rule-of-thumb inputs for the savings estimates. They are typical planning values,
# not measurements of a specific building: override them with site data where known.
DEFAULT_ASSUMPTIONS = {
    # Share of total load by end use
    'hvac_load_share': 0.40,
    'lighting_load_share': 0.20,

    # Peak load shifting with smart controls
    'shiftable_peak_fraction': 0.20,          # share of peak-hour energy that can move off-peak
    'peak_offpeak_price_gap': 0.30,           # off-peak price discount vs. average rate
    'smart_controls_cost': 5000.0,            # $ per building

    # Time-of-use tariff enrollment
    'tou_shift_fraction': 0.10,
    'tou_price_gap': 0.25,
    'tou_enrollment_cost': 500.0,

    # HVAC operations
    'setpoint_hvac_savings': 0.15,            # HVAC savings from a 2°F setpoint change
    'setpoint_cost': 200.0,
    'weekend_offpeak_hvac_fraction': 0.20,
    'weekend_schedule_cost': 100.0,

    # Morning startup staggering
    'startup_savings': 0.20,
    'startup_demand_premium': 1.20,           # demand charges make startup energy pricier
    'startup_cost': 1000.0,

    # Equipment upgrades
    'led_lighting_savings': 0.50,
    'led_cost_per_sqm': 5.0,
    'led_min_building_age': 10,
    'hvac_upgrade_savings': 0.25,
    'hvac_upgrade_cost_per_sqm': 20.0,
    'hvac_upgrade_max_energy_star': 75,

    # Occupant behaviour
    'behavioral_peak_savings': 0.05,
    'behavioral_program_cost': 2000.0,

    # Rooftop solar
    'solar_offset_fraction': 0.30,
    'solar_peak_sun_hours': 4.5,              # kWh generated per kW of panels per day
    'solar_cost_per_kw': 2500.0,              # installed commercial PV, $ per kW
    'solar_building_types': ['office', 'retail', 'industrial', 'education', 'warehouse/storage'],
}


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
    """
    Data class for optimization recommendations.

    impact_kwh and cost_savings cover the analysed period. annual_cost_savings and
    payback_period_months are filled in by EnergyOptimizer once the period is known.
    confidence is a heuristic rating of how reliably the measure delivers, not a
    statistical probability.
    """
    recommendation_id: str
    type: OptimizationType
    title: str
    description: str
    impact_kwh: float
    cost_savings: float
    implementation_cost: float
    priority: int
    confidence: float
    implementation_steps: List[str]
    prerequisites: List[str]
    annual_cost_savings: float = 0.0
    payback_period_months: Optional[float] = None

    @property
    def roi(self) -> float:
        """Annual return on investment, in percent."""
        if self.implementation_cost == 0:
            return float('inf')
        return (self.annual_cost_savings / self.implementation_cost) * 100

    def to_dict(self) -> Dict:
        """JSON-safe representation (no infinities)."""
        data = vars(self).copy()
        data['type'] = self.type.value
        data['roi'] = self.roi if np.isfinite(self.roi) else None
        return data


class EnergyOptimizer:
    """Energy optimization engine."""

    def __init__(self, electricity_rate: float = 0.12, assumptions: Optional[Dict] = None):
        """
        Initialize optimizer.

        Args:
            electricity_rate: Cost per kWh in dollars
            assumptions: Overrides for DEFAULT_ASSUMPTIONS
        """
        self.electricity_rate = electricity_rate
        self.assumptions = {**DEFAULT_ASSUMPTIONS, **(assumptions or {})}
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
        consumption_data = consumption_data.copy()
        consumption_data['timestamp'] = pd.to_datetime(consumption_data['timestamp'])
        consumption_data['hour'] = consumption_data['timestamp'].dt.hour
        consumption_data['day_of_week'] = consumption_data['timestamp'].dt.dayofweek
        consumption_data['month'] = consumption_data['timestamp'].dt.month
        
        timestamps = consumption_data['timestamp']
        analysis['period_days'] = (timestamps.max() - timestamps.min()).total_seconds() / 86400 + 1 / 24
        
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
            'weekend_vs_weekday_ratio': self._weekend_vs_weekday_ratio(consumption_data)
        }
        
        # Detect anomalies (consumption > 2 std dev from mean)
        mean_consumption = consumption_data['predicted_consumption_kwh'].mean()
        std_consumption = consumption_data['predicted_consumption_kwh'].std()
        threshold = mean_consumption + 2 * std_consumption
        
        anomalies = consumption_data[consumption_data['predicted_consumption_kwh'] > threshold]
        if not anomalies.empty:
            analysis['anomalies'] = anomalies[['timestamp', 'predicted_consumption_kwh']].to_dict('records')
        
        return analysis

    @staticmethod
    def _weekend_vs_weekday_ratio(consumption_data: pd.DataFrame) -> Optional[float]:
        """Average weekend-day total over average weekday total; None unless both occur."""
        daily_totals = consumption_data.groupby(consumption_data['timestamp'].dt.normalize())[
            'predicted_consumption_kwh'
        ].sum()
        is_weekend = daily_totals.index.dayofweek >= 5
        if not is_weekend.any() or is_weekend.all():
            return None
        return float(daily_totals[is_weekend].mean() / daily_totals[~is_weekend].mean())
    
    def generate_peak_load_shifting_recommendations(self, analysis: Dict,
                                                  building_metadata: Dict) -> List[OptimizationRecommendation]:
        """Generate recommendations for peak load shifting."""
        a = self.assumptions
        recommendations = []

        # Calculate potential savings from shifting peak loads
        peak_percentage = analysis['peak_usage']['percentage']
        if peak_percentage > 60:
            # Significant peak usage - recommend load shifting
            shiftable_load = analysis['peak_usage']['total_kwh'] * a['shiftable_peak_fraction']
            cost_savings = shiftable_load * self.electricity_rate * a['peak_offpeak_price_gap']

            recommendations.append(OptimizationRecommendation(
                recommendation_id="PLS001",
                type=OptimizationType.PEAK_LOAD_SHIFTING,
                title="Implement Peak Load Shifting Strategy",
                description=f"Shift {shiftable_load:.0f} kWh of load from peak hours to off-peak hours. "
                          f"Currently {peak_percentage:.1f}% of consumption occurs during peak hours.",
                impact_kwh=shiftable_load,
                cost_savings=cost_savings,
                implementation_cost=a['smart_controls_cost'],
                priority=1,
                confidence=0.85,
                implementation_steps=[
                    "Install smart building management system",
                    "Program automated load shifting schedules",
                    "Train staff on peak load management",
                    "Monitor and optimize shifting patterns"
                ],
                prerequisites=["Smart meter installation", "Flexible equipment schedules"]
            ))

        # Time-of-use optimization
        if analysis['daily_patterns']['peak_hour_consumption'] > analysis['daily_patterns']['lowest_hour_consumption'] * 3:
            shifted = analysis['peak_usage']['total_kwh'] * a['tou_shift_fraction']
            recommendations.append(OptimizationRecommendation(
                recommendation_id="PLS002",
                type=OptimizationType.PEAK_LOAD_SHIFTING,
                title="Time-of-Use Rate Optimization",
                description="Enroll in time-of-use electricity rates and adjust consumption patterns",
                impact_kwh=shifted,
                cost_savings=shifted * self.electricity_rate * a['tou_price_gap'],
                implementation_cost=a['tou_enrollment_cost'],
                priority=2,
                confidence=0.9,
                implementation_steps=[
                    "Contact utility for time-of-use rate enrollment",
                    "Install real-time energy monitoring",
                    "Create consumption scheduling guidelines",
                    "Educate occupants on rate structure"
                ],
                prerequisites=["Utility offers TOU rates"]
            ))

        return recommendations

    def generate_hvac_optimization_recommendations(self, analysis: Dict,
                                                  building_metadata: Dict,
                                                  weather_data: Optional[pd.DataFrame] = None) -> List[OptimizationRecommendation]:
        """Generate HVAC optimization recommendations."""
        a = self.assumptions
        recommendations = []

        # Temperature setpoint optimization
        hvac_peak_kwh = analysis['peak_usage']['total_kwh'] * a['hvac_load_share']
        setpoint_kwh = hvac_peak_kwh * a['setpoint_hvac_savings']

        recommendations.append(OptimizationRecommendation(
            recommendation_id="HVAC001",
            type=OptimizationType.HVAC_OPTIMIZATION,
            title="Optimize HVAC Temperature Setpoints",
            description="Adjust temperature setpoints by 2°F during peak hours to reduce HVAC load",
            impact_kwh=setpoint_kwh,
            cost_savings=setpoint_kwh * self.electricity_rate,
            implementation_cost=a['setpoint_cost'],
            priority=1,
            confidence=0.95,
            implementation_steps=[
                "Program smart thermostats with optimized setpoints",
                "Create seasonal setpoint schedules",
                "Monitor occupant comfort levels",
                "Fine-tune based on feedback"
            ],
            prerequisites=["Programmable thermostats"]
        ))

        # Weekend/holiday scheduling
        weekend_ratio = analysis['weekly_patterns']['weekend_vs_weekday_ratio']
        if weekend_ratio is not None and weekend_ratio > 0.7:
            weekend_savings = (weekend_ratio - 0.5) * \
                analysis['off_peak_usage']['total_kwh'] * a['weekend_offpeak_hvac_fraction']

            recommendations.append(OptimizationRecommendation(
                recommendation_id="HVAC002",
                type=OptimizationType.HVAC_OPTIMIZATION,
                title="Implement Weekend HVAC Scheduling",
                description="Reduce HVAC operation during weekends when building occupancy is lower",
                impact_kwh=weekend_savings,
                cost_savings=weekend_savings * self.electricity_rate,
                implementation_cost=a['weekend_schedule_cost'],
                priority=2,
                confidence=0.9,
                implementation_steps=[
                    "Program weekend setback schedules",
                    "Verify occupancy patterns",
                    "Create override procedures for special events",
                    "Monitor energy savings"
                ],
                prerequisites=["Programmable HVAC controls"]
            ))

        return recommendations

    def generate_schedule_optimization_recommendations(self, analysis: Dict,
                                                     building_metadata: Dict) -> List[OptimizationRecommendation]:
        """Generate operational schedule optimization recommendations."""
        a = self.assumptions
        recommendations = []

        # Early morning startup optimization: average kWh used in hours 5-7 each day
        early_hours = [5, 6, 7]
        early_consumption = sum(analysis['daily_patterns']['hourly_averages'].get(h, 0) for h in early_hours)

        if early_consumption > analysis['daily_patterns']['lowest_hour_consumption'] * 6:
            startup_kwh = early_consumption * a['startup_savings'] * analysis['period_days']
            recommendations.append(OptimizationRecommendation(
                recommendation_id="SCH001",
                type=OptimizationType.SCHEDULE_OPTIMIZATION,
                title="Optimize Morning Startup Sequence",
                description="Stagger equipment startup to reduce morning demand spike",
                impact_kwh=startup_kwh,
                cost_savings=startup_kwh * self.electricity_rate * a['startup_demand_premium'],
                implementation_cost=a['startup_cost'],
                priority=2,
                confidence=0.8,
                implementation_steps=[
                    "Analyze equipment startup sequences",
                    "Create staggered startup schedule",
                    "Program automated controls",
                    "Train maintenance staff"
                ],
                prerequisites=["Equipment with programmable controls"]
            ))

        return recommendations

    def generate_equipment_upgrade_recommendations(self, analysis: Dict,
                                                 building_metadata: Dict) -> List[OptimizationRecommendation]:
        """Generate equipment upgrade recommendations."""
        a = self.assumptions
        recommendations = []
        floor_area = building_metadata.get('floor_area_sqm') or 5000

        # LED lighting upgrade
        building_age = building_metadata.get('building_age')
        if building_age is None and building_metadata.get('year_built'):
            building_age = datetime.now().year - building_metadata['year_built']
        if building_age is None:
            building_age = a['led_min_building_age']

        if building_age > a['led_min_building_age']:
            led_kwh = analysis['peak_usage']['total_kwh'] * a['lighting_load_share'] * a['led_lighting_savings']
            recommendations.append(OptimizationRecommendation(
                recommendation_id="EQP001",
                type=OptimizationType.EQUIPMENT_UPGRADE,
                title="LED Lighting Retrofit",
                description="Replace traditional lighting with LED fixtures for 50% energy reduction",
                impact_kwh=led_kwh,
                cost_savings=led_kwh * self.electricity_rate,
                implementation_cost=floor_area * a['led_cost_per_sqm'],
                priority=3,
                confidence=0.95,
                implementation_steps=[
                    "Conduct lighting audit",
                    "Select appropriate LED replacements",
                    "Phase installation by area",
                    "Dispose of old fixtures properly"
                ],
                prerequisites=["Building has non-LED lighting"]
            ))

        # High-efficiency HVAC
        energy_star = building_metadata.get('energy_star_rating')
        if energy_star is None or pd.isna(energy_star):
            energy_star = 50
        if energy_star < a['hvac_upgrade_max_energy_star']:
            hvac_upgrade_kwh = analysis['peak_usage']['total_kwh'] * a['hvac_load_share'] * a['hvac_upgrade_savings']

            recommendations.append(OptimizationRecommendation(
                recommendation_id="EQP002",
                type=OptimizationType.EQUIPMENT_UPGRADE,
                title="High-Efficiency HVAC System Upgrade",
                description="Replace aging HVAC equipment with high-efficiency models",
                impact_kwh=hvac_upgrade_kwh,
                cost_savings=hvac_upgrade_kwh * self.electricity_rate,
                implementation_cost=floor_area * a['hvac_upgrade_cost_per_sqm'],
                priority=4,
                confidence=0.85,
                implementation_steps=[
                    "Perform HVAC system assessment",
                    "Size new equipment properly",
                    "Select ENERGY STAR certified models",
                    "Schedule phased replacement"
                ],
                prerequisites=["HVAC system > 15 years old"]
            ))

        return recommendations

    def generate_behavioral_recommendations(self, analysis: Dict,
                                          building_metadata: Dict) -> List[OptimizationRecommendation]:
        """Generate behavioral change recommendations."""
        a = self.assumptions
        behavioral_savings = analysis['peak_usage']['total_kwh'] * a['behavioral_peak_savings']

        return [OptimizationRecommendation(
            recommendation_id="BEH001",
            type=OptimizationType.BEHAVIORAL_CHANGE,
            title="Energy Awareness Campaign",
            description="Implement occupant education program to reduce energy waste",
            impact_kwh=behavioral_savings,
            cost_savings=behavioral_savings * self.electricity_rate,
            implementation_cost=a['behavioral_program_cost'],
            priority=2,
            confidence=0.7,
            implementation_steps=[
                "Create energy awareness materials",
                "Conduct training sessions",
                "Install real-time energy displays",
                "Implement recognition program"
            ],
            prerequisites=["Management support"]
        )]

    def generate_renewable_recommendations(self, analysis: Dict,
                                         building_metadata: Dict) -> List[OptimizationRecommendation]:
        """Generate renewable energy recommendations."""
        a = self.assumptions
        recommendations = []

        # Solar PV installation
        if str(building_metadata.get('building_type', '')).lower() in a['solar_building_types']:
            daily_consumption = sum(analysis['daily_patterns']['hourly_averages'].values())
            daily_offset_kwh = daily_consumption * a['solar_offset_fraction']
            system_size_kw = daily_offset_kwh / a['solar_peak_sun_hours']
            period_kwh = daily_offset_kwh * analysis['period_days']

            recommendations.append(OptimizationRecommendation(
                recommendation_id="REN001",
                type=OptimizationType.RENEWABLE_INTEGRATION,
                title="Rooftop Solar PV Installation",
                description=f"Install a {system_size_kw:.0f} kW solar array to offset "
                            f"{a['solar_offset_fraction'] * 100:.0f}% of electricity consumption",
                impact_kwh=period_kwh,
                cost_savings=period_kwh * self.electricity_rate,
                implementation_cost=system_size_kw * a['solar_cost_per_kw'],
                priority=5,
                confidence=0.8,
                implementation_steps=[
                    "Conduct solar feasibility study",
                    "Obtain permits and approvals",
                    "Select qualified installer",
                    "Apply for incentives and tax credits"
                ],
                prerequisites=["Suitable roof space", "Structural capacity"]
            ))

        return recommendations

    def _finalize(self, recommendation: OptimizationRecommendation, annualization_factor: float):
        """Fill in annual savings and payback from the period savings and cost."""
        recommendation.annual_cost_savings = recommendation.cost_savings * annualization_factor
        if recommendation.annual_cost_savings > 0:
            recommendation.payback_period_months = (
                recommendation.implementation_cost / recommendation.annual_cost_savings * 12
            )
        else:
            recommendation.payback_period_months = None

    def generate_optimization_report(self, consumption_data: pd.DataFrame,
                                   building_metadata: Dict,
                                   weather_data: Optional[pd.DataFrame] = None) -> Dict:
        """Generate comprehensive optimization report."""
        # Analyze consumption patterns
        analysis = self.analyze_consumption_patterns(consumption_data, building_metadata)

        # Generate recommendations by category
        all_recommendations = []
        all_recommendations.extend(self.generate_peak_load_shifting_recommendations(analysis, building_metadata))
        all_recommendations.extend(
            self.generate_hvac_optimization_recommendations(analysis, building_metadata, weather_data)
        )
        all_recommendations.extend(self.generate_schedule_optimization_recommendations(analysis, building_metadata))
        all_recommendations.extend(self.generate_equipment_upgrade_recommendations(analysis, building_metadata))
        all_recommendations.extend(self.generate_behavioral_recommendations(analysis, building_metadata))
        all_recommendations.extend(self.generate_renewable_recommendations(analysis, building_metadata))

        # Savings are computed over the analysed period; scale them to a full year
        annualization_factor = 365 / analysis['period_days']
        for rec in all_recommendations:
            self._finalize(rec, annualization_factor)

        # Sort by priority and ROI
        all_recommendations.sort(key=lambda x: (x.priority, -x.roi))

        # Calculate total potential savings
        # Measures overlap (several act on the same HVAC load), so their savings don't add up.
        # Apply them one after another, each to the consumption the previous ones left.
        total_consumption = consumption_data['predicted_consumption_kwh'].sum()
        remaining_fraction = 1.0
        for rec in all_recommendations:
            if total_consumption > 0:
                remaining_fraction *= max(0.0, 1 - rec.impact_kwh / total_consumption)
        total_kwh_savings = total_consumption * (1 - remaining_fraction)
        standalone_kwh_savings = sum(rec.impact_kwh for rec in all_recommendations)
        overlap_factor = total_kwh_savings / standalone_kwh_savings if standalone_kwh_savings > 0 else 0.0
        total_annual_savings = sum(rec.annual_cost_savings for rec in all_recommendations) * overlap_factor
        total_implementation_cost = sum(rec.implementation_cost for rec in all_recommendations)
        paybacks = [rec.payback_period_months for rec in all_recommendations
                    if rec.payback_period_months is not None]

        report = {
            'analysis': analysis,
            'recommendations': [rec.to_dict() for rec in all_recommendations],
            'summary': {
                'total_recommendations': len(all_recommendations),
                'total_kwh_savings_potential': total_kwh_savings,
                'total_cost_savings_annual': total_annual_savings,
                'total_implementation_cost': total_implementation_cost,
                'average_payback_months': float(np.mean(paybacks)) if paybacks else None,
                'savings_percentage': (total_kwh_savings / total_consumption) * 100 if total_consumption > 0 else 0.0,
                # Per-recommendation savings are standalone; the totals above account for overlap
                'standalone_kwh_savings_sum': standalone_kwh_savings
            },
            'assumptions': dict(self.assumptions, electricity_rate=self.electricity_rate),
            'recommendations_by_type': {}
        }

        # Group by type
        for opt_type in OptimizationType:
            type_recs = [rec for rec in all_recommendations if rec.type == opt_type]
            if type_recs:
                report['recommendations_by_type'][opt_type.value] = {
                    'count': len(type_recs),
                    'total_savings_kwh': sum(rec.impact_kwh for rec in type_recs),
                    'total_cost_savings': sum(rec.annual_cost_savings for rec in type_recs)
                }

        return report
