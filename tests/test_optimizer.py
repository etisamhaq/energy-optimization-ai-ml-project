"""
Tests for optimization engine.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
sys.path.append('..')
from src.optimization.optimizer import (
    EnergyOptimizer, OptimizationType, OptimizationRecommendation
)


class TestEnergyOptimizer:
    """Test cases for EnergyOptimizer class."""
    
    @pytest.fixture
    def optimizer(self):
        """Create EnergyOptimizer instance."""
        return EnergyOptimizer(electricity_rate=0.15)
    
    @pytest.fixture
    def consumption_data(self):
        """Create sample consumption data."""
        # Create hourly data for one week
        dates = pd.date_range('2024-01-01', periods=24*7, freq='H')
        
        # Create consumption pattern with peak hours
        consumption = []
        for date in dates:
            hour = date.hour
            # Higher consumption during business hours
            if 8 <= hour <= 18:
                base_consumption = 150
            else:
                base_consumption = 80
            
            # Add some variation
            consumption.append(base_consumption + np.random.normal(0, 10))
        
        df = pd.DataFrame({
            'timestamp': dates,
            'building_id': 'B001',
            'predicted_consumption_kwh': consumption
        })
        
        return df
    
    @pytest.fixture
    def building_metadata(self):
        """Create sample building metadata."""
        return {
            'building_id': 'B001',
            'building_type': 'office',
            'floor_area_sqm': 5000,
            'year_built': 2005,
            'occupancy_type': 'multi_tenant',
            'hvac_type': 'central',
            'energy_star_rating': 65,
            'building_age': 19
        }
    
    def test_analyze_consumption_patterns(self, optimizer, consumption_data, building_metadata):
        """Test consumption pattern analysis."""
        analysis = optimizer.analyze_consumption_patterns(consumption_data, building_metadata)
        
        # Check analysis structure
        assert 'peak_usage' in analysis
        assert 'off_peak_usage' in analysis
        assert 'daily_patterns' in analysis
        assert 'weekly_patterns' in analysis
        
        # Check peak usage analysis
        assert analysis['peak_usage']['total_kwh'] > 0
        assert analysis['peak_usage']['percentage'] > 50  # Should be higher during peak
        
        # Check daily patterns
        assert 0 <= analysis['daily_patterns']['peak_hour'] <= 23
        assert analysis['daily_patterns']['peak_hour_consumption'] > analysis['daily_patterns']['lowest_hour_consumption']
    
    def test_optimization_recommendation_dataclass(self):
        """Test OptimizationRecommendation dataclass."""
        rec = OptimizationRecommendation(
            recommendation_id="TEST001",
            type=OptimizationType.PEAK_LOAD_SHIFTING,
            title="Test Recommendation",
            description="Test description",
            impact_kwh=1000,
            cost_savings=120,
            implementation_cost=1000,
            priority=1,
            confidence=0.9,
            implementation_steps=["Step 1", "Step 2"],
            prerequisites=["Prereq 1"],
            annual_cost_savings=1440
        )
        
        # Test ROI calculation (annual savings over implementation cost)
        assert rec.roi == pytest.approx(144.0)
        assert rec.to_dict()['type'] == OptimizationType.PEAK_LOAD_SHIFTING.value
    
    def test_payback_is_calculated_from_annual_savings(self, optimizer, consumption_data, building_metadata):
        """Payback and ROI must follow from cost and savings, scaled from the analysed week to a year."""
        report = optimizer.generate_optimization_report(consumption_data, building_metadata)
        period_days = report['analysis']['period_days']
        
        for rec in report['recommendations']:
            assert rec['annual_cost_savings'] == pytest.approx(rec['cost_savings'] * 365 / period_days)
            assert rec['payback_period_months'] == pytest.approx(
                rec['implementation_cost'] / rec['annual_cost_savings'] * 12
            )
            assert rec['roi'] == pytest.approx(rec['annual_cost_savings'] / rec['implementation_cost'] * 100)
    
    def test_report_is_json_safe_without_weekend(self, optimizer, building_metadata):
        """A window with no weekend days must not produce NaN/inf (invalid JSON)."""
        import json
        thursday = pd.Timestamp('2024-01-04')
        data = pd.DataFrame({
            'timestamp': pd.date_range(thursday, periods=48, freq='H'),
            'predicted_consumption_kwh': np.random.uniform(50, 150, 48)
        })
        report = optimizer.generate_optimization_report(data, building_metadata)
        
        assert report['analysis']['weekly_patterns']['weekend_vs_weekday_ratio'] is None
        json.dumps(report, default=str, allow_nan=False)
    
    def test_weekend_ratio_uses_day_of_week(self, optimizer):
        """Weekend days at half the weekday load give a ratio of 0.5, whichever days are present."""
        hours = pd.date_range('2024-01-05', periods=24 * 3, freq='H')  # Fri, Sat, Sun
        data = pd.DataFrame({
            'timestamp': hours,
            'predicted_consumption_kwh': [50.0 if t.dayofweek >= 5 else 100.0 for t in hours]
        })
        assert optimizer._weekend_vs_weekday_ratio(data) == pytest.approx(0.5)
    
    def test_total_savings_account_for_overlap(self, optimizer, consumption_data, building_metadata):
        """Overlapping measures combine to less than the sum of their standalone savings."""
        report = optimizer.generate_optimization_report(consumption_data, building_metadata)
        summary = report['summary']
        standalone = sum(r['impact_kwh'] for r in report['recommendations'])
        
        assert summary['standalone_kwh_savings_sum'] == pytest.approx(standalone)
        assert 0 < summary['total_kwh_savings_potential'] < standalone
        total = consumption_data['predicted_consumption_kwh'].sum()
        expected_remaining = np.prod([1 - r['impact_kwh'] / total for r in report['recommendations']])
        assert summary['savings_percentage'] == pytest.approx((1 - expected_remaining) * 100)
    
    def test_assumptions_can_be_overridden(self, consumption_data, building_metadata):
        """Cheaper smart controls should shorten the load-shifting payback."""
        default = EnergyOptimizer(electricity_rate=0.15)
        cheaper = EnergyOptimizer(electricity_rate=0.15, assumptions={'smart_controls_cost': 1000.0})
        
        def payback(opt):
            report = opt.generate_optimization_report(consumption_data, building_metadata)
            return next(r for r in report['recommendations'] if r['recommendation_id'] == 'PLS001')['payback_period_months']
        
        assert payback(cheaper) == pytest.approx(payback(default) / 5)
    
    def test_generate_peak_load_shifting_recommendations(self, optimizer, consumption_data, building_metadata):
        """Test peak load shifting recommendations."""
        analysis = optimizer.analyze_consumption_patterns(consumption_data, building_metadata)
        recommendations = optimizer.generate_peak_load_shifting_recommendations(analysis, building_metadata)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Check recommendation structure
        rec = recommendations[0]
        assert rec.type == OptimizationType.PEAK_LOAD_SHIFTING
        assert rec.impact_kwh > 0
        assert rec.cost_savings > 0
        assert len(rec.implementation_steps) > 0
    
    def test_generate_hvac_optimization_recommendations(self, optimizer, consumption_data, building_metadata):
        """Test HVAC optimization recommendations."""
        analysis = optimizer.analyze_consumption_patterns(consumption_data, building_metadata)
        recommendations = optimizer.generate_hvac_optimization_recommendations(analysis, building_metadata)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Check HVAC specific recommendations
        rec = recommendations[0]
        assert rec.type == OptimizationType.HVAC_OPTIMIZATION
        assert "temperature" in rec.description.lower() or "hvac" in rec.description.lower()
    
    def test_generate_equipment_upgrade_recommendations(self, optimizer, consumption_data, building_metadata):
        """Test equipment upgrade recommendations."""
        analysis = optimizer.analyze_consumption_patterns(consumption_data, building_metadata)
        recommendations = optimizer.generate_equipment_upgrade_recommendations(analysis, building_metadata)
        
        assert isinstance(recommendations, list)
        
        # Should generate recommendations for old building
        if building_metadata['building_age'] > 10:
            assert len(recommendations) > 0
            
            # Check for LED recommendation
            led_recs = [r for r in recommendations if "LED" in r.title]
            assert len(led_recs) > 0
    
    def test_generate_optimization_report(self, optimizer, consumption_data, building_metadata):
        """Test complete optimization report generation."""
        report = optimizer.generate_optimization_report(consumption_data, building_metadata)
        
        # Check report structure
        assert 'analysis' in report
        assert 'recommendations' in report
        assert 'summary' in report
        assert 'recommendations_by_type' in report
        
        # Check summary
        summary = report['summary']
        assert summary['total_recommendations'] >= 0
        assert summary['total_kwh_savings_potential'] >= 0
        assert summary['total_cost_savings_annual'] >= 0
        assert 0 <= summary['savings_percentage'] <= 100
        
        # Check recommendations
        if report['recommendations']:
            rec = report['recommendations'][0]
            assert 'recommendation_id' in rec
            assert 'type' in rec
            assert 'priority' in rec
            assert 'impact_kwh' in rec
    
    def test_recommendation_prioritization(self, optimizer, consumption_data, building_metadata):
        """Test that recommendations are properly prioritized."""
        report = optimizer.generate_optimization_report(consumption_data, building_metadata)
        recommendations = report['recommendations']
        
        if len(recommendations) > 1:
            # Check that recommendations are sorted by priority (ascending) and ROI (descending)
            for i in range(1, len(recommendations)):
                rec1 = recommendations[i-1]
                rec2 = recommendations[i]
                
                # If same priority, ROI should be descending
                if rec1['priority'] == rec2['priority']:
                    # Calculate ROI from the recommendation data
                    if rec1['roi'] is not None and rec2['roi'] is not None:
                        assert rec1['roi'] >= rec2['roi']
    
    def test_electricity_rate_impact(self, consumption_data, building_metadata):
        """Test that electricity rate affects recommendations."""
        # Create optimizers with different rates
        optimizer_low = EnergyOptimizer(electricity_rate=0.10)
        optimizer_high = EnergyOptimizer(electricity_rate=0.20)
        
        report_low = optimizer_low.generate_optimization_report(consumption_data, building_metadata)
        report_high = optimizer_high.generate_optimization_report(consumption_data, building_metadata)
        
        # Higher electricity rate should result in higher savings
        assert report_high['summary']['total_cost_savings_annual'] > report_low['summary']['total_cost_savings_annual']