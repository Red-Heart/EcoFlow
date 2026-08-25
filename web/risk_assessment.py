"""
EcoFlow Risk Assessment Module

Implements the three risk categories described in the proposal:
1. Environmental & Health Risk - based on estimated CO2 emissions exceeding safety thresholds
2. Road Safety & Congestion Risk - based on concentration of heavy vehicles (bus + truck)
3. Operational Hardware Risk - static/contextual flag for edge hardware exposure

Thresholds below are illustrative starting points for a mini-project prototype.
For a MIROS-facing report, these should be calibrated against real traffic volume
and air-quality guideline data for the E37 SALAK highway specifically.
"""

from dataclasses import dataclass, field

# --- Thresholds (tune these based on your segment length / typical traffic volume) ---
CO2_LOW_THRESHOLD_KG = 2.0
CO2_MEDIUM_THRESHOLD_KG = 5.0
CO2_HIGH_THRESHOLD_KG = 10.0

HEAVY_VEHICLE_RATIO_LOW = 0.15
HEAVY_VEHICLE_RATIO_MEDIUM = 0.30

RISK_LEVELS = ["Low", "Medium", "High", "Critical"]


@dataclass
class RiskAssessment:
    environmental_risk: str = "Low"
    congestion_risk: str = "Low"
    hardware_risk: str = "Low"
    overall_risk: str = "Low"
    details: dict = field(default_factory=dict)


def assess_environmental_risk(total_co2_kg: float) -> str:
    if total_co2_kg >= CO2_HIGH_THRESHOLD_KG:
        return "Critical"
    elif total_co2_kg >= CO2_MEDIUM_THRESHOLD_KG:
        return "High"
    elif total_co2_kg >= CO2_LOW_THRESHOLD_KG:
        return "Medium"
    return "Low"


def assess_congestion_risk(class_counts: dict) -> str:
    """Heavy vehicles (bus + truck) produce more particulate matter and reduce visibility,
    increasing accident risk per the proposal's Road Safety & Congestion Risk description."""
    total_vehicles = sum(class_counts.values())
    if total_vehicles == 0:
        return "Low"

    heavy_count = class_counts.get("bus", 0) + class_counts.get("truck", 0)
    heavy_ratio = heavy_count / total_vehicles

    if heavy_ratio >= HEAVY_VEHICLE_RATIO_MEDIUM:
        return "High"
    elif heavy_ratio >= HEAVY_VEHICLE_RATIO_LOW:
        return "Medium"
    return "Low"


def assess_hardware_risk(is_outdoor_deployment: bool = True, has_weatherproof_enclosure: bool = True) -> str:
    """Static/contextual risk based on deployment conditions (proposal's Operational Hardware Risk).
    This isn't derived from video data -- it's a deployment-configuration flag you set per site."""
    if is_outdoor_deployment and not has_weatherproof_enclosure:
        return "High"
    elif is_outdoor_deployment and has_weatherproof_enclosure:
        return "Medium"
    return "Low"


def combine_overall_risk(environmental_risk: str, congestion_risk: str, hardware_risk: str) -> str:
    """Overall risk takes the highest severity across the three categories."""
    levels = [environmental_risk, congestion_risk, hardware_risk]
    for level in reversed(RISK_LEVELS):
        if level in levels:
            return level
    return "Low"


def build_risk_matrix(class_counts: dict, total_co2_kg: float,
                       is_outdoor_deployment: bool = True,
                       has_weatherproof_enclosure: bool = True) -> RiskAssessment:
    environmental_risk = assess_environmental_risk(total_co2_kg)
    congestion_risk = assess_congestion_risk(class_counts)
    hardware_risk = assess_hardware_risk(is_outdoor_deployment, has_weatherproof_enclosure)
    overall_risk = combine_overall_risk(environmental_risk, congestion_risk, hardware_risk)

    total_vehicles = sum(class_counts.values())
    heavy_count = class_counts.get("bus", 0) + class_counts.get("truck", 0)
    heavy_ratio = round(heavy_count / total_vehicles, 3) if total_vehicles else 0.0

    return RiskAssessment(
        environmental_risk=environmental_risk,
        congestion_risk=congestion_risk,
        hardware_risk=hardware_risk,
        overall_risk=overall_risk,
        details={
            "total_co2_kg": total_co2_kg,
            "total_vehicles": total_vehicles,
            "heavy_vehicle_count": heavy_count,
            "heavy_vehicle_ratio": heavy_ratio,
        },
    )


if __name__ == "__main__":
    example_counts = {"car": 42, "motorcycle": 18, "bus": 6, "truck": 9}
    example_total_co2 = 6.8

    result = build_risk_matrix(example_counts, example_total_co2)

    print(f"Environmental Risk : {result.environmental_risk}")
    print(f"Congestion Risk     : {result.congestion_risk}")
    print(f"Hardware Risk       : {result.hardware_risk}")
    print(f"Overall Risk        : {result.overall_risk}")
    print(f"Details             : {result.details}")
