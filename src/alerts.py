"""
AgriShield Quantum - Bilingual Early Alert & Farmer Notification Service (Phase 3)
Generates targeted SMS and WhatsApp advisories in English and Telugu for high-risk farm plots.
Includes realistic farmer persona modeling for Krishna District, Andhra Pradesh.

NOTE: Simulation / Preview only - no live external telecommunication dispatch.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from src.utils import load_cache_json


FARMER_PERSONA = {
    "name_en": "Venkata Subba Rao",
    "name_te": "వెంకట సుబ్బారావు",
    "village_en": "Avanigadda, Krishna District, Andhra Pradesh",
    "village_te": "అవనిగడ్డ, కృష్ణా జిల్లా, ఆంధ్రప్రదేశ్",
    "crop_en": "Kharif Paddy (BPT 5204 Samba Mahsuri)",
    "crop_te": "ఖరీఫ్ వరి (BPT 5204 సాంబ మసూరి)",
    "landholding": "4.5 Acres (monitored in 10x10 sub-plots)",
    "primary_rbk": "Avanigadda Rythu Bharosa Kendra (RBK #14)",
    "contact_masked": "+91 98480 •••••",
    "preferred_language": "Telugu (తెలుగు)"
}


def get_farmer_persona() -> Dict[str, Any]:
    """Return agricultural persona details."""
    return FARMER_PERSONA


def create_sms_alert(
    cell_id: str,
    risk_score: float,
    risk_tier: str,
    date: str,
    humidity: float,
    ndre: float,
    lang: str = "en"
) -> str:
    """
    Format standard SMS text preview (under 160 chars in English, or UTF-8 Telugu).
    """
    pct = int(risk_score * 100)
    if lang.lower() == "te":
        if risk_tier == "High":
            return (
                f"🚨 [AgriShield హెచ్చరిక] పొలం ప్లాట్ {cell_id}: తీవ్ర ముప్పు ({pct}%). "
                f"తేమ {humidity:.0f}%, క్లోరోఫిల్ లోపం కనుగొనబడింది. "
                f"దయచేసి 24 గంటల్లోగా తనిఖీ చేసి, వేపనూనె లేదా సిఫార్సు చేసిన మందు పిచికారీ చేయండి."
            )
        else:
            return (
                f"⚠️ [AgriShield సమాచారం] పొలం ప్లాట్ {cell_id}: మధ్యస్థ ముప్పు ({pct}%). "
                f"48 గంటల్లోగా పరిశీలించండి."
            )
    else:
        if risk_tier == "High":
            return (
                f"🚨 [AgriShield Alert] Plot {cell_id}: HIGH RISK ({pct}%). "
                f"Hum: {humidity:.0f}%, NDRE drop detected. "
                f"Inspect within 24h & apply preventive bio-fungicide. Call RBK-14."
            )
        else:
            return (
                f"⚠️ [AgriShield Advisory] Plot {cell_id}: MODERATE RISK ({pct}%). "
                f"Microclimate humidity elevated. Monitor plot within 48h."
            )


def create_whatsapp_alert(
    cell_id: str,
    risk_score: float,
    risk_tier: str,
    date: str,
    humidity: float,
    ndvi: float,
    ndre: float,
    lang: str = "en"
) -> str:
    """
    Format rich WhatsApp preview with emoji structure, agronomic diagnostics,
    and actionable extension advice.
    """
    pct = int(risk_score * 100)
    if lang.lower() == "te":
        return f"""🌾 *AgriShield Quantum - పంట సంరక్షణ హెచ్చరిక* 🌾
---------------------------------------------
👤 *రైతు:* వెంకట సుబ్బారావు గారు
📍 *పొలం వివరాలు:* ప్లాట్ ID `{cell_id}` (అవనిగడ్డ, కృష్ణా జిల్లా)
📅 *తేదీ:* {date}
⚠️ *ముప్పు స్థాయి:* *{risk_tier.upper()} ({pct}%)*

🔬 *క్వాంటం AI పరిశీలన:*
• ఎరుపు అంచు సూచిక (NDRE): `{ndre:.3f}` (ముందస్తు క్షీణత)
• పచ్చదనం సూచిక (NDVI): `{ndvi:.3f}`
• గాలిలో తేమ: `{humidity:.1f}%` (శిలీంధ్ర వ్యాప్తికి అనుకూలం)

🛡️ *తీసుకోవాల్సిన చర్యలు:*
1. రాబోయే *24 నుండి 36 గంటల్లోగా* సదరు ప్లాట్ ను స్వయంగా పరిశీలించండి.
2. ఆకుమచ్చ / అగ్గి తెగులు ప్రారంభ లక్షణాలను గమనించండి.
3. అవసరమైతే వెంటనే స్థానిక RBK వ్యవసాయ అధికారిని సంప్రదించండి.

_రైతు భరోసా కేంద్రం (RBK #14) భాగస్వామ్యంతో AgriShield Quantum_"""
    else:
        return f"""🌾 *AgriShield Quantum - Early Disease Advisory* 🌾
---------------------------------------------
👤 *Farmer:* Venkata Subba Rao
📍 *Location:* Plot ID `{cell_id}` (Avanigadda, Krishna District)
📅 *Date:* {date}
⚠️ *Risk Level:* *{risk_tier.upper()} ({pct}%)*

🔬 *Quantum AI Diagnostics:*
• Red-Edge Index (NDRE): `{ndre:.3f}` (Pre-symptomatic stress anomaly)
• Canopy Greenness (NDVI): `{ndvi:.3f}`
• Canopy Humidity: `{humidity:.1f}%` (Pathogen proliferation window)

🛡️ *Recommended Action:*
1. Inspect the indicated sub-plot within *24 to 36 hours*.
2. Verify for initial leaf blast / sheath blight lesions.
3. Apply prophylactic neem-based spray or consult your RBK agronomist.

_AgriShield Quantum in cooperation with Rythu Bharosa Kendra (RBK #14)_"""


def get_weekly_alerts(week_idx: int) -> List[Dict[str, Any]]:
    """Fetch high and medium risk alerts for a selected week."""
    risk_maps = load_cache_json("risk_maps.json")
    if not risk_maps or str(week_idx) not in risk_maps:
        return []
    
    week_data = risk_maps[str(week_idx)]
    alerts = []
    
    for cell in week_data["cells"]:
        if cell["final_risk"] >= 0.50:  # Flag elevated risk
            alerts.append({
                "cell_id": cell["cell_id"],
                "row": cell["row"],
                "col": cell["col"],
                "risk_score": cell["final_risk"],
                "risk_tier": cell["risk_tier"],
                "sms_en": create_sms_alert(cell["cell_id"], cell["final_risk"], cell["risk_tier"], week_data["date"], cell["humidity"], cell["ndre"], "en"),
                "sms_te": create_sms_alert(cell["cell_id"], cell["final_risk"], cell["risk_tier"], week_data["date"], cell["humidity"], cell["ndre"], "te"),
                "whatsapp_en": create_whatsapp_alert(cell["cell_id"], cell["final_risk"], cell["risk_tier"], week_data["date"], cell["humidity"], cell["ndvi"], cell["ndre"], "en"),
                "whatsapp_te": create_whatsapp_alert(cell["cell_id"], cell["final_risk"], cell["risk_tier"], week_data["date"], cell["humidity"], cell["ndvi"], cell["ndre"], "te")
            })
            
    # Sort by descending risk
    alerts.sort(key=lambda x: x["risk_score"], reverse=True)
    return alerts
