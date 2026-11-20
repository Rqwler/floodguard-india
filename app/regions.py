"""The five flood-prone basins tracked by FloodGuard India.

Coordinates are deliberately placed on the GloFAS river network so that the
Open-Meteo Flood API returns the main-stem discharge rather than a dry grid
cell. They were selected by sampling the river channel around each city.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Region:
    id: str
    name: str
    state: str
    area: str
    river: str
    station: str
    latitude: float
    longitude: float
    color: str
    description: str

    def to_dict(self) -> dict:
        return asdict(self)


REGIONS: tuple[Region, ...] = (
    Region(
        id="assam-brahmaputra",
        name="Assam",
        state="Assam",
        area="Brahmaputra Valley",
        river="Brahmaputra",
        station="Tezpur",
        latitude=26.625,
        longitude=92.825,
        color="#38bdf8",
        description=(
            "The Brahmaputra carries enormous monsoon flows through densely "
            "populated floodplains. Bank erosion and embankment breaches make "
            "Tezpur a critical watch point."
        ),
    ),
    Region(
        id="bihar-ganga",
        name="Bihar",
        state="Bihar",
        area="Middle Ganga Plain",
        river="Ganga",
        station="Patna",
        latitude=25.625,
        longitude=85.175,
        color="#a78bfa",
        description=(
            "Bihar's low-lying Gangetic plain is exposed to main-stem Ganga "
            "floods and tributary backflow. Patna anchors the state's most "
            "consequential flood corridor."
        ),
    ),
    Region(
        id="kerala-periyar",
        name="Kerala",
        state="Kerala",
        area="Periyar Basin",
        river="Periyar",
        station="Aluva",
        latitude=10.125,
        longitude=76.375,
        color="#34d399",
        description=(
            "Intense coastal monsoon bursts and reservoir releases can raise "
            "the Periyar rapidly. Aluva is the downstream gateway to the "
            "Kochi urban agglomeration."
        ),
    ),
    Region(
        id="mumbai-ulhas",
        name="Mumbai",
        state="Maharashtra",
        area="Mumbai Metropolitan Region",
        river="Ulhas",
        station="Kalyan",
        latitude=19.275,
        longitude=73.125,
        color="#fbbf24",
        description=(
            "Mumbai combines extreme rainfall with highly impervious urban "
            "surfaces. The Ulhas basin and Mithi river network are key flood "
            "sentinels for the metropolitan region."
        ),
    ),
    Region(
        id="westbengal-hooghly",
        name="West Bengal",
        state="West Bengal",
        area="Hooghly Estuary",
        river="Hooghly",
        station="Kolkata",
        latitude=22.925,
        longitude=88.375,
        color="#fb7185",
        description=(
            "The Hooghly estuary faces a compound threat of upstream discharge, "
            "monsoon rain and tidal surges. Kolkata is the downstream gauge for "
            "this densely settled delta."
        ),
    ),
)

REGION_BY_ID = {region.id: region for region in REGIONS}
REGION_IDS = [region.id for region in REGIONS]
REGION_INDEX = {region.id: index for index, region in enumerate(REGIONS)}


def get_region(region_id: str) -> Region:
    try:
        return REGION_BY_ID[region_id]
    except KeyError as exc:
        raise KeyError(f"Unknown region: {region_id}") from exc
