"""Convert a pseudo-CT (Hounsfield-like units) into k-Wave acoustic medium
properties (density, sound speed, attenuation).

Formulas and constants follow the companion tool to mr-to-pct from the same
research group, tussim_skull_3D.m in
https://github.com/sitiny/BRIC_TUS_Simulation_Tools, which in turn cites
Marsac et al. 2017 and Bancel et al. 2021 for the density/sound-speed
relations, F. A. Duck 2013 for max skull sound speed, and Robertson et al.
2017 / Fry 1978 for the attenuation relation.
"""

from dataclasses import dataclass

import numpy as np

# Medium parameters (tussim_skull_3D.m lines 128-135)
C_MIN = 1500.0            # sound speed in water/soft tissue [m/s]
C_MAX = 3100.0            # max. speed of sound in skull (F. A. Duck, 2013) [m/s]
RHO_MIN = 1000.0          # density in water/soft tissue [kg/m^3]
RHO_MAX = 1900.0          # max. skull density [kg/m^3]
ALPHA_POWER = 1.43        # frequency power law exponent (Robertson et al., PMB 2017)
ALPHA_COEFF_WATER = 0.0   # [dB/(MHz^y cm)]
ALPHA_COEFF_MIN = 4.0     # [dB/(MHz cm)]
ALPHA_COEFF_MAX = 8.7     # [dB/(MHz cm)] (Fry 1978 at 0.5MHz)

HU_MIN_DEFAULT = 300      # minimum HU counted as skull (below this -> water)
HU_MAX_DEFAULT = 2000     # maximum HU for scaling (clipped above this)


@dataclass
class AcousticMaps:
    density: np.ndarray       # kg/m^3
    sound_speed: np.ndarray   # m/s
    alpha_coeff: np.ndarray   # dB/(MHz^y cm)
    alpha_power: float
    skull_mask: np.ndarray    # bool, True where treated as skull (not water)
    hu_min: int
    hu_max: int


def pct_to_acoustic_maps(pct_hu: np.ndarray, hu_min: int = HU_MIN_DEFAULT, hu_max: int = HU_MAX_DEFAULT) -> AcousticMaps:
    hu_max = min(hu_max, int(np.max(pct_hu)))

    model = pct_hu.copy()
    model[model < hu_min] = 0.0        # below skull threshold -> treated as water
    model[model > hu_max] = hu_max     # clip

    skull_mask = model > 0

    density = RHO_MIN + (RHO_MAX - RHO_MIN) * (model - 0) / (hu_max - 0)
    sound_speed = C_MIN + (C_MAX - C_MIN) * (density - RHO_MIN) / (RHO_MAX - RHO_MIN)
    with np.errstate(invalid="ignore"):
        alpha_coeff = ALPHA_COEFF_MIN + (ALPHA_COEFF_MAX - ALPHA_COEFF_MIN) * np.sqrt(
            np.clip(1 - (model - hu_min) / (hu_max - hu_min), 0, None)
        )

    density[~skull_mask] = RHO_MIN
    sound_speed[~skull_mask] = C_MIN
    alpha_coeff[~skull_mask] = ALPHA_COEFF_WATER

    return AcousticMaps(
        density=density,
        sound_speed=sound_speed,
        alpha_coeff=alpha_coeff,
        alpha_power=ALPHA_POWER,
        skull_mask=skull_mask,
        hu_min=hu_min,
        hu_max=hu_max,
    )
