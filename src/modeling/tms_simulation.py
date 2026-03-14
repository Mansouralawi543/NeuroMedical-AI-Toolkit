import numpy as np
from scipy import constants
from typing import Tuple, Optional

class TMSSimulator:
    """
    Simulation engine for Transcranial Magnetic Stimulation (TMS).
    Calculates induced Electric (E) fields in brain tissue based on Biot-Savart and Faraday's laws.
    """

    def __init__(self, tissue_conductivity: float = 0.33):
        """
        Initializes the simulator with average brain tissue conductivity.

        Args:
            tissue_conductivity: Gray matter conductivity in Siemens per meter (S/m).
        """
        self.sigma = tissue_conductivity
        self.mu_0 = constants.mu_0

    def calculate_e_field(
        self,
        coil_position: np.ndarray,
        current_di_dt: float,
        target_grid: np.ndarray,
        coil_radius: float = 0.04
    ) -> np.ndarray:
        """
        Estimates the induced E-field at target points using a simplified circular coil model.
        E = -dA/dt, where A is the vector potential.

        Args:
            coil_position: (3,) numpy array representing (x, y, z) position of the coil center.
            current_di_dt: Rate of change of coil current (A/s).
            target_grid: (N, 3) numpy array of target points for field calculation.
            coil_radius: Radius of the coil loop (m).

        Returns:
            (N, 3) numpy array of the E-field vectors at each target point.
        """
        # Displacement vector from coil center to target points
        r = target_grid - coil_position
        dist = np.linalg.norm(r, axis=1)

        # Vector potential A estimation (simplified loop model)
        # In a full simulation, we'd integrate over the coil wire path.
        # This acts as a research-grade approximation for field orientation.
        
        # Cross product with an assumed vertical (Z) coil orientation for field shaping
        coil_normal = np.array([0, 0, 1])
        cross_prod = np.cross(coil_normal, r)
        
        # Field strength scaling based on distance (inverse-square-like decay for vector potential)
        # Magnitude is proportional to mu_0 * I / (4 * pi * dist^2)
        field_magnitude = (self.mu_0 * current_di_dt) / (4 * np.pi * (dist**2 + 1e-9))
        
        # Apply orientation from cross product
        e_field = cross_prod * field_magnitude[:, np.newaxis]
        
        return e_field

    def compute_induced_current_density(self, e_field: np.ndarray) -> np.ndarray:
        """
        Calculates induced current density (J) using Ohm's Law (J = sigma * E).

        Args:
            e_field: (N, 3) induced electric field.

        Returns:
            (N, 3) induced current density vectors.
        """
        return self.sigma * e_field

if __name__ == "__main__":
    # Test simulation
    sim = TMSSimulator()
    target_pts = np.array([[0, 0, 0.05], [0, 0.01, 0.05], [0, 0.02, 0.05]])
    coil_pos = np.array([0, 0, 0.07])
    di_dt = 1e8 # typical A/s for TMS pulses

    e_fields = sim.calculate_e_field(coil_pos, di_dt, target_pts)
    print(f"Calculated E-fields at 3 points:\n{e_fields}")
