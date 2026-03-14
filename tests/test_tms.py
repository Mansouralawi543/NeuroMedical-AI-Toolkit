import pytest
import numpy as np
from src.modeling.tms_simulation import TMSSimulator

def test_tms_field_calculation():
    """
    Validates E-field calculation magnitude and shape.
    """
    sim = TMSSimulator()
    
    # Coil at (0, 0, 10cm)
    coil_pos = np.array([0, 0, 0.1])
    # Target points directly below the coil (on axis) should have zero field due to symmetry (cross product)
    target_pts = np.array([[0, 0, 0.05]])
    
    e_fields = sim.calculate_e_field(coil_pos, 1e8, target_pts)
    
    # On axis (Z-axis), with Z-oriented coil normal, the cross product should be 0.
    np.testing.assert_allclose(e_fields, 0, atol=1e-7)

def test_tms_field_magnitude():
    """
    Checks that field decays with distance.
    """
    sim = TMSSimulator()
    coil_pos = np.array([0, 0, 0.1])
    
    # Points at different distances
    pt_near = np.array([[0.01, 0, 0.09]]) # 1cm away
    pt_far = np.array([[0.01, 0, 0.01]])  # 9cm away
    
    e_near = sim.calculate_e_field(coil_pos, 1e8, pt_near)
    e_far = sim.calculate_e_field(coil_pos, 1e8, pt_far)
    
    mag_near = np.linalg.norm(e_near)
    mag_far = np.linalg.norm(e_far)
    
    assert mag_near > mag_far, "Field should decay with distance"

def test_induced_current():
    """
    Checks Ohm's Law implementation.
    """
    sigma = 0.5
    sim = TMSSimulator(tissue_conductivity=sigma)
    e_field = np.array([[1.0, 2.0, 3.0]])
    
    j_field = sim.compute_induced_current_density(e_field)
    
    np.testing.assert_allclose(j_field, sigma * e_field)
