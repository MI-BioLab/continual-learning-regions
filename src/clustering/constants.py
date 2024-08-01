import math

"""
Constants of the dynamic scattering-based clustering algorithm.
"""

#3⁽¹/⁴⁾ * (2π)⁽⁻¹/²⁾
K = 0.525

#K * 2π
K_2_PI = K * 2 * math.pi

__all__ = ["K", "K_2_PI"]