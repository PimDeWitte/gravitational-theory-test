# Default Validations

This directory contains standard observational tests that all theories are validated against.

## Structure

Each validation file:
- Inherits from `GravitationalTheory` or uses the base validation framework
- Implements a specific observational test (e.g., pulsar timing, Shapiro delay)
- Returns standardized results for comparison
- Can be run independently or as part of the full validation suite

## Available Tests

- **pulsar_timing.py**: PSR B1913+16 periastron advance
- **shapiro_delay.py**: Solar system Shapiro time delay
- **mercury_perihelion.py**: Mercury's perihelion precession
- **cassini_ppn.py**: Cassini PPN-γ parameter constraints
- **gravitational_redshift.py**: Pound-Rebka experiment

All theories are automatically tested against these unless `--skip-validations` is specified.
