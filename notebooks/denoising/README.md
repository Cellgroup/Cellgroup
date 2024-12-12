# Tests for Cellgroup

This folder contains test files that check if the functions in Cellgroup work correctly. Each analysis tool has its own test file.

## Test Files

The test files match the module structure:

```
notebooks/
└── denoising/
    ├── test_gradients.py     - Tests for image gradient analysis
    ├── test_intensity.py     - Tests for light intensity measurements
    ├── test_frequencies.py   - Tests for frequency domain analysis
    ├── test_spatial.py       - Tests for spatial pattern analysis
    └── test_utils.py         - Tests for common utility functions
```

## Purpose

These tests verify that:
- Images load correctly
- Analysis functions give expected results 
- Calculations are accurate
- Functions handle errors properly

## Using Tests

You can run tests using pytest:

```bash
# Run all tests
pytest

# Test a specific module
pytest tests/denoising/test_gradients.py
```

For developers: When adding new features, add corresponding tests to ensure everything works as expected.