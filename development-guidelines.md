# Cellgroup Development Guidelines

## Code Organization and Style

### Object-Oriented Programming
- All major components must be implemented using OOP principles
- Create abstract base classes for common functionality using `abc` module
- Use inheritance to implement specific cases (e.g., different Dataset implementations)
- Leverage `pydantic` for data validation and `dataclasses` for data containers
- Keep class hierarchies shallow (max 2-3 levels) to maintain clarity

### Type Hints
- All function parameters and return values must have type hints
- Use type hints from `typing` module (e.g., `List`, `Dict`, `Optional`, `Union`)
- Add complex type definitions in a separate `types.py` file
- Document custom types with clear examples in docstrings
- Use `TypeVar` for generic types when appropriate

### Documentation
- All public classes and methods must have docstrings
- Follow NumPy docstring format for consistency
- Include parameters, return types, and examples in docstrings
- Add usage examples for complex functionality
- Temporary implementations can skip detailed documentation but must include TODO comments

### Testing (Future Implementation)
- Tests will be mandatory once core pipeline is stable
- Write unit tests for core functionality using pytest
- Include integration tests for end-to-end workflows
- Maintain test coverage above 80%
- Include example datasets for testing

### Code Style and Linting
- Maximum line length: 88 characters
- Function arguments indentation: 1 tab
- Use Black for code formatting
- Follow isort for import sorting
- Enable flake8 for additional style checks

## Development Workflow

### Pull Requests
- Keep PRs small (target < 100 lines)
- Include example notebooks for new features
- Branch naming convention: `namesurname/type/description`
  - Types: `feat`, `refac`, `fix`, `style`, `example`, `test`
  - Example: `jdoe/feat/add_cell_clustering`
- PR description must include:
  - Purpose of changes
  - How to test
  - Related issues
  - Breaking changes (if any)

### Code Review Process
- Reviews will be mandatory in later stages
- Assign appropriate reviewers based on code ownership
- Address all comments before merging
- Squash commits before merging

### Dependency Management
- Minimize external dependencies
- Prefer standard scientific Python stack:
  - numpy
  - torch
  - scikit-image
  - scipy
  - xarray
- Document all dependencies in `requirements.txt` and `setup.py`
- Include version constraints for critical dependencies

### TODO Management
- Mark incomplete/future work with "TODO" comments
- Include brief explanation and ticket number (if applicable)
- Format: `# TODO(username): description [TICKET-123]`
- Use "TODOS: TREE" extension for tracking
- Review TODOs regularly in team meetings

## Data Handling

### Coordinate System
- Use xarray.DataArray for all multi-dimensional data
- Include proper coordinate labels and units
- Document coordinate systems in metadata
- Provide conversion utilities between different coordinate systems

### Image Processing
- Implement image processing pipelines as composable operations
- Cache intermediate results when beneficial
- Include progress bars for long-running operations
- Add proper error handling for edge cases

### Memory Management
- Implement lazy loading for large datasets
- Include memory usage estimates in documentation
- Provide memory-efficient alternatives for large-scale processing
- Add proper cleanup methods for temporary files

## Best Practices

1. Write self-documenting code using clear variable and function names
2. Include logging at appropriate levels
3. Handle errors gracefully with informative messages
4. Write modular, reusable components
5. Include performance benchmarks for critical operations
6. Document any assumptions in the code
7. Add deprecation warnings for features planned for removal
8. Maintain backward compatibility when possible

## Segmentation-Specific Guidelines

1. Implement pipeline stages as separate, composable modules
2. Include visualization tools for debugging
3. Document model parameters and their effects
4. Provide examples with different cell types
5. Include metrics for evaluation
6. Add proper validation steps
