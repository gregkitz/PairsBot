# Documentation Overview

This directory contains the documentation for the trading system. We've recently reorganized the documentation to make it more accessible, easier to maintain, and to address AI context window limitations.

## Documentation Structure

```
docs/
├── architecture_dir/         # System architecture and design documents
│   ├── PAIRS_DESIGN.md       # Overall design of the pairs trading system
│   ├── data_flow.md          # Data flow throughout the system
│   ├── component_dependencies.md # Dependencies between components
│   └── ...
│
├── context/                  # Documentation for AI context preservation
│   ├── implementation_status.md  # Current state of implementation
│   ├── implementation_notes.md   # Notes on implementation decisions
│   ├── ai_blindspots_mitigation.md # Addressing AI development blind spots
│   └── ...
│
├── plans/                    # Current and future work
│   ├── next_steps.md         # Checklist of next implementation steps
│   ├── intraday_ml_next_steps.md # Detailed next steps for the intraday ML system
│   └── ...
│
├── technical/                # Technical implementation documentation
│   ├── statistical_methods.md     # Statistical methods overview
│   ├── johansen_implementation.md # Johansen test implementation details
│   ├── engle_granger_implementation.md # Engle-Granger test implementation details
│   ├── cointegration_framework.md # Comprehensive cointegration framework
│   ├── statistical_validation_methods.md # Statistical validation approaches
│   ├── zscore_strategy_implementation.md # Z-Score strategy details
│   └── ...
│
├── user_manual/              # End-user documentation
│   ├── intraday_ml_system_user_guide.md # User guide for intraday ML system
│   ├── troubleshooting_guide.md # Troubleshooting common issues
│   └── ...
│
├── components/               # Component-specific documentation
│   └── ...
│
├── README.md                 # This file
├── documentation_guide.md    # Detailed guide to the documentation structure
└── ... other documentation files
```

## Key Documentation Files

For a complete overview of the documentation structure and how to use it, see [documentation_guide.md](documentation_guide.md).

### Most Important Documents

1. **Architecture Design**: [PAIRS_DESIGN.md](architecture_dir/PAIRS_DESIGN.md)
2. **Current Status**: [implementation_status.md](context/implementation_status.md)
3. **Current Work**: [next_steps.md](plans/next_steps.md)
4. **Implementation Notes**: [implementation_notes.md](context/implementation_notes.md)

### Technical Implementation Documents

1. **Statistical Methods**: [statistical_methods.md](technical/statistical_methods.md)
2. **Cointegration Framework**: [cointegration_framework.md](technical/cointegration_framework.md)
3. **Johansen Test**: [johansen_implementation.md](technical/johansen_implementation.md)
4. **Engle-Granger Test**: [engle_granger_implementation.md](technical/engle_granger_implementation.md)
5. **Statistical Validation**: [statistical_validation_methods.md](technical/statistical_validation_methods.md)
6. **Z-Score Strategy**: [zscore_strategy_implementation.md](technical/zscore_strategy_implementation.md)

### User Guides

1. **Intraday ML System**: [intraday_ml_system_user_guide.md](user_manual/intraday_ml_system_user_guide.md)
2. **Troubleshooting**: [troubleshooting_guide.md](user_manual/troubleshooting_guide.md)

## Recent Changes to Documentation

We've reorganized the documentation to address several challenges:

1. **Context Preservation**: Created a dedicated `context/` folder to preserve important implementation context between AI sessions.

2. **Architecture Documentation**: Consolidated design documents in `architecture_dir/` to provide a clear overview of the system.

3. **Work Planning**: Centralized next steps and planning documents in `plans/` to make it easier to track progress.

4. **Technical Documentation**: Added comprehensive documentation for statistical methods, cointegration tests, and implementation details in the `technical/` folder.

5. **User Guides**: Created a dedicated `user_manual/` folder for end-user documentation.

## Using This Documentation

The documentation is designed to support various use cases:

- **Understanding the System**: Start with the architecture documents
- **Current Development**: Check the implementation status and next steps
- **Technical Implementation**: Refer to the technical folder for implementation details
- **Maintaining Context**: Use the context folder for implementation details
- **End Users**: Refer to the user guides for operational instructions

## Documentation Updates

As the system evolves, the documentation will be updated to reflect changes. When making significant code changes, please make corresponding updates to relevant documentation.

## Additional Resources

- [Main README.md](../README.md) - Main project documentation
- [Audit Report](../audit/reports/audit_summary.md) - Summary of codebase audit findings 