# Documentation Guide

This guide explains the organization of documentation in this project and how to find relevant information. We've structured our documentation into logical folders to make it easier to navigate and maintain.

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
└── ... other documentation files
```

## Key Documentation Files

### Architecture Documentation

- **PAIRS_DESIGN.md**: The foundational design document for the pairs trading system, including core components, implementation roadmap, and key considerations.

- **data_flow.md**: A detailed description of how data moves through the system, from data sources through processing to trading execution.

- **component_dependencies.md**: Outlines dependencies between system components and provides guidelines for avoiding circular dependencies.

### Context Documentation

- **implementation_status.md**: A snapshot of what has been implemented and what still needs to be done. This is useful for understanding the current state of the system.

- **implementation_notes.md**: Captures implementation decisions, thought processes, and notes. Particularly useful when implementations are interrupted.

- **ai_blindspots_mitigation.md**: Describes common AI blind spots in development and how we're addressing them in our system.

### Technical Documentation

- **statistical_methods.md**: Comprehensive overview of the statistical methods used in the system, focusing on cointegration techniques.

- **cointegration_framework.md**: Explains the complete cointegration framework with component interactions, validation approaches, and examples.

- **johansen_implementation.md**: Detailed documentation of the Johansen test implementation with mathematical foundation and pseudocode.

- **engle_granger_implementation.md**: Detailed documentation of the Engle-Granger test implementation with step-by-step procedures.

- **statistical_validation_methods.md**: Covers validation methodologies for ensuring robustness of cointegration relationships.

- **zscore_strategy_implementation.md**: Details the Z-Score Strategy implementation for pairs trading with code examples and usage patterns.

### Plans and Next Steps

- **next_steps.md**: A detailed checklist of current and upcoming work items. This is the primary reference for what to work on next.

## How to Use This Documentation

### For New Contributors

1. Start with **PAIRS_DESIGN.md** to understand the overall system design
2. Review **data_flow.md** to see how data moves through the system
3. Check **implementation_status.md** to understand what's been built
4. Consult **next_steps.md** to see current priorities
5. Explore the **technical** folder for implementation details on specific components

### For Developers Implementing Statistical Methods

1. Begin with **statistical_methods.md** to understand the mathematical foundations
2. Read the specific implementation docs (**johansen_implementation.md** or **engle_granger_implementation.md**)
3. Review **statistical_validation_methods.md** for robustness considerations
4. Refer to **cointegration_framework.md** to understand component interactions
5. Check **zscore_strategy_implementation.md** if working on trading logic

### During Development

1. Before implementing a new component, check **implementation_status.md** to avoid duplication
2. Document design decisions in **implementation_notes.md**
3. Update **next_steps.md** as tasks are completed
4. Keep context documentation updated to preserve knowledge between sessions
5. Add technical documentation for newly implemented components

### For Context Preservation

The `docs/context/` folder is specifically designed to preserve knowledge between coding sessions, addressing the limitation of the AI context window. When working on a component:

1. Check the relevant context documents before starting
2. Update them with new information as you progress
3. If interrupted, document your current approach and thought process
4. Reference these docs in the .cursorrules file for future sessions

## Documentation Guidelines

When updating or adding documentation:

1. **Be Concise**: Focus on essential information
2. **Be Accurate**: Ensure documentation reflects the current state of the system
3. **Cross-Reference**: Link to related documents where appropriate
4. **Code Examples**: Include relevant code snippets where they help understand concepts
5. **Update Regularly**: Keep documentation in sync with code changes
6. **Comprehensive Coverage**: For technical documents, include:
   - Theoretical foundation
   - Implementation details
   - Usage examples
   - Edge cases
   - References to academic literature when relevant

## Documentation Maintenance

The documentation structure is designed to grow with the project. As new components are added:

1. Update architecture documents to include new components
2. Create new technical documentation for implementations
3. Update user guides as needed
4. Maintain the implementation status document
5. Periodically review and consolidate documentation 