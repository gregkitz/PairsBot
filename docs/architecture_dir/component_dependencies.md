# Component Dependencies

This document outlines the dependencies between major components in the system and provides guidelines for avoiding circular dependencies that can lead to maintenance issues.

## Dependency Hierarchy

The system follows a layered architecture with the following hierarchy (from lowest to highest):

1. **Data Management Layer**
   - Data collection, storage, and retrieval
   - Raw data processing and normalization

2. **Analysis Layer**
   - Feature engineering
   - Statistical analysis
   - Machine learning models

3. **Strategy Layer**
   - Signal generation
   - Trade decision logic
   - Position sizing

4. **Execution Layer**
   - Order management
   - Risk controls
   - Execution optimization

5. **Monitoring Layer**
   - Performance reporting
   - System monitoring
   - Alerting

## Core Component Dependencies

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Data Pipeline  │────>│ Feature Engine  │────>│  ML Pipeline   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │                       │
         │                       │                       │
         │                       ▼                       ▼
         │               ┌─────────────────┐     ┌─────────────────┐
         └───────────────>│ Signal Generator│<────│  Regime Detector│
                         └─────────────────┘     └─────────────────┘
                                  │
                                  │
                                  ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Risk Management │<────│ Trading Engine  │────>│  Reporting     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Detailed Dependencies

### Data Pipeline
- **Depends on**: None (lowest level)
- **Depended on by**: Feature Engine, Signal Generator, ML Pipeline

### Feature Engine
- **Depends on**: Data Pipeline
- **Depended on by**: ML Pipeline, Signal Generator

### ML Pipeline
- **Depends on**: Data Pipeline, Feature Engine
- **Depended on by**: Signal Generator

### Regime Detector
- **Depends on**: Data Pipeline, Feature Engine
- **Depended on by**: Signal Generator, ML Pipeline

### Signal Generator
- **Depends on**: Data Pipeline, Feature Engine, ML Pipeline, Regime Detector
- **Depended on by**: Trading Engine

### Risk Management
- **Depends on**: Signal Generator
- **Depended on by**: Trading Engine

### Trading Engine
- **Depends on**: Signal Generator, Risk Management
- **Depended on by**: Reporting

### Reporting
- **Depends on**: Trading Engine
- **Depended on by**: None (highest level)

## Avoiding Circular Dependencies

To prevent circular dependencies, follow these guidelines:

1. **Dependency Injection**:
   - Use dependency injection to pass references to required components
   - Initialize dependencies in a central location (e.g., main.py)

2. **Interfaces**:
   - Define clear interfaces for component interactions
   - Components should depend on interfaces, not implementations

3. **Event-Based Communication**:
   - Use event mechanisms for upward communication in the dependency hierarchy
   - Higher-level components subscribe to events from lower-level components

4. **Refactor When Necessary**:
   - If circular dependencies are detected, refactor to extract common functionality
   - Create intermediate components that both dependent components can use

## Specific Circular Dependency Risks

1. **ML Pipeline ↔ Signal Generator**
   - Risk: Signal generator needs ML predictions, but ML pipeline might need signal history
   - Solution: ML pipeline should not directly depend on signal generator; use data layer as intermediary

2. **Risk Management ↔ Trading Engine**
   - Risk: Trading engine executes trades but risk management controls trade execution
   - Solution: Use a clear interface for risk management with callbacks/events

3. **Regime Detector ↔ ML Pipeline**
   - Risk: ML pipeline might need regime information, but regime detector uses ML models
   - Solution: Separate ML models for regime detection from trading signal models

## Dependency Management Tools

The codebase uses the following techniques to manage dependencies:

1. **Callback Registration**:
   - Components register callbacks with lower-level components
   - Example: Trading engine registers callbacks with data connectors

2. **Configuration Objects**:
   - Centralized configuration objects passed to components
   - Prevents hard-coding dependencies between components

3. **Factory Pattern**:
   - Factory classes create instances of components
   - Centralizes dependency creation and injection

4. **Observable Pattern**:
   - Components emit events that others can subscribe to
   - Decouples direct dependencies between components 