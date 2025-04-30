# Multi-Agent Coordination Guidelines

This document defines how different specialized agents should coordinate their efforts to ensure efficient collaboration and avoid duplication or conflict.

## Agent Roles and Responsibilities

### Implementation Agent
- **Primary Role**: Core functionality implementation and documentation
- **Key Areas**: Cointegration testing, pair selection, z-score strategy, Kalman filter, spread analytics, signal generation, and documentation
- **Deliverables**: Working code with core functionality and comprehensive documentation

### Testing Agent
- **Primary Role**: Testing, validation, and execution strategy implementation
- **Key Areas**: Test data, test suites, validation frameworks, transaction costs, execution algorithms
- **Deliverables**: Comprehensive tests, validation tools, and execution components

### Environment and DevOps Agent
- **Primary Role**: Development environment optimization and automation
- **Key Areas**: Docker configuration, testing automation, distributed processing
- **Deliverables**: Optimized environment and developer tools

### Project Manager Agent
- **Primary Role**: Project oversight and coordination
- **Key Areas**: Phase validation, milestone tracking, priority adjustment
- **Deliverables**: Status updates, directive enforcement, phase transitions

## Coordination Workflows

### Implementation Workflow

1. **Implementation Agent** implements core components including statistical methods
2. **Testing Agent** creates tests and validation for these components
3. **Environment and DevOps Agent** ensures efficient execution
4. **PM Agent** verifies the implementation meets requirements

### Documentation Workflow

1. **Implementation Agent** creates documentation and examples
2. **Testing Agent** adds testing and validation guidance
3. **Environment and DevOps Agent** adds deployment and environment details
4. **PM Agent** reviews and ensures documentation completeness

### Testing Workflow

1. **Testing Agent** defines test requirements and frameworks
2. **Implementation Agent** ensures code is testable and provides test cases
3. **Environment and DevOps Agent** sets up automated test execution
4. **PM Agent** verifies test coverage meets requirements

## Cross-Agent Dependencies

### Implementation Agent Dependencies
- Requires **Testing Agent** to create test data and validation
- Uses **Environment and DevOps Agent's** optimized environment

### Testing Agent Dependencies
- Requires **Implementation Agent** to implement testable code
- Uses **Environment and DevOps Agent's** test automation tools

### Environment and DevOps Agent Dependencies
- Coordinates with **Implementation Agent** on task system requirements
- Works with **Testing Agent** on test automation needs

### PM Agent Dependencies
- Receives status updates from all agents
- Enforces phase completion criteria across agents
- Adjusts priorities based on blocking issues

## Conflict Resolution

When conflicts arise between agent implementations or approaches:

1. **Technical Merit**: Favor approaches with stronger technical foundation
2. **Alignment with Design**: Prefer solutions that align with PAIRS_DESIGN.md
3. **Phase Priorities**: Solutions that support current phase completion take precedence
4. **Simplicity**: Prefer simpler solutions that achieve the same result
5. **PM Decision**: In case of unresolved conflicts, the PM Agent makes the final decision

## Communication Protocol

Agents should communicate in these ways:

1. **Status Updates**: Regular updates in agent_status.md files
2. **Implementation Notes**: Shared details in implementation_notes.md
3. **Dependency Documentation**: Clear documentation of dependencies
4. **Blocking Issues**: Immediate notification of blocking issues

## Resource Sharing

All agents should:

1. Update shared documentation for completed work
2. Respect file ownership as defined in task documents
3. Note deprecated approaches that should be avoided
4. Document reusable components for other agents
5. Share performance insights and bottlenecks

## Phase Transition Coordination

During phase transitions:

1. **Implementation Agent** completes all implementation tasks and documentation for the phase
2. **Testing Agent** verifies all tests pass for phase components
3. **Environment and DevOps Agent** validates environment compatibility
4. **PM Agent** reviews and authorizes phase transition 