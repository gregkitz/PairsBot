# System Architecture Diagram

This document provides a visual representation of the system architecture, showing how different components interact with each other.

## High-Level System Architecture

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                      Trading System Architecture                                   │
└───────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                  │
                 ┌──────────────────────┐         │         ┌──────────────────────┐
                 │                      │         │         │                      │
                 │   User Interface     │◄────────┼────────►│      API Layer       │
                 │                      │         │         │                      │
                 └──────────────────────┘         │         └──────────────────────┘
                                                  │                    │
                                                  ▼                    │
┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     Core Trading System                                            │
│                                                                                                   │
│  ┌─────────────────┐   ┌─────────────────┐   ┌──────────────────┐   ┌────────────────────┐       │
│  │ Asset Classes   │   │  Pair Trading   │   │ Signal Generation │   │ Paper/Live Trading │       │
│  │ ┌─────────────┐ │   │ ┌─────────────┐ │   │ ┌──────────────┐ │   │ ┌────────────────┐ │       │
│  │ │ Futures     │ │   │ │ Pair        │ │   │ │ Signal       │ │   │ │ Paper Trader   │ │       │
│  │ │             │─┼───┼─►Selection    │ │   │ │ Processor    │ │   │ │                │ │       │
│  │ └─────────────┘ │   │ │             │─┼───┼─►              │─┼───┼─►                │ │       │
│  │ ┌─────────────┐ │   │ └─────────────┘ │   │ └──────────────┘ │   │ └────────────────┘ │       │
│  │ │ Equities    │ │   │ ┌─────────────┐ │   │ ┌──────────────┐ │   │ ┌────────────────┐ │       │
│  │ │             │─┼───┼─►Spread       │ │   │ │ Z-Score      │ │   │ │ Live Trader    │ │       │
│  │ └─────────────┘ │   │ │Analytics    │─┼───┼─►Calculation   │─┼───┼─►                │ │       │
│  │ ┌─────────────┐ │   │ └─────────────┘ │   │ └──────────────┘ │   │ └────────────────┘ │       │
│  │ │ Fixed Income │ │   │ ┌─────────────┐ │   │ ┌──────────────┐ │   │                    │       │
│  │ │             │─┼───┼─►Cointegration │ │   │ │ Confirmation │ │   │                    │       │
│  │ └─────────────┘ │   │ │Testing      │─┼───┼─►Filters       │ │   │                    │       │
│  │                 │   │ └─────────────┘ │   │ └──────────────┘ │   │                    │       │
│  └─────────────────┘   └─────────────────┘   └──────────────────┘   └────────────────────┘       │
│                                                                                                   │
│  ┌─────────────────┐   ┌─────────────────┐   ┌──────────────────┐   ┌────────────────────┐       │
│  │ ML Enhancements │   │ Backtesting     │   │ Risk Management  │   │ Data Processing    │       │
│  │ ┌─────────────┐ │   │ ┌─────────────┐ │   │ ┌──────────────┐ │   │ ┌────────────────┐ │       │
│  │ │ Feature     │ │   │ │ Backtest    │ │   │ │ Position     │ │   │ │ Data Pipeline  │ │       │
│  │ │ Engineering │ │   │ │ Engine      │ │   │ │ Sizing       │ │   │ │                │ │       │
│  │ └─────────────┘ │   │ └─────────────┘ │   │ └──────────────┘ │   │ └────────────────┘ │       │
│  │ ┌─────────────┐ │   │ ┌─────────────┐ │   │ ┌──────────────┐ │   │ ┌────────────────┐ │       │
│  │ │ Intraday    │ │   │ │ Strategy    │ │   │ │ Stop Loss   │ │   │ │ Data Cleaner   │ │       │
│  │ │ Signals     │ │   │ │ Optimization │ │   │ │ Management  │ │   │ │                │ │       │
│  │ └─────────────┘ │   │ └─────────────┘ │   │ └──────────────┘ │   │ └────────────────┘ │       │
│  │ ┌─────────────┐ │   │ ┌─────────────┐ │   │ ┌──────────────┐ │   │ ┌────────────────┐ │       │
│  │ │ Regime      │ │   │ │ Performance │ │   │ │ Exposure     │ │   │ │ Normalizer     │ │       │
│  │ │ Detection   │ │   │ │ Metrics     │ │   │ │ Management   │ │   │ │                │ │       │
│  │ └─────────────┘ │   │ └─────────────┘ │   │ └──────────────┘ │   │ └────────────────┘ │       │
│  └─────────────────┘   └─────────────────┘   └──────────────────┘   └────────────────────┘       │
│                                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                  │
                                                  ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    Infrastructure Components                                       │
│                                                                                                   │
│  ┌─────────────────┐   ┌─────────────────┐   ┌──────────────────┐   ┌────────────────────┐       │
│  │ Connectors      │   │ Monitoring      │   │ Reporting        │   │ Task Orchestration │       │
│  │ ┌─────────────┐ │   │ ┌─────────────┐ │   │ ┌──────────────┐ │   │ ┌────────────────┐ │       │
│  │ │ IB Connector│ │   │ │ Dashboard   │ │   │ │ Metrics      │ │   │ │ Celery Tasks   │ │       │
│  │ │             │ │   │ │             │ │   │ │              │ │   │ │                │ │       │
│  │ └─────────────┘ │   │ └─────────────┘ │   │ └──────────────┘ │   │ └────────────────┘ │       │
│  │ ┌─────────────┐ │   │ ┌─────────────┐ │   │ ┌──────────────┐ │   │ ┌────────────────┐ │       │
│  │ │ Data Vendor │ │   │ │ Alerting    │ │   │ │ Performance  │ │   │ │ Docker         │ │       │
│  │ │ Connectors  │ │   │ │ System      │ │   │ │ Reports      │ │   │ │ Containers     │ │       │
│  │ └─────────────┘ │   │ └─────────────┘ │   │ └──────────────┘ │   │ └────────────────┘ │       │
│  └─────────────────┘   └─────────────────┘   └──────────────────┘   └────────────────────┘       │
│                                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Component Interactions

### Data Flow Between Components

```
┌───────────────┐     ┌─────────────┐     ┌───────────────┐     ┌─────────────┐
│ Data Sources  │────►│ Asset Class │────►│ Pair Selection│────►│   Signal    │
│ (IB, Vendors) │     │ Components  │     │ & Analytics   │     │ Generation  │
└───────────────┘     └─────────────┘     └───────────────┘     └──────┬──────┘
                                                                        │
       ┌────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐     ┌───────────────┐     ┌─────────────┐
│     ML      │────►│    Risk     │────►│  Paper/Live   │────►│ Performance │
│ Enhancements│     │ Management  │     │    Trading    │     │ Monitoring  │
└─────────────┘     └─────────────┘     └───────────────┘     └─────────────┘
```

### Technical Debt Priorities

The system has several areas of technical debt that need to be addressed, with the following priority order:

```
┌───────────────────────────────────────────────────────────────────────────┐
│                     Technical Debt Priority Order                          │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────┐     ┌───────────────────────┐     ┌───────────────────────┐
│ 1. Large Files        │     │ 2. Complex Functions  │     │ 3. Duplicate Code     │
│                       │     │                       │     │                       │
│ - intraday_ml_paper_  │     │ - enhance_signals     │     │ - Asset Class         │
│   trader.py           │     │ - apply_intraday_     │     │   get_data methods    │
│ - intraday_signals.py │     │   adaptations         │     │ - error handling      │
│ - paper_trader.py     │     │ - run_backtest        │     │ - connector callbacks │
└───────────────────────┘     └───────────────────────┘     └───────────────────────┘
```

## Component Responsibilities

### Asset Classes
- **Responsibilities**: Data retrieval, metadata management, price information
- **Key Components**: Futures, Equities, Fixed Income, Cryptocurrencies

### Pair Trading
- **Responsibilities**: Pair selection, spread calculation, cointegration testing
- **Key Components**: Pair Selection, Spread Analytics, Cointegration Testing

### Signal Generation
- **Responsibilities**: Trading signal creation, signal filtering, timing optimization
- **Key Components**: Signal Processor, Z-Score Calculation, Confirmation Filters

### ML Enhancements
- **Responsibilities**: Machine learning model training, feature engineering, signal enhancement
- **Key Components**: Feature Engineering, Intraday Signals, Regime Detection

### Paper/Live Trading
- **Responsibilities**: Order execution, position management, performance tracking
- **Key Components**: Paper Trader, Live Trader, Portfolio Manager

### Risk Management
- **Responsibilities**: Position sizing, stop loss management, exposure control
- **Key Components**: Position Sizer, Stop Loss Manager, Exposure Manager

### Backtesting
- **Responsibilities**: Historical performance testing, strategy optimization
- **Key Components**: Backtest Engine, Strategy Optimization, Performance Metrics

### Infrastructure
- **Responsibilities**: System orchestration, monitoring, data processing
- **Key Components**: Docker Containers, Celery Tasks, Monitoring Dashboard
``` 