# E-scooter Rebalancing Framework

A comprehensive reinforcement learning framework for e-scooter fleet management using hierarchical multi-agent coordination.

## Overview

This framework implements a sophisticated approach to e-scooter rebalancing that combines:

- **Regional Distribution Coordinator (RDC)**: A multi-head DQN agent that coordinates vehicle distribution across multiple communities
- **User Incentive Coordinator (UIC)**: PPO-based agents that influence user behavior through incentives within individual communities
- **Demand Forecasting**: Neural network-based demand prediction using IrConv-LSTM architecture

## Architecture

### Components

1. **Regional Distribution Coordinator (`regional_distribution_coordinator/`)**
   - Multi-head Deep Q-Network with prioritized experience replay
   - Coordinates vehicle rebalancing across communities
   - Uses Double DQN with soft target network updates

2. **User Incentive Coordinator (`user_incentive_coordinator/`)**
   - PPO-based agents for individual communities
   - Learns to provide optimal user incentives
   - Influences user dropoff behavior through monetary incentives

3. **Demand Forecasting (`demand_forecasting/`)**
   - IrConv-LSTM neural network for spatio-temporal demand prediction
   - Historical data-based forecaster
   - Supports zone-level and community-level predictions

4. **Demand Provider (`demand_provider/`)**
   - Historical demand data access and management
   - Supports temporal queries and random episode initialization

5. **Hyperparameter Optimization (`hyperparameter_optimization/`)**
   - Optuna-based optimization for both RDC and UIC
   - Automated hyperparameter tuning with parallel execution

6. **Evaluation (`evaluation/`)**
   - Comprehensive evaluation framework for the complete HRL system
   - Performance metrics and analysis tools

## Installation


## Usage


## Citation

If you use this framework in your research, please cite:

```bibtex
[Add your citation information here]
```

## Disclaimer

**Important Note on AI Assistance**: 

This codebase represents original research and implementation work. **Artificial Intelligence tools were used solely for generating code comments and docstrings to improve code documentation and readability.** 

All core algorithms, system architecture, research ideas, implementation logic, and technical solutions are the original intellectual work of the author. The AI assistance was limited exclusively to:

- Writing function and class docstrings
- Adding inline code comments
- Improving code documentation formatting

No AI was used for:
- Algorithm design or implementation
- Research methodology or approach
- System architecture decisions
- Technical problem-solving
- Core functionality development
- Scientific insights or innovations

The research contributions, technical innovations, and all substantive code development remain entirely the work of the human author.
