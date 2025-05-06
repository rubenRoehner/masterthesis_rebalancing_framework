import numpy as np

def main():
    # global parameters
    NUM_COMMUNITIES = 8
    NUM_EPISODES = 1000
    MAX_STEPS_PER_EPISODE = 100

    # ALLOCATOR_AGENT parameters
    ALLOCATOR_AGENT_ACTION_VALUES = [-15, -10, -5, 0, 5, 10, 15]
    ALLOCATOR_AGENT_FEATURES_PER_COMMUNITY = 3 # forecast for pickup, forecast for dropoff, and current vehicle counts
    ALLOCATOR_AGENT_LR = 0.001

    # REBALANCER_AGENT parameters
    REBALANCER_AGENT_FEATURES_PER_ZONE = 3 # forecast for pickup, forecast for dropoff, and current vehicle counts

if __name__ == "__main__":
    main()