# Combat Agent for RimWorld

A reinforcement learning project that trains combat agents for RimWorld using various algorithms including DQN, PPO, and PGM. The agents learn optimal combat strategies in simulated RimWorld environments.

## Prerequisites
- [Combat Agent Client](https://github.com/zivmax/rimworld-combat-agent-client) installed and activate on RimWorld
- A legal copy of RimWorld, Linux version required.
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Visual Studio Code](https://code.visualstudio.com/)
- [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) for VSCode

## Setup the files

1. Create a empty folder
```sh
mkdir combat-agent
```

2. Clone the repository into the folder
```sh
git clone https://github.com/zivmax/rimworld-combat-agent-server.git combat-agent/agent-server
```

3. Put the game into the folder and make sure the game folder just named `game`, and `game/RimWorldLinux` is the game executable.
```sh
if [ -x combat-agent/game/RimWorldLinux ]; then
  echo "Game is ready."
else
  echo "Game not found or misnamed."
fi
```

## Quick Start with Dev Container

1. Open the project in VSCode
2. When prompted, click "Reopen in Container" or:
   - Press F1
   - Select "Dev Containers: Reopen in Container"

The devcontainer will automatically:
- Build the Docker environment
- Install all Python dependencies
- Configure Git settings
- Set up RimWorld mod configurations

## Project Structure

```
.
├── agents/               # RL agent implementations
│   ├── dqn-cnn/         # DQN with CNN architecture
│   ├── dqn-naive/       # Basic DQN implementation  
│   ├── dqn-resnet/      # DQN with ResNet
│   ├── pgm/             # Policy Gradient Methods
│   ├── ppo/             # Proximal Policy Optimization
│   └── random/          # Random action baseline
├── env/                 # Environment implementation
│   ├── wrappers/        # Gym environment wrappers
│   ├── action.py        # Action space definition
│   ├── game.py          # Game interface
│   └── state.py         # State representation
└── utils/              # Utility functions
```

## Training Agents

Choose an agent type and run its training script:

```bash
# Train DQN with ResNet
python -m agents.dqn-resnet.train

# Train PPO agent
python -m agents.ppo.train

# Train baseline random agent
python -m agents.random.train
```

## Deploying Trained Agents

Run the deployment script for a trained agent:

```bash
python -m agents.dqn-resnet.deploy
```

## Development

### Key Dependencies

- PyTorch
- Gymnasium
- Pandas
- Matplotlib
- Seaborn

### Code Organization

- `agents/`: Each agent type has its own implementation with:
  - `agent.py`: Core agent class
  - `model.py`: Neural network architecture
  - `train.py`: Training loop
  - `deploy.py`: Evaluation script

### Training Artifacts

Results are saved under agent-specific directories:

- `models/`: Saved model weights
- `plots/`: Training visualizations
- `histories/`: Raw csv data of the plots

### Utility Scripts

Clean up generated files:
```bash
# Clean training plots
./utils/clean-plots.sh

# Clean history files
./utils/clean-histories.sh

# Clean log files
./utils/clean-logs.sh

# Clean tracing files
./utils/clean-tracing.sh
```

## Environment Configuration

The RimWorld environment can be configured through `EnvOptions` in training scripts:
- Map size and features
- Team sizes
- Reward structures
- Game speed and intervals

## Monitoring

Training progress can be monitored through:
- Real-time progress in console
- Saved history plots
- Episode statistics
- Model checkpoints
