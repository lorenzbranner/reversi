# AlphaZero for Reversi (Multi-Player, Custom Maps)

This project implements **AlphaZero** to learn and play a custom variant of the **Reversi (Othello)** board game using **self-play** and **Monte Carlo Tree Search (MCTS)**. It supports:
- Arbitrary map sizes and shapes
- Multiple players (not just 2!)
- Integration with a fast C++ Reversi backend
- Parallel self-play for efficient training

> 💡 The actual Reversi game logic is implemented in C++ and exposed via Python. For details, see the [Reversi C++ Engine](reversi_game/README.md) readme.

---

## 🧠 Core Idea

This project reimplements the AlphaZero algorithm for general multi-player Reversi using:

- A custom PyTorch **ResNet** with two heads:
  - A **policy head** predicting the move distribution
  - A **value head** estimating each player's final score
- **Dirichlet noise** for exploration
- **Temperature-controlled sampling** to encourage diverse openings
- **Self-play** to generate training data with MCTS-based moves
- **Support for custom board generators** (maps, sizes, and obstacles)

---

## 🚀 Training
Train AlphaZero from scratch with:

```bash
python train.py
```

You can configure:
    Number of self-play games
    MCTS simulations
    Batch size and epochs
    Parallelism
    Checkpoints are saved in models/checkpoints/.

## Evaluate 
Evaluate AlphaZero with the evaluate script.

## 📊 Features
✅ Multi-player support (not just 2 players)
✅ Pluggable board generation
✅ Parallel self-play for speedup
✅ Full AlphaZero training loop
✅ Temperature annealing for more deterministic endgames
✅ Modular and extensible code

## 🧩 Reversi C++ Engine
All game logic (valid moves, scoring, disqualification, next-player logic) is implemented in optimized C++.
To understand the board format, blocked fields, and player handling, [see](reversi_game/README.md)

## 📈 Future Improvements
 Add GUI with PyGame or Web Interface
 Visualization of MCTS tree
 More advanced heuristics for initial value network bootstrap
 League training (train against past models)

## 🧠 Acknowledgements
DeepMind AlphaZero Paper
Leela Chess Zero for inspiration on training infrastructure
Your own sweat and GPU hours.

## 📜 License
MIT License – use freely we do not care but credit would be nice.

