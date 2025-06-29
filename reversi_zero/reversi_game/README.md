# Reversi Zero – High-Performance Reversi Engine (Python + C++)

This project implements a customizable Reversi engine that supports up to **4 players** (because blocked fileds are 5 if you change this more players are possible), **blocked fields**, and **map-based game setup**. It combines a fast **C++ backend** with a flexible **Python interface**, designed for AI agents like **MCTS** and **AlphaZero**.

---

## 🔧 Features

- Supports 2–4 player Reversi on a 15×15 board
- Optional blocked fields (`-` in `.map` files)
- High-speed C++ game logic (optional but recommended)
- Full Python wrapper and environment

---

## ⚙️ C++ Backend Build Instructions

To use the fast C++ backend, compile it using `pybind11`. The Python module will be built from the C++ source files using the provided `setup.py` file.

### ✅ Requirements

- Python ≥ 3.8
- `pybind11` (`pip install pybind11`)
- A C++17-compatible compiler (e.g., `g++`, `clang`)
- `setuptools` and `wheel`

### 🛠 Build the Extension

In the project root, run:

```bash
python setup.py build_ext --inplace