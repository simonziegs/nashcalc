# Fighting Game Nash Equilibrium Calculator

A graphical user interface (GUI) tool built with Python to calculate Nash Equilibria and related game theory strategies for fighting games. This tool is designed to help players analyze move interactions between an attacker and defender, computing mixed-strategy Nash Equilibria, best responses, and subtle exploits based on payoff matrices.

## Features
- Mixed-Strategy Nash Equilibrium: Compute optimal mixed strategies for both attacker and defender.

- Best Response Calculation: Determine the best response strategy for one player given locked probabilities for the opponent.

- Subtle Exploit Analysis: Adjust strategies to subtly exploit an opponent’s fixed strategy while staying close to the Nash Equilibrium.

- Dynamic Payoff Matrix: Add, delete, and edit moves dynamically with a customizable payoff matrix.

- Save/Load Scenarios: Export and import game scenarios as JSON files.

- Option to set locked strategies according to QRE (Probability of chosen moves based on their payoff, introduces some randomness based on Lambda paramter)

- Binary Conversion: Convert payoff values to a ternary system (-1, 0, 1) based on a threshold (±1000).

- Simplify attack/defense strategies while keeping a desired percentage of EV.

- Interactive GUI: Built with tkinter for an intuitive user experience.

## Prerequisites
- Python 3.6+

- Required Libraries:

  - numpy

  - scipy

## Installation
***1. Clone or Download the Repository:***
```
git clone https://github.com/simonziegs/nashcalc.git
```
***2. Install Dependencies:***
```
python -m pip install -r requirements.txt
```
***3. Run the Application:***
```
python main.py
```
## Usage
Set Moves:
- Add/remove moves.

- Edit move names in the "Move Names" section.

Input Payoff Matrix:
- Enter numerical payoffs in the matrix, representing the attacker’s utility for each move pair.

- Positive values favor the attacker, negative values favor the defender.

Lock Probabilities (Optional):
- Check the "Lock" box and enter a probability (0-100%) to fix it for best response or subtle exploit calculations.

- Alternatively use QRE to set it automatically (Lambda values from 0.1 to 20, the bigger the closer to nash, the lower the more random)

Calculate Strategies:
- Calculate Nash: Computes the mixed-strategy Nash Equilibrium.

- Calculate Best Response: Optimizes one player’s strategy given locked probabilities for the other.

- Calculate Subtle Exploit: Adjusts one player’s Nash strategy to exploit a fully locked opponent strategy.

View Results:
- Results appear in the right panel, showing strategy frequencies and expected payoff (EV).

Manage Scenarios:
- Save Scenario: Save the current setup to a JSON file.

- Load Scenario: Load a previously saved scenario.

- Make Binary: Convert payoffs to -1, 0, or 1 based on a ±1000 threshold.

