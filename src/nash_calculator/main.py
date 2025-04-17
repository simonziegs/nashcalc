import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import os
import numpy as np
from dotenv import load_dotenv
from .game_theory import GameTheory

# Load variables from .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../.env'))

class NashCalculatorGUI:
    def __init__(self, root):
        self.root = root
        self.personal_path = os.getenv("FILE_PATH", "./scenarios")
        self.root.title("Fighting Game Nash Equilibrium Calculator")
        self.root.geometry("900x600")
        
        self.style = ttk.Style()
        self.style.configure("TLabel", font=("Arial", 10))
        self.style.configure("TButton", font=("Arial", 10))
        
        self.input_frame = ttk.Frame(root, padding="10")
        self.input_frame.grid(row=0, column=0, sticky="nsew")
        
        self.result_frame = ttk.Frame(root, padding="10")
        self.result_frame.grid(row=0, column=1, sticky="nsew")
        
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        self.create_input_widgets()
        self.create_result_widgets()
        
    def create_input_widgets(self):
        ttk.Label(self.input_frame, text="Number of Attacker Moves:").grid(row=0, column=0, padx=5, pady=5)
        self.attacker_count = tk.Spinbox(self.input_frame, from_=2, to=15, width=5, command=self.update_inputs)
        self.attacker_count.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(self.input_frame, text="Number of Defender Moves:").grid(row=1, column=0, padx=5, pady=5)
        self.defender_count = tk.Spinbox(self.input_frame, from_=2, to=15, width=5, command=self.update_inputs)
        self.defender_count.grid(row=1, column=1, padx=5, pady=5)
        
        self.moves_frame = ttk.LabelFrame(self.input_frame, text="Move Names", padding="5")
        self.moves_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky="ew")
        
        self.matrix_frame = ttk.LabelFrame(self.input_frame, text="Payoff Matrix", padding="5")
        self.matrix_frame.grid(row=3, column=0, columnspan=2, pady=10, sticky="nsew")

        # Threshold spinbox
        ttk.Label(self.input_frame, text="Simplification Threshold (%):").grid(row=5, column=0, padx=5, pady=5)
        self.threshold_spinbox = tk.Spinbox(self.input_frame, from_=0, to=20, increment=1, width=5)
        self.threshold_spinbox.grid(row=5, column=1, padx=5, pady=5)
        self.threshold_spinbox.delete(0, tk.END)
        self.threshold_spinbox.insert(0, "10")  # Default to 10%

        # Exploit spinbox
        ttk.Label(self.input_frame, text="Exploit Weight:").grid(row=6, column=0, padx=5, pady=5)
        self.exploit_spinbox = tk.Spinbox(self.input_frame, from_=0, to=1, increment=0.1, width=5)
        self.exploit_spinbox.grid(row=6, column=1, padx=5, pady=5)
        self.exploit_spinbox.delete(0, tk.END)
        self.exploit_spinbox.insert(0, "0.5")  # Default to 10%
            
        button_frame = ttk.Frame(self.input_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=10)
        ttk.Button(button_frame, text="Calculate Nash", command=self.calculate).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Calculate Best Response", command=self.calculate_best_response).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Calculate Subtle Exploit", command=self.calculate_subtle_exploit).grid(row=0, column=2, padx=5)
        ttk.Button(button_frame, text="Save Scenario", command=self.save_scenario).grid(row=0, column=3, padx=5)
        ttk.Button(button_frame, text="Load Scenario", command=self.load_scenario).grid(row=0, column=4, padx=5)
        ttk.Button(button_frame, text="Make Binary", command=self.make_binary).grid(row=0, column=5, padx=5)
        ttk.Button(button_frame, text="Simplify Attacker", command=self.simplify_attacker_strategy).grid(row=0, column=6, padx=5)
        ttk.Button(button_frame, text="Simplify Defender", command=self.simplify_defender_strategy).grid(row=0, column=7, padx=5)
        ttk.Button(button_frame, text="Compare EVs", command=self.compare_existing_scenarios).grid(row=0, column=8, padx=5)
        
        self.attacker_entries = []
        self.defender_entries = []
        self.payoff_entries = []
        self.attacker_delete_buttons = []
        self.defender_delete_buttons = []
        self.attacker_locks = []
        self.defender_locks = []
        self.attacker_prob_entries = []
        self.defender_prob_entries = []
        self.update_inputs()
        
    def create_result_widgets(self):
        ttk.Label(self.result_frame, text="Results", font=("Arial", 12, "bold")).grid(row=0, column=0, pady=5)
        self.result_text = tk.Text(self.result_frame, height=20, width=50, font=("Arial", 10), bg="#f0f0f0", relief="flat")
        self.result_text.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        
        self.result_text.tag_configure("title", font=("Arial", 12, "bold"), foreground="#2c3e50", justify="center")
        self.result_text.tag_configure("header", font=("Arial", 11, "bold"), foreground="#34495e")
        self.result_text.tag_configure("item", font=("Arial", 10), foreground="#333333")
        self.result_text.tag_configure("value_active", font=("Arial", 10, "bold"), foreground="#2980b9")
        self.result_text.tag_configure("value_inactive", font=("Arial", 10), foreground="#7f8c8d")
        
        self.result_frame.columnconfigure(0, weight=1)
        self.result_frame.rowconfigure(1, weight=1)

    def compare_existing_scenarios(self):
        """Compare expected values of saved scenarios."""
        scenario_dir = os.path.join(os.path.dirname(__file__), '../../scenarios')
        if not os.path.exists(scenario_dir):
            scenario_dir = self.personal_path
            if not scenario_dir or not os.path.exists(scenario_dir):
                messagebox.showerror("Error", "Scenario directory not found")
                return
        scenario_files = []
        for root, _, files in os.walk(scenario_dir):
            for file in files:
                if file.endswith('.json'):
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, scenario_dir)
                    scenario_files.append((relative_path, full_path))
        if not scenario_files:
            messagebox.showinfo("Info", f"No saved scenarios found in '{scenario_dir}'")
            return
        compare_window = tk.Toplevel(self.root)
        compare_window.title("Compare Scenario EVs")
        tree = ttk.Treeview(compare_window, columns=("Name", "EV"), show="headings")
        tree.heading("Name", text="Scenario Name")
        tree.heading("EV", text="Expected Value")
        scenario_data = []
        for rel_path, full_path in scenario_files:
            try:
                with open(full_path, 'r') as f:
                    scenario = json.load(f)
                attacker_moves = scenario["attacker_moves"]
                defender_moves = scenario["defender_moves"]
                payoffs = [float(p) for p in scenario["payoffs"]]
                payoff_matrix = np.array(payoffs).reshape(scenario["n_attacker"], scenario["n_defender"])
                game = GameTheory(attacker_moves, defender_moves, payoff_matrix)
                result = game.calculate_mixed_nash()
                ev = result[4]
                scenario_data.append((rel_path[:-5], ev, full_path))
            except (ValueError, json.JSONDecodeError, KeyError) as e:
                scenario_data.append((rel_path[:-5], float('-inf'), full_path))
        scenario_data.sort(key=lambda x: x[1], reverse=True)
        for name, ev, _ in scenario_data:
            ev_display = f"{ev:.4f}" if ev != float('-inf') else "Error"
            tree.insert("", "end", values=(name, ev_display))
        tree.pack(fill="both", expand=True)
        button_frame = ttk.Frame(compare_window)
        button_frame.pack(pady=10)
        def load_selected():
            selected = tree.selection()
            if selected:
                rel_name = tree.item(selected[0])["values"][0]
                full_path = next(item[2] for item in scenario_data if item[0] == rel_name)
                self.load_scenario_from_file(full_path)
                compare_window.destroy()
        ttk.Button(button_frame, text="Load Selected", command=load_selected).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Close", command=compare_window.destroy).grid(row=0, column=1, padx=5)

    def load_scenario_from_file(self, file_path):
        try:
            if file_path and os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    scenario = json.load(f)
                
                # Update move counts
                self.attacker_count.delete(0, tk.END)
                self.attacker_count.insert(0, scenario["n_attacker"])
                self.defender_count.delete(0, tk.END)
                self.defender_count.insert(0, scenario["n_defender"])
                
                # Recreate input fields (this will clear existing entries)
                self.update_inputs()
                
                # Populate move names and payoffs
                for i, move in enumerate(scenario["attacker_moves"]):
                    self.attacker_entries[i].delete(0, tk.END)
                    self.attacker_entries[i].insert(0, move)
                for j, move in enumerate(scenario["defender_moves"]):
                    self.defender_entries[j].delete(0, tk.END)
                    self.defender_entries[j].insert(0, move)
                for k, payoff in enumerate(scenario["payoffs"]):
                    self.payoff_entries[k].delete(0, tk.END)
                    self.payoff_entries[k].insert(0, str(payoff))
                
                messagebox.showinfo("Success", "Scenario loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load scenario: {str(e)}")

    def display_simplified_result(self, result, player, original_ev):
        """Display the simplified strategy result with sorted strategies and moves reduced."""
        attacker_moves, defender_moves, attacker_probs, defender_probs, simplified_ev = result
        self.result_text.delete(1.0, tk.END)
        
        # Title based on the player being simplified
        title = f"Simplified {'Attacker' if player == 'attacker' else 'Defender'} Strategy"
        self.result_text.insert(tk.END, f"{title}\n\n", "title")
        
        # Sort attacker strategies by probability (descending)
        attacker_data = sorted(zip(attacker_moves, attacker_probs), key=lambda x: x[1], reverse=True)
        sorted_attacker_moves, sorted_attacker_probs = zip(*attacker_data)
        
        self.result_text.insert(tk.END, "Attacker Strategies (Sorted by Frequency):\n", "header")
        max_move_len = max(len(move) for move in sorted_attacker_moves)
        for move, prob in zip(sorted_attacker_moves, sorted_attacker_probs):
            formatted_move = f"{move:<{max_move_len}}"
            tag = "value_active" if prob > 0 else "value_inactive"
            self.result_text.insert(tk.END, f"  {formatted_move}: ", "item")
            self.result_text.insert(tk.END, f"{100*prob:6.2f}%\n", tag)
        self.result_text.insert(tk.END, "\n")
        
        # Sort defender strategies by probability (descending)
        defender_data = sorted(zip(defender_moves, defender_probs), key=lambda x: x[1], reverse=True)
        sorted_defender_moves, sorted_defender_probs = zip(*defender_data)
        
        self.result_text.insert(tk.END, "Defender Strategies (Sorted by Frequency):\n", "header")
        max_move_len = max(len(move) for move in sorted_defender_moves)
        for move, prob in zip(sorted_defender_moves, sorted_defender_probs):
            formatted_move = f"{move:<{max_move_len}}"
            tag = "value_active" if prob > 0 else "value_inactive"
            self.result_text.insert(tk.END, f"  {formatted_move}: ", "item")
            self.result_text.insert(tk.END, f"{100*prob:6.2f}%\n", tag)
        self.result_text.insert(tk.END, "\n")
        
        # EV information
        self.result_text.insert(tk.END, "Expected Payoff (Attacker’s Perspective):\n", "header")
        self.result_text.insert(tk.END, f"  Original EV: {original_ev:6.4f}\n", "item")
        self.result_text.insert(tk.END, f"  Simplified EV: {simplified_ev:6.4f}\n", "value_active")
        percent_retained = (simplified_ev / original_ev) * 100
        self.result_text.insert(tk.END, f"  EV Retained: {percent_retained:.2f}%\n", "item")
        
        # Moves reduced
        original_n_attacker = len(self.attacker_entries)
        original_n_defender = len(self.defender_entries)
        if player == "attacker":
            simplified_n = len(attacker_moves)
            self.result_text.insert(tk.END, f"\nAttacker Moves Reduced: {original_n_attacker} → {simplified_n}\n", "item")
        elif player == "defender":
            simplified_n = len(defender_moves)
            self.result_text.insert(tk.END, f"\nDefender Moves Reduced: {original_n_defender} → {simplified_n}\n", "item")
        
    def update_inputs(self):
        old_attacker_moves = [entry.get() for entry in self.attacker_entries] if self.attacker_entries else []
        old_defender_moves = [entry.get() for entry in self.defender_entries] if self.defender_entries else []
        old_payoffs = [entry.get() for entry in self.payoff_entries] if self.payoff_entries else []
        old_n_attacker = len(old_attacker_moves)
        old_n_defender = len(old_defender_moves)
        
        for widget in self.moves_frame.winfo_children():
            widget.destroy()
        for widget in self.matrix_frame.winfo_children():
            widget.destroy()
            
        self.attacker_entries = []
        self.defender_entries = []
        self.payoff_entries = []
        self.attacker_delete_buttons = []
        self.defender_delete_buttons = []
        self.attacker_locks = []
        self.defender_locks = []
        self.attacker_prob_entries = []
        self.defender_prob_entries = []
        
        n_attacker = int(self.attacker_count.get())
        n_defender = int(self.defender_count.get())
        
        ttk.Label(self.moves_frame, text="Attacker Moves:").grid(row=0, column=0, padx=5, pady=2)
        ttk.Label(self.moves_frame, text="Lock?").grid(row=0, column=2, padx=5, pady=2)
        ttk.Label(self.moves_frame, text="Prob (%)").grid(row=0, column=3, padx=5, pady=2)
        for i in range(n_attacker):
            entry = ttk.Entry(self.moves_frame, width=15)
            entry.grid(row=i+1, column=0, padx=5, pady=2)
            if i < len(old_attacker_moves):
                entry.insert(0, old_attacker_moves[i])
            else:
                entry.insert(0, f"Move {i+1}")
            self.attacker_entries.append(entry)
            
            delete_btn = ttk.Button(self.moves_frame, text="X", width=2, 
                                  command=lambda x=i: self.delete_attacker_move(x))
            delete_btn.grid(row=i+1, column=1, padx=2, pady=2)
            self.attacker_delete_buttons.append(delete_btn)
            
            lock_var = tk.BooleanVar()
            lock_check = ttk.Checkbutton(self.moves_frame, variable=lock_var)
            lock_check.grid(row=i+1, column=2, padx=2, pady=2)
            self.attacker_locks.append(lock_var)
            
            prob_entry = ttk.Entry(self.moves_frame, width=8)
            prob_entry.grid(row=i+1, column=3, padx=2, pady=2)
            prob_entry.insert(0, "0")
            self.attacker_prob_entries.append(prob_entry)
            
        ttk.Label(self.moves_frame, text="Defender Moves:").grid(row=0, column=4, padx=5, pady=2)
        ttk.Label(self.moves_frame, text="Lock?").grid(row=0, column=6, padx=5, pady=2)
        ttk.Label(self.moves_frame, text="Prob (%)").grid(row=0, column=7, padx=5, pady=2)
        for i in range(n_defender):
            entry = ttk.Entry(self.moves_frame, width=15)
            entry.grid(row=i+1, column=4, padx=5, pady=2)
            if i < len(old_defender_moves):
                entry.insert(0, old_defender_moves[i])
            else:
                entry.insert(0, f"Move {i+1}")
            self.defender_entries.append(entry)
            
            delete_btn = ttk.Button(self.moves_frame, text="X", width=2, 
                                  command=lambda x=i: self.delete_defender_move(x))
            delete_btn.grid(row=i+1, column=5, padx=2, pady=2)
            self.defender_delete_buttons.append(delete_btn)
            
            lock_var = tk.BooleanVar()
            lock_check = ttk.Checkbutton(self.moves_frame, variable=lock_var)
            lock_check.grid(row=i+1, column=6, padx=2, pady=2)
            self.defender_locks.append(lock_var)
            
            prob_entry = ttk.Entry(self.moves_frame, width=8)
            prob_entry.grid(row=i+1, column=7, padx=2, pady=2)
            prob_entry.insert(0, "0")
            self.defender_prob_entries.append(prob_entry)
            
        attacker_moves = [entry.get() for entry in self.attacker_entries]
        defender_moves = [entry.get() for entry in self.defender_entries]
        
        for j, move in enumerate(defender_moves):
            ttk.Label(self.matrix_frame, text=move, wraplength=60).grid(row=0, column=j+1, padx=2, pady=2)
            
        for i, a_move in enumerate(attacker_moves):
            ttk.Label(self.matrix_frame, text=a_move, wraplength=60).grid(row=i+1, column=0, padx=2, pady=2, sticky="e")
            for j in range(n_defender):
                entry = ttk.Entry(self.matrix_frame, width=12)
                entry.grid(row=i+1, column=j+1, padx=2, pady=2)
                if i < old_n_attacker and j < old_n_defender and old_payoffs:
                    idx = i * old_n_defender + j
                    if idx < len(old_payoffs):
                        entry.insert(0, old_payoffs[idx])
                else:
                    entry.insert(0, "0")
                self.payoff_entries.append(entry)
                
    def delete_attacker_move(self, index):
        if len(self.attacker_entries) <= 2:
            messagebox.showwarning("Warning", "Cannot delete move: Minimum of 2 attacker moves required.")
            return
            
        n_defender = int(self.defender_count.get())
        old_payoffs = [entry.get() for entry in self.payoff_entries]
        payoff_matrix = np.array(old_payoffs).reshape(-1, n_defender)
        
        del self.attacker_entries[index]
        self.attacker_count.delete(0, tk.END)
        self.attacker_count.insert(0, len(self.attacker_entries))
        
        new_payoffs = np.delete(payoff_matrix, index, axis=0).flatten().tolist()
        self.update_inputs_with_payoffs(new_payoffs)
        
    def delete_defender_move(self, index):
        if len(self.defender_entries) <= 2:
            messagebox.showwarning("Warning", "Cannot delete move: Minimum of 2 defender moves required.")
            return
            
        n_defender = int(self.defender_count.get())
        old_payoffs = [entry.get() for entry in self.payoff_entries]
        payoff_matrix = np.array(old_payoffs).reshape(-1, n_defender)
        
        del self.defender_entries[index]
        self.defender_count.delete(0, tk.END)
        self.defender_count.insert(0, len(self.defender_entries))
        
        new_payoffs = np.delete(payoff_matrix, index, axis=1).flatten().tolist()
        self.update_inputs_with_payoffs(new_payoffs)
        
    def update_inputs_with_payoffs(self, new_payoffs):
        attacker_moves = [entry.get() for entry in self.attacker_entries]
        defender_moves = [entry.get() for entry in self.defender_entries]
        
        for widget in self.moves_frame.winfo_children():
            widget.destroy()
        for widget in self.matrix_frame.winfo_children():
            widget.destroy()
            
        self.attacker_entries = []
        self.defender_entries = []
        self.payoff_entries = []
        self.attacker_delete_buttons = []
        self.defender_delete_buttons = []
        self.attacker_locks = []
        self.defender_locks = []
        self.attacker_prob_entries = []
        self.defender_prob_entries = []
        
        n_attacker = len(attacker_moves)
        n_defender = len(defender_moves)
        
        ttk.Label(self.moves_frame, text="Attacker Moves:").grid(row=0, column=0, padx=5, pady=2)
        ttk.Label(self.moves_frame, text="Lock?").grid(row=0, column=2, padx=5, pady=2)
        ttk.Label(self.moves_frame, text="Prob (%)").grid(row=0, column=3, padx=5, pady=2)
        for i, move in enumerate(attacker_moves):
            entry = ttk.Entry(self.moves_frame, width=15)
            entry.grid(row=i+1, column=0, padx=5, pady=2)
            entry.insert(0, move)
            self.attacker_entries.append(entry)
            
            delete_btn = ttk.Button(self.moves_frame, text="X", width=2, 
                                  command=lambda x=i: self.delete_attacker_move(x))
            delete_btn.grid(row=i+1, column=1, padx=2, pady=2)
            self.attacker_delete_buttons.append(delete_btn)
            
            lock_var = tk.BooleanVar()
            lock_check = ttk.Checkbutton(self.moves_frame, variable=lock_var)
            lock_check.grid(row=i+1, column=2, padx=2, pady=2)
            self.attacker_locks.append(lock_var)
            
            prob_entry = ttk.Entry(self.moves_frame, width=8)
            prob_entry.grid(row=i+1, column=3, padx=2, pady=2)
            prob_entry.insert(0, "0")
            self.attacker_prob_entries.append(prob_entry)
            
        ttk.Label(self.moves_frame, text="Defender Moves:").grid(row=0, column=4, padx=5, pady=2)
        ttk.Label(self.moves_frame, text="Lock?").grid(row=0, column=6, padx=5, pady=2)
        ttk.Label(self.moves_frame, text="Prob (%)").grid(row=0, column=7, padx=5, pady=2)
        for i, move in enumerate(defender_moves):
            entry = ttk.Entry(self.moves_frame, width=15)
            entry.grid(row=i+1, column=4, padx=5, pady=2)
            entry.insert(0, move)
            self.defender_entries.append(entry)
            
            delete_btn = ttk.Button(self.moves_frame, text="X", width=2, 
                                  command=lambda x=i: self.delete_defender_move(x))
            delete_btn.grid(row=i+1, column=5, padx=2, pady=2)
            self.defender_delete_buttons.append(delete_btn)
            
            lock_var = tk.BooleanVar()
            lock_check = ttk.Checkbutton(self.moves_frame, variable=lock_var)
            lock_check.grid(row=i+1, column=6, padx=2, pady=2)
            self.defender_locks.append(lock_var)
            
            prob_entry = ttk.Entry(self.moves_frame, width=8)
            prob_entry.grid(row=i+1, column=7, padx=2, pady=2)
            prob_entry.insert(0, "0")
            self.defender_prob_entries.append(prob_entry)
            
        for j, move in enumerate(defender_moves):
            ttk.Label(self.matrix_frame, text=move, wraplength=60).grid(row=0, column=j+1, padx=2, pady=2)
            
        for i, a_move in enumerate(attacker_moves):
            ttk.Label(self.matrix_frame, text=a_move, wraplength=60).grid(row=i+1, column=0, padx=2, pady=2, sticky="e")
            for j in range(n_defender):
                entry = ttk.Entry(self.matrix_frame, width=12)
                entry.grid(row=i+1, column=j+1, padx=2, pady=2)
                idx = i * n_defender + j
                if idx < len(new_payoffs):
                    entry.insert(0, str(new_payoffs[idx]))
                else:
                    entry.insert(0, "0")
                self.payoff_entries.append(entry)
                
    def calculate(self):
        """Calculate and display the mixed-strategy Nash equilibrium."""
        try:
            attacker_moves = [entry.get() for entry in self.attacker_entries]
            defender_moves = [entry.get() for entry in self.defender_entries]
            payoffs = [float(entry.get()) for entry in self.payoff_entries]
            payoff_matrix = np.array(payoffs).reshape(len(attacker_moves), len(defender_moves))
            game = GameTheory(attacker_moves, defender_moves, payoff_matrix)
            result = game.calculate_mixed_nash()
            self.display_result(result, title="Mixed Strategy Nash Equilibrium")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input or calculation failed: {str(e)}")
        except Exception as e:
            messagebox.showerror("Unexpected Error", f"An error occurred: {str(e)}")
    
    def display_result(self, result, title="Mixed Strategy Nash Equilibrium"):
        """Display calculation results in the text widget."""
        self.result_text.delete(1.0, tk.END)
        
        if isinstance(result, str):
            self.result_text.insert(tk.END, result)
            return
        attacker_moves, defender_moves, attacker_probs, defender_probs, game_value = result
        
        # Optimal Strategy
        attacker_data = sorted(zip(attacker_moves, attacker_probs), key=lambda x: x[1], reverse=True)
        sorted_attacker_moves, sorted_attacker_probs = zip(*attacker_data)
        
        defender_data = sorted(zip(defender_moves, defender_probs), key=lambda x: x[1], reverse=True)
        sorted_defender_moves, sorted_defender_probs = zip(*defender_data)
        
        self.result_text.insert(tk.END, f"{title}\n\n", "title")
        
        self.result_text.insert(tk.END, "Attacker Strategies (Sorted by Frequency):\n", "header")
        max_move_len = max(len(move) for move in sorted_attacker_moves)
        for move, prob in zip(sorted_attacker_moves, sorted_attacker_probs):
            formatted_move = f"{move:<{max_move_len}}"
            self.result_text.insert(tk.END, f"  {formatted_move}: ", "item")
            tag = "value_active" if prob > 0 else "value_inactive"
            self.result_text.insert(tk.END, f"{100*prob:6.2f}%\n", tag)
        self.result_text.insert(tk.END, "\n")
        
        self.result_text.insert(tk.END, "Defender Strategies (Sorted by Frequency):\n", "header")
        max_move_len = max(len(move) for move in sorted_defender_moves)
        for move, prob in zip(sorted_defender_moves, sorted_defender_probs):
            formatted_move = f"{move:<{max_move_len}}"
            self.result_text.insert(tk.END, f"  {formatted_move}: ", "item")
            tag = "value_active" if prob > 0 else "value_inactive"
            self.result_text.insert(tk.END, f"{100*prob:6.2f}%\n", tag)
        self.result_text.insert(tk.END, "\n")
        
        self.result_text.insert(tk.END, "Expected Payoff (Attacker’s Perspective):\n", "header")
        self.result_text.insert(tk.END, f"  EV: {game_value:6.4f}\n", "value_active")
            
    def save_scenario(self):
        try:
            attacker_moves = [entry.get() for entry in self.attacker_entries]
            defender_moves = [entry.get() for entry in self.defender_entries]
            payoffs = [entry.get() for entry in self.payoff_entries]
            
            scenario = {
                "attacker_moves": attacker_moves,
                "defender_moves": defender_moves,
                "payoffs": payoffs,
                "n_attacker": len(attacker_moves),
                "n_defender": len(defender_moves)
            }
            
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Save Scenario"
            )
            
            if file_path:
                with open(file_path, 'w') as f:
                    json.dump(scenario, f, indent=4)
                messagebox.showinfo("Success", "Scenario saved successfully!")
                
        except ValueError:
            messagebox.showerror("Error", "Please enter valid data before saving")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save scenario: {str(e)}")
    
    def load_scenario(self):
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Load Scenario"
            )
            
            if file_path and os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    scenario = json.load(f)
                
                self.attacker_count.delete(0, tk.END)
                self.attacker_count.insert(0, scenario["n_attacker"])
                self.defender_count.delete(0, tk.END)
                self.defender_count.insert(0, scenario["n_defender"])
                
                self.attacker_entries = []
                self.defender_entries = []
                self.payoff_entries = []
                for move in scenario["attacker_moves"]:
                    entry = ttk.Entry(self.moves_frame)
                    entry.insert(0, move)
                    self.attacker_entries.append(entry)
                for move in scenario["defender_moves"]:
                    entry = ttk.Entry(self.moves_frame)
                    entry.insert(0, move)
                    self.defender_entries.append(entry)
                for payoff in scenario["payoffs"]:
                    entry = ttk.Entry(self.matrix_frame)
                    entry.insert(0, str(payoff))
                    self.payoff_entries.append(entry)
                
                self.update_inputs()
                
                messagebox.showinfo("Success", "Scenario loaded successfully!")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load scenario: {str(e)}")
    
    def make_binary(self):
        try:
            for entry in self.payoff_entries:
                value = entry.get()
                try:
                    num = float(value)
                    if num >= 1000:
                        new_value = 1
                    elif num <= -1000:
                        new_value = -1
                    else:
                        new_value = 0
                    entry.delete(0, tk.END)
                    entry.insert(0, str(new_value))
                except ValueError:
                    continue
            messagebox.showinfo("Success", "Payoffs converted to binary values (threshold ±1000)!")
        except Exception as e:
            messagebox.showerror("Error", f"Error converting to binary: {str(e)}")

    def calculate_best_response(self):
        """Calculate best response for one player given opponent's locked probabilities."""
        try:
            # Collect moves and payoffs
            attacker_moves = [entry.get() for entry in self.attacker_entries]
            defender_moves = [entry.get() for entry in self.defender_entries]
            payoffs = [float(entry.get()) for entry in self.payoff_entries]
            payoff_matrix = np.array(payoffs).reshape(len(attacker_moves), len(defender_moves))
            
            # Create GameTheory instance
            game = GameTheory(attacker_moves, defender_moves, payoff_matrix)
            
            # Collect locked probabilities
            attacker_locked = [var.get() for var in self.attacker_locks]
            defender_locked = [var.get() for var in self.defender_locks]
            
            # Validate and collect probabilities
            attacker_probs = []
            for i, locked in enumerate(attacker_locked):
                if locked:
                    try:
                        prob = float(self.attacker_prob_entries[i].get()) / 100
                        if not 0 <= prob <= 1:
                            raise ValueError(f"Probability for {attacker_moves[i]} must be between 0 and 100%")
                        attacker_probs.append(prob)
                    except ValueError as e:
                        if str(e).startswith("Probability"):
                            raise
                        raise ValueError(f"Invalid probability for {attacker_moves[i]}: must be a number")
                else:
                    attacker_probs.append(None)
            
            defender_probs = []
            for i, locked in enumerate(defender_locked):
                if locked:
                    try:
                        prob = float(self.defender_prob_entries[i].get()) / 100
                        if not 0 <= prob <= 1:
                            raise ValueError(f"Probability for {defender_moves[i]} must be between 0 and 100%")
                        defender_probs.append(prob)
                    except ValueError as e:
                        if str(e).startswith("Probability"):
                            raise
                        raise ValueError(f"Invalid probability for {defender_moves[i]}: must be a number")
                else:
                    defender_probs.append(None)
            
            # Validate locked probabilities sum
            attacker_locked_sum = sum(p for p in attacker_probs if p is not None)
            defender_locked_sum = sum(p for p in defender_probs if p is not None)
            if attacker_locked_sum > 1 or defender_locked_sum > 1:
                raise ValueError("Locked probabilities must not exceed 100%")
            
            # Determine which side to optimize
            if any(attacker_locked) and not any(defender_locked):
                result = game.calculate_defender_best_response(attacker_probs)
                self.display_result(result, title="Best Response (Defender Optimized)")
            elif any(defender_locked) and not any(attacker_locked):
                result = game.calculate_attacker_best_response(defender_probs)
                self.display_result(result, title="Best Response (Attacker Optimized)")
            else:
                messagebox.showwarning("Warning", "Lock strategies for exactly one side (attacker or defender)")
                return
        
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input or calculation failed: {str(e)}")
        except Exception as e:
            messagebox.showerror("Unexpected Error", f"An error occurred: {str(e)}")

    def calculate_subtle_exploit(self):
        """Calculate a subtle exploit strategy based on locked probabilities and exploit weight."""
        try:
            # Collect moves and payoffs
            attacker_moves = [entry.get() for entry in self.attacker_entries]
            defender_moves = [entry.get() for entry in self.defender_entries]
            payoffs = [float(entry.get()) for entry in self.payoff_entries]
            payoff_matrix = np.array(payoffs).reshape(len(attacker_moves), len(defender_moves))
            
            # Create GameTheory instance
            game = GameTheory(attacker_moves, defender_moves, payoff_matrix)
            
            # Collect locked probabilities
            attacker_locked = [var.get() for var in self.attacker_locks]
            defender_locked = [var.get() for var in self.defender_locks]
            
            # Validate and collect attacker probabilities
            attacker_probs = []
            for i, locked in enumerate(attacker_locked):
                if locked:
                    try:
                        prob = float(self.attacker_prob_entries[i].get()) / 100
                        if not 0 <= prob <= 1:
                            raise ValueError(f"Probability for {attacker_moves[i]} must be between 0 and 100%")
                        attacker_probs.append(prob)
                    except ValueError as e:
                        if str(e).startswith("Probability"):
                            raise
                        raise ValueError(f"Invalid probability for {attacker_moves[i]}: must be a number")
                else:
                    attacker_probs.append(None)
            
            # Validate and collect defender probabilities
            defender_probs = []
            for i, locked in enumerate(defender_locked):
                if locked:
                    try:
                        prob = float(self.defender_prob_entries[i].get()) / 100
                        if not 0 <= prob <= 1:
                            raise ValueError(f"Probability for {defender_moves[i]} must be between 0 and 100%")
                        defender_probs.append(prob)
                    except ValueError as e:
                        if str(e).startswith("Probability"):
                            raise
                        raise ValueError(f"Invalid probability for {defender_moves[i]}: must be a number")
                else:
                    defender_probs.append(None)
            
            # Validate exploit weight
            try:
                exploit_weight = float(self.exploit_spinbox.get())
                if not 0 <= exploit_weight <= 1:
                    raise ValueError("Exploit weight must be between 0 and 1")
            except ValueError:
                raise ValueError("Exploit weight must be a valid number between 0 and 1")
            
            # Call GameTheory.calculate_subtle_exploit
            result = game.calculate_subtle_exploit(attacker_probs, defender_probs, exploit_weight)
            
            # Display results
            self.display_result(result, title="Subtle Exploit")
            
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input or calculation failed: {str(e)}")
        except Exception as e:
            messagebox.showerror("Unexpected Error", f"An error occurred: {str(e)}")

    def simplify_attacker_strategy(self):
        """Simplify attacker's strategy while maintaining EV within threshold."""
        try:
            attacker_moves = [entry.get() for entry in self.attacker_entries]
            defender_moves = [entry.get() for entry in self.defender_entries]
            payoffs = [float(entry.get()) for entry in self.payoff_entries]
            payoff_matrix = np.array(payoffs).reshape(len(attacker_moves), len(defender_moves))
            game = GameTheory(attacker_moves, defender_moves, payoff_matrix)
            threshold_percent = float(self.threshold_spinbox.get())
            result = game.simplify_attacker_strategy(threshold_percent)
            nash_result = game.calculate_mixed_nash()
            original_ev = nash_result[4]
            self.display_simplified_result(result, "attacker", original_ev)
        except ValueError as e:
            messagebox.showerror("Error", f"Failed to simplify attacker strategy: {str(e)}")
        except Exception as e:
            messagebox.showerror("Unexpected Error", f"An error occurred: {str(e)}")

    def simplify_defender_strategy(self):
        """Simplify defender's strategy while maintaining EV within threshold."""
        try:
            attacker_moves = [entry.get() for entry in self.attacker_entries]
            defender_moves = [entry.get() for entry in self.defender_entries]
            payoffs = [float(entry.get()) for entry in self.payoff_entries]
            payoff_matrix = np.array(payoffs).reshape(len(attacker_moves), len(defender_moves))
            game = GameTheory(attacker_moves, defender_moves, payoff_matrix)
            threshold_percent = float(self.threshold_spinbox.get())
            result = game.simplify_defender_strategy(threshold_percent)
            nash_result = game.calculate_mixed_nash()
            original_ev = nash_result[4]
            self.display_simplified_result(result, "defender", original_ev)
        except ValueError as e:
            messagebox.showerror("Error", f"Failed to simplify defender strategy: {str(e)}")
        except Exception as e:
            messagebox.showerror("Unexpected Error", f"An error occurred: {str(e)}")
        
if __name__ == "__main__":
    root = tk.Tk()
    app = NashCalculatorGUI(root)
    root.mainloop()