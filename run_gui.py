#!/usr/bin/env python3
"""
Launch GUI for Flood Prediction System
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox

def check_dependencies():
    """Check required libraries"""
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 
        'matplotlib', 'mysql-connector-python'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Missing Libraries",
            f"The following libraries are not installed:\n{', '.join(missing_packages)}\n\n"
            f"Please run the command:\npip install {' '.join(missing_packages)}"
        )
        return False
    
    return True

def main():
    """Launch the application"""
    print("Starting Flood Prediction System...")
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Import and run GUI
    try:
        from flood_prediction_gui import main as run_gui
        run_gui()
    except ImportError as e:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Import Error",
            f"Unable to import GUI module:\n{str(e)}\n\n"
            "Please check the flood_prediction_gui.py file"
        )
    except Exception as e:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Error", f"Application startup error:\n{str(e)}")

if __name__ == "__main__":
    main()