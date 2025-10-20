import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import threading
import subprocess
import sys
import os
import json

# Import project modules with error handling
try:
    from setup_db import get_connection, close_connection, setup_database
    from predictor import (load_combined_data, load_data_from_db, train_model, 
                          predict_flood_risk, create_flood_labels, 
                          generate_advanced_training_data)
    IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Critical import error: {e}. Some features may not work.")
    IMPORT_SUCCESS = False

class FloodPredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Flood Prediction System")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Storage variables with defaults
        self.model = None
        self.features = None
        self.is_advanced = False
        self.current_data = None
        
        # UI components (init to None to avoid errors)
        self.rainfall_tree = None
        self.river_tree = None
        self.predictions_tree = None
        self.dashboard_fig = None
        self.reports_fig = None
        
        # Create interface
        self.setup_styles()
        self.create_menu()
        self.create_main_interface()
        self.create_status_bar()
        
        # Initial checks
        if IMPORT_SUCCESS:
            self.check_database_connection()
        else:
            self.update_status("Import error - some features disabled")

    def setup_styles(self):
        """Set up styles for the interface"""
        try:
            style = ttk.Style()
            style.theme_use('clam')
            
            # Main colors
            style.configure('Title.TLabel', font=('Arial', 16, 'bold'), foreground='#2c3e50')
            style.configure('Header.TLabel', font=('Arial', 12, 'bold'), foreground='#34495e')
            style.configure('Success.TLabel', foreground='#27ae60', font=('Arial', 10, 'bold'))
            style.configure('Warning.TLabel', foreground='#f39c12', font=('Arial', 10, 'bold'))
            style.configure('Error.TLabel', foreground='#e74c3c', font=('Arial', 10, 'bold'))
            style.configure('Accent.TButton', foreground='white', font=('Arial', 10, 'bold'))
        except Exception as e:
            print(f"Style setup error: {e}")

    def create_menu(self):
        """Create menu bar"""
        try:
            menubar = tk.Menu(self.root)
            self.root.config(menu=menubar)
            
            # File menu
            file_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="File", menu=file_menu)
            file_menu.add_command(label="Setup Database", command=self.setup_database)
            file_menu.add_command(label="Export Report", command=self.export_report)
            
            # Data menu
            data_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="Data", menu=data_menu)
            data_menu.add_command(label="Crawl Weather Data", command=self.crawl_weather_data)
            data_menu.add_command(label="Crawl River Data", command=self.crawl_river_data)
            data_menu.add_command(label="Manage Database", command=self.manage_database)
            
            # Model menu
            model_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="Model", menu=model_menu)
            model_menu.add_command(label="Train Model", command=self.train_prediction_model)
            model_menu.add_command(label="Evaluate Model", command=self.evaluate_model)
            
            # Help menu
            help_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="Help", menu=help_menu)
            help_menu.add_command(label="User Guide", command=self.show_help)
            help_menu.add_command(label="About Software", command=self.show_about)
        except Exception as e:
            messagebox.showerror("Error", f"Menu creation failed: {str(e)}")

    def create_main_interface(self):
        """Create main interface"""
        try:
            # Create notebook for tabs
            self.notebook = ttk.Notebook(self.root)
            self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            # Tab 1: Dashboard
            self.create_dashboard_tab()
            
            # Tab 2: Prediction
            self.create_prediction_tab()
            
            # Tab 3: Data
            self.create_data_tab()
            
            # Tab 4: Reports
            self.create_reports_tab()
            
            # Tab 5: Settings
            self.create_settings_tab()
        except Exception as e:
            messagebox.showerror("Error", f"Interface creation failed: {str(e)}")

    def create_dashboard_tab(self):
        """Dashboard Tab - System Overview"""
        try:
            dashboard_frame = ttk.Frame(self.notebook)
            self.notebook.add(dashboard_frame, text="Dashboard")
            
            # Left frame - System information
            left_frame = ttk.LabelFrame(dashboard_frame, text="System Status", padding=10)
            left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Database status
            ttk.Label(left_frame, text="Database Status:", style='Header.TLabel').pack(anchor='w')
            self.db_status_label = ttk.Label(left_frame, text="Checking...", style='Warning.TLabel')
            self.db_status_label.pack(anchor='w', pady=(0,10))
            
            # Model status
            ttk.Label(left_frame, text="Model Status:", style='Header.TLabel').pack(anchor='w')
            self.model_status_label = ttk.Label(left_frame, text="Not Trained", style='Error.TLabel')
            self.model_status_label.pack(anchor='w', pady=(0,10))
            
            # Data summary
            ttk.Label(left_frame, text="Data Statistics:", style='Header.TLabel').pack(anchor='w')
            self.data_summary_text = tk.Text(left_frame, height=8, wrap=tk.WORD)
            result_scroll = ttk.Scrollbar(left_frame, orient="vertical", command=self.data_summary_text.yview)
            self.data_summary_text.configure(yscrollcommand=result_scroll.set)
            self.data_summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            result_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Refresh button
            ttk.Button(left_frame, text="Refresh", command=self.refresh_dashboard).pack(anchor='ne', padx=10, pady=10)
            
            # Right frame - Charts
            right_frame = ttk.LabelFrame(dashboard_frame, text="Overview Charts", padding=10)
            right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Matplotlib figure
            self.dashboard_fig, self.dashboard_axes = plt.subplots(2, 2, figsize=(8, 6))
            self.dashboard_fig.suptitle("System Data Statistics")
            
            self.dashboard_canvas = FigureCanvasTkAgg(self.dashboard_fig, right_frame)
            self.dashboard_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            messagebox.showerror("Error", f"Dashboard creation failed: {str(e)}")

    def create_prediction_tab(self):
        """Prediction Tab - Perform flood prediction"""
        try:
            prediction_frame = ttk.Frame(self.notebook)
            self.notebook.add(prediction_frame, text="Prediction")
            
            # Top frame - Input parameters
            input_frame = ttk.LabelFrame(prediction_frame, text="Enter Prediction Data", padding=10)
            input_frame.pack(fill=tk.X, padx=10, pady=5)
            
            # Create 2 columns for input
            left_input = ttk.Frame(input_frame)
            left_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
            
            right_input = ttk.Frame(input_frame)
            right_input.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
            
            # Left column - Weather data
            ttk.Label(left_input, text="Weather Data:", style='Header.TLabel').pack(anchor='w')
            
            # Temperature
            ttk.Label(left_input, text="Temperature (°C):").pack(anchor='w', pady=(10,0))
            self.temp_var = tk.DoubleVar(value=26.0)
            temp_frame = ttk.Frame(left_input)
            temp_frame.pack(fill=tk.X, pady=2)
            temp_scale = tk.Scale(temp_frame, from_=15, to=40, resolution=0.1, 
                                 orient=tk.HORIZONTAL, variable=self.temp_var)
            temp_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.temp_value_label = ttk.Label(temp_frame, text="26.0°C", width=8)
            self.temp_value_label.pack(side=tk.RIGHT)
            temp_scale.config(command=lambda v: self.temp_value_label.config(text=f"{float(v):.1f}°C"))
            
            # Humidity
            ttk.Label(left_input, text="Humidity (%):").pack(anchor='w', pady=(10,0))
            self.humidity_var = tk.DoubleVar(value=70.0)
            humidity_frame = ttk.Frame(left_input)
            humidity_frame.pack(fill=tk.X, pady=2)
            humidity_scale = tk.Scale(humidity_frame, from_=20, to=100, resolution=1, 
                                     orient=tk.HORIZONTAL, variable=self.humidity_var)
            humidity_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.humidity_value_label = ttk.Label(humidity_frame, text="70%", width=8)
            self.humidity_value_label.pack(side=tk.RIGHT)
            humidity_scale.config(command=lambda v: self.humidity_value_label.config(text=f"{int(float(v))}%"))
            
            # Pressure
            ttk.Label(left_input, text="Pressure (hPa):").pack(anchor='w', pady=(10,0))
            self.pressure_var = tk.DoubleVar(value=1013.0)
            pressure_frame = ttk.Frame(left_input)
            pressure_frame.pack(fill=tk.X, pady=2)
            pressure_scale = tk.Scale(pressure_frame, from_=950, to=1050, resolution=1, 
                                     orient=tk.HORIZONTAL, variable=self.pressure_var)
            pressure_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.pressure_value_label = ttk.Label(pressure_frame, text="1013hPa", width=8)
            self.pressure_value_label.pack(side=tk.RIGHT)
            pressure_scale.config(command=lambda v: self.pressure_value_label.config(text=f"{int(float(v))}hPa"))
            
            # Rainfall 1h
            ttk.Label(left_input, text="Rainfall 1h (mm):").pack(anchor='w', pady=(10,0))
            self.rainfall_1h_var = tk.DoubleVar(value=0.0)
            rainfall_1h_frame = ttk.Frame(left_input)
            rainfall_1h_frame.pack(fill=tk.X, pady=2)
            rainfall_1h_scale = tk.Scale(rainfall_1h_frame, from_=0, to=100, resolution=0.1, 
                                     orient=tk.HORIZONTAL, variable=self.rainfall_1h_var)
            rainfall_1h_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.rainfall_1h_value_label = ttk.Label(rainfall_1h_frame, text="0.0mm", width=8)
            self.rainfall_1h_value_label.pack(side=tk.RIGHT)
            rainfall_1h_scale.config(command=lambda v: self.rainfall_1h_value_label.config(text=f"{float(v):.1f}mm"))
            
            # Rainfall 3h
            ttk.Label(left_input, text="Rainfall 3h (mm):").pack(anchor='w', pady=(10,0))
            self.rainfall_3h_var = tk.DoubleVar(value=0.0)
            rainfall_3h_frame = ttk.Frame(left_input)
            rainfall_3h_frame.pack(fill=tk.X, pady=2)
            rainfall_3h_scale = tk.Scale(rainfall_3h_frame, from_=0, to=300, resolution=0.5, 
                                     orient=tk.HORIZONTAL, variable=self.rainfall_3h_var)
            rainfall_3h_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.rainfall_3h_value_label = ttk.Label(rainfall_3h_frame, text="0.0mm", width=8)
            self.rainfall_3h_value_label.pack(side=tk.RIGHT)
            rainfall_3h_scale.config(command=lambda v: self.rainfall_3h_value_label.config(text=f"{float(v):.1f}mm"))
            
            # Wind speed
            ttk.Label(left_input, text="Wind Speed (km/h):").pack(anchor='w', pady=(10,0))
            self.wind_var = tk.DoubleVar(value=10.0)
            wind_frame = ttk.Frame(left_input)
            wind_frame.pack(fill=tk.X, pady=2)
            wind_scale = tk.Scale(wind_frame, from_=0, to=100, resolution=1, 
                                 orient=tk.HORIZONTAL, variable=self.wind_var)
            wind_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.wind_value_label = ttk.Label(wind_frame, text="10km/h", width=8)
            self.wind_value_label.pack(side=tk.RIGHT)
            wind_scale.config(command=lambda v: self.wind_value_label.config(text=f"{int(float(v))}km/h"))
            
            # Right column - River data
            ttk.Label(right_input, text="River Data:", style='Header.TLabel').pack(anchor='w')
            
            # Water level
            ttk.Label(right_input, text="Water Level (cm):").pack(anchor='w', pady=(10,0))
            self.water_level_var = tk.DoubleVar(value=150.0)
            water_frame = ttk.Frame(right_input)
            water_frame.pack(fill=tk.X, pady=2)
            water_scale = tk.Scale(water_frame, from_=50, to=500, resolution=1, 
                                  orient=tk.HORIZONTAL, variable=self.water_level_var)
            water_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.water_value_label = ttk.Label(water_frame, text="150cm", width=8)
            self.water_value_label.pack(side=tk.RIGHT)
            water_scale.config(command=lambda v: self.water_value_label.config(text=f"{int(float(v))}cm"))
            
            # Flow rate
            ttk.Label(right_input, text="Flow Rate (m³/s):").pack(anchor='w', pady=(10,0))
            self.flow_rate_var = tk.DoubleVar(value=800.0)
            flow_frame = ttk.Frame(right_input)
            flow_frame.pack(fill=tk.X, pady=2)
            flow_scale = tk.Scale(flow_frame, from_=100, to=3000, resolution=10, 
                                 orient=tk.HORIZONTAL, variable=self.flow_rate_var)
            flow_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.flow_value_label = ttk.Label(flow_frame, text="800m³/s", width=8)
            self.flow_value_label.pack(side=tk.RIGHT)
            flow_scale.config(command=lambda v: self.flow_value_label.config(text=f"{int(float(v))}m³/s"))
            
            # Trend
            ttk.Label(right_input, text="Water Level Trend:").pack(anchor='w', pady=(10,0))
            self.trend_var = tk.StringVar(value="stable")
            trend_combo = ttk.Combobox(right_input, textvariable=self.trend_var, 
                                      values=["stable", "rising", "falling"], state="readonly")
            trend_combo.pack(fill=tk.X, pady=2)
            
            # Location
            ttk.Label(right_input, text="Location:").pack(anchor='w', pady=(10,0))
            self.location_var = tk.StringVar(value="Hanoi")
            location_combo = ttk.Combobox(right_input, textvariable=self.location_var,
                                         values=["Hanoi", "Ho_Chi_Minh_City", "Da_Nang", 
                                                "Hue", "Can_Tho", "Hai_Phong", "Nha_Trang"])
            location_combo.pack(fill=tk.X, pady=2)
            
            # Bind event to load data when location is selected
            location_combo.bind("<<ComboboxSelected>>", self.on_location_selected)
            
            # Predict button
            predict_btn = ttk.Button(right_input, text="PREDICT FLOOD", 
                                    command=self.perform_prediction, style='Accent.TButton')
            predict_btn.pack(pady=20, ipadx=20, ipady=10)
            
            # Bottom frame - Results
            result_frame = ttk.LabelFrame(prediction_frame, text="Prediction Results", padding=10)
            result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            # Left result - Text result
            left_result = ttk.Frame(result_frame)
            left_result.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,10))
            
            self.result_text = tk.Text(left_result, height=15, wrap=tk.WORD, font=('Courier', 10))
            result_scroll = ttk.Scrollbar(left_result, orient="vertical", command=self.result_text.yview)
            self.result_text.configure(yscrollcommand=result_scroll.set)
            self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            result_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Right result - Risk visualization
            right_result = ttk.LabelFrame(result_frame, text="Visual Display", padding=10)
            right_result.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
            
            # Risk level display
            self.risk_display_frame = ttk.Frame(right_result)
            self.risk_display_frame.pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            messagebox.showerror("Error", f"Prediction tab creation failed: {str(e)}")

    def on_location_selected(self, event):
        """Load latest river data for selected location and update UI"""
        try:
            location = self.location_var.get()
            
            conn = get_connection()
            if conn:
                cursor = conn.cursor()
                
                # Query latest river data for the selected location
                query = """
                SELECT water_level, flow_rate, trend FROM river_level_data 
                WHERE location_name = %s ORDER BY created_at DESC LIMIT 1
                """
                
                cursor.execute(query, (location,))
                result = cursor.fetchone()
                
                if result:
                    water_level, flow_rate, trend = result
                    
                    # Update slider variables
                    self.water_level_var.set(float(water_level))
                    self.flow_rate_var.set(float(flow_rate))
                    self.trend_var.set(trend)
                    
                    # Update slider labels
                    self.water_value_label.config(text=f"{int(water_level)}cm")
                    self.flow_value_label.config(text=f"{int(flow_rate)}m³/s")
                    
                    # Update status
                    self.update_status(f"Loaded data for {location}")
                else:
                    # No data found, reset to defaults
                    self.water_level_var.set(150.0)
                    self.flow_rate_var.set(800.0)
                    self.trend_var.set("stable")
                    self.water_value_label.config(text="150cm")
                    self.flow_value_label.config(text="800m³/s")
                    self.update_status(f"No river data found for {location}, using defaults")
                
                cursor.close()
                close_connection(conn)
            else:
                messagebox.showerror("Error", "Cannot connect to database")
                
        except Exception as e:
            self.update_status("Error loading location data")
            messagebox.showerror("Error", f"Error loading data for {location}: {str(e)}")

    def create_data_tab(self):
        """Data Tab - Manage and view data"""
        try:
            data_frame = ttk.Frame(self.notebook)
            self.notebook.add(data_frame, text="Data")
            
            # Create sub-notebook for Rainfall, River, and Predictions data
            self.data_notebook = ttk.Notebook(data_frame)
            self.data_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            # Rainfall Data sub-tab
            self.create_rainfall_data_subtab()
            
            # River Level Data sub-tab
            self.create_river_data_subtab()
            
            # Predictions Data sub-tab
            self.create_predictions_data_subtab()
            
            # Control panel for both sub-tabs
            control_frame = ttk.LabelFrame(data_frame, text="Data Control", padding=10)
            control_frame.pack(fill=tk.X, padx=10, pady=5)
            
            # Buttons row
            btn_frame = ttk.Frame(control_frame)
            btn_frame.pack(fill=tk.X, pady=2)
            
            ttk.Button(btn_frame, text="Crawl Weather", 
                      command=self.crawl_weather_data).pack(side=tk.LEFT, padx=5)
            ttk.Button(btn_frame, text="Crawl River", 
                      command=self.crawl_river_data).pack(side=tk.LEFT, padx=5)
            ttk.Button(btn_frame, text="Refresh All Data", 
                      command=self.refresh_all_data).pack(side=tk.LEFT, padx=5)
            ttk.Button(btn_frame, text="Cleanup DB", 
                      command=self.cleanup_database).pack(side=tk.LEFT, padx=5)
        except Exception as e:
            messagebox.showerror("Error", f"Data tab creation failed: {str(e)}")

    def create_rainfall_data_subtab(self):
        """Create Rainfall Data sub-tab"""
        try:
            rainfall_frame = ttk.Frame(self.data_notebook)
            self.data_notebook.add(rainfall_frame, text="Rainfall Data")
            
            # Control frame for rainfall
            rainfall_control = ttk.Frame(rainfall_frame)
            rainfall_control.pack(fill=tk.X, padx=10, pady=5)
            
            ttk.Button(rainfall_control, text="Refresh Rainfall Data", 
                      command=self.refresh_rainfall_data).pack(side=tk.LEFT, padx=5)
            
            # Treeview frame
            tree_frame = ttk.Frame(rainfall_frame)
            tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            # Treeview for rainfall data
            columns = ['Location', 'Time', 'Temperature', 'Humidity', 'Rainfall 1h', 'Rainfall 3h', 'Wind Speed']
            self.rainfall_tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=20)
            
            # Set column headings and widths
            column_widths = {'Location': 120, 'Time': 150, 'Temperature': 100, 'Humidity': 80, 
                            'Rainfall 1h': 100, 'Rainfall 3h': 100, 'Wind Speed': 100}
            
            for col in columns:
                self.rainfall_tree.heading(col, text=col)
                self.rainfall_tree.column(col, width=column_widths.get(col, 100))
            
            # Scrollbars
            v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.rainfall_tree.yview)
            h_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.rainfall_tree.xview)
            self.rainfall_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
            
            # Pack treeview and scrollbars
            self.rainfall_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        except Exception as e:
            messagebox.showerror("Error", f"Rainfall sub-tab creation failed: {str(e)}")

    def create_river_data_subtab(self):
        """Create River Level Data sub-tab"""
        try:
            river_frame = ttk.Frame(self.data_notebook)
            self.data_notebook.add(river_frame, text="River Level Data")
            
            # Control frame for river
            river_control = ttk.Frame(river_frame)
            river_control.pack(fill=tk.X, padx=10, pady=5)
            
            ttk.Button(river_control, text="Refresh River Data", 
                      command=self.refresh_river_data).pack(side=tk.LEFT, padx=5)
            
            # Treeview frame
            tree_frame = ttk.Frame(river_frame)
            tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            # Treeview for river data
            columns = ['Location', 'Time', 'Water Level', 'Flow Rate', 'Trend']
            self.river_tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=20)
            
            # Set column headings and widths
            column_widths = {'Location': 120, 'Time': 150, 'Water Level': 100, 'Flow Rate': 100, 'Trend': 80}
            
            for col in columns:
                self.river_tree.heading(col, text=col)
                self.river_tree.column(col, width=column_widths.get(col, 100))
            
            # Scrollbars
            v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.river_tree.yview)
            h_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.river_tree.xview)
            self.river_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
            
            # Pack treeview and scrollbars
            self.river_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        except Exception as e:
            messagebox.showerror("Error", f"River sub-tab creation failed: {str(e)}")

    def create_predictions_data_subtab(self):
        """Create Predictions Data sub-tab"""
        try:
            predictions_frame = ttk.Frame(self.data_notebook)
            self.data_notebook.add(predictions_frame, text="Predictions")
            
            # Treeview for predictions
            columns = ('Location', 'Time', 'Risk', 'Probability', 'Water Level', 
                      'Rain 1h', 'Rain 3h', 'Alert Level', 'Version')
            self.predictions_tree = ttk.Treeview(predictions_frame, columns=columns, show='headings')
            
            # Define headings
            column_widths = {
                'Location': 100,
                'Time': 150,
                'Risk': 80,
                'Probability': 90,
                'Water Level': 90,
                'Rain 1h': 80,
                'Rain 3h': 80,
                'Alert Level': 120,
                'Version': 100
            }
            
            for col in columns:
                self.predictions_tree.heading(col, text=col)
                self.predictions_tree.column(col, width=column_widths.get(col, 80))
            
            self.predictions_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Scrollbar
            scrollbar = ttk.Scrollbar(predictions_frame, orient=tk.VERTICAL, 
                                     command=self.predictions_tree.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            self.predictions_tree.configure(yscrollcommand=scrollbar.set)
            
            # Refresh button
            ttk.Button(predictions_frame, text="Refresh", 
                      command=self.refresh_predictions_data).pack(pady=5)
            
        except Exception as e:
            print(f"Error creating predictions tab: {e}")

    def create_reports_tab(self):
        """Reports Tab - Statistics and charts"""
        try:
            reports_frame = ttk.Frame(self.notebook)
            self.notebook.add(reports_frame, text="Reports")
            
            # Control frame
            control_frame = ttk.LabelFrame(reports_frame, text="Report Options", padding=10)
            control_frame.pack(fill=tk.X, padx=10, pady=5)
            
            # Date range selection
            ttk.Label(control_frame, text="Time Range:").pack(side=tk.LEFT)
            self.date_range_var = tk.StringVar(value="7 days")
            date_combo = ttk.Combobox(control_frame, textvariable=self.date_range_var,
                                     values=["1 day", "7 days", "30 days", "All"],
                                     state="readonly", width=15)
            date_combo.pack(side=tk.LEFT, padx=5)
            
            ttk.Button(control_frame, text="Generate Report", 
                      command=self.generate_reports).pack(side=tk.LEFT, padx=10)
            ttk.Button(control_frame, text="Export to Excel", 
                      command=self.export_to_excel).pack(side=tk.LEFT, padx=5)
            
            # Charts frame
            charts_frame = ttk.LabelFrame(reports_frame, text="Analysis Charts", padding=10)
            charts_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            # Matplotlib figure for reports
            self.reports_fig, self.reports_axes = plt.subplots(2, 2, figsize=(12, 8))
            self.reports_fig.suptitle("Flood Data Analysis Report")
            
            self.reports_canvas = FigureCanvasTkAgg(self.reports_fig, charts_frame)
            self.reports_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            messagebox.showerror("Error", f"Reports tab creation failed: {str(e)}")

    def create_settings_tab(self):
        """Settings Tab - System configuration"""
        try:
            settings_frame = ttk.Frame(self.notebook)
            self.notebook.add(settings_frame, text="Settings")
            
            # Database settings
            db_frame = ttk.LabelFrame(settings_frame, text="Database Settings", padding=15)
            db_frame.pack(fill=tk.X, padx=10, pady=5)
            
            # Create grid
            ttk.Label(db_frame, text="Host:").grid(row=0, column=0, sticky='w', pady=5)
            self.db_host_var = tk.StringVar(value="localhost")
            ttk.Entry(db_frame, textvariable=self.db_host_var, width=30).grid(row=0, column=1, padx=10, pady=5)
            
            ttk.Label(db_frame, text="Port:").grid(row=1, column=0, sticky='w', pady=5)
            self.db_port_var = tk.StringVar(value="3306")
            ttk.Entry(db_frame, textvariable=self.db_port_var, width=30).grid(row=1, column=1, padx=10, pady=5)
            
            ttk.Label(db_frame, text="Username:").grid(row=2, column=0, sticky='w', pady=5)
            self.db_user_var = tk.StringVar(value="root")
            ttk.Entry(db_frame, textvariable=self.db_user_var, width=30).grid(row=2, column=1, padx=10, pady=5)
            
            ttk.Label(db_frame, text="Password:").grid(row=3, column=0, sticky='w', pady=5)
            self.db_pass_var = tk.StringVar(value="")
            ttk.Entry(db_frame, textvariable=self.db_pass_var, show="*", width=30).grid(row=3, column=1, padx=10, pady=5)
            
            # Buttons
            btn_frame = ttk.Frame(db_frame)
            btn_frame.grid(row=4, column=0, columnspan=2, pady=15)
            ttk.Button(btn_frame, text="Test Connection", command=self.test_db_connection).pack(side=tk.LEFT, padx=5)
            ttk.Button(btn_frame, text="Save Settings", command=self.save_db_settings).pack(side=tk.LEFT, padx=5)
            
            # API settings
            api_frame = ttk.LabelFrame(settings_frame, text="API Settings", padding=15)
            api_frame.pack(fill=tk.X, padx=10, pady=5)
            
            ttk.Label(api_frame, text="Windy API Key:").grid(row=0, column=0, sticky='w', pady=5)
            self.api_key_var = tk.StringVar()
            ttk.Entry(api_frame, textvariable=self.api_key_var, width=50, show="*").grid(row=0, column=1, padx=10, pady=5)
            
            api_btn_frame = ttk.Frame(api_frame)
            api_btn_frame.grid(row=1, column=0, columnspan=2, pady=10)
            ttk.Button(api_btn_frame, text="Test API", command=self.test_api_key).pack(side=tk.LEFT, padx=5)
            ttk.Button(api_btn_frame, text="Save API Key", command=self.save_api_key).pack(side=tk.LEFT, padx=5)
            
            # Model settings
            model_frame = ttk.LabelFrame(settings_frame, text="Model Settings", padding=15)
            model_frame.pack(fill=tk.X, padx=10, pady=5)
            
            ttk.Label(model_frame, text="Number of Random Forest Trees:").grid(row=0, column=0, sticky='w', pady=5)
            self.n_estimators_var = tk.IntVar(value=150)
            estimators_frame = ttk.Frame(model_frame)
            estimators_frame.grid(row=0, column=1, sticky='ew', padx=10, pady=5)
            ttk.Scale(estimators_frame, from_=50, to=500, orient=tk.HORIZONTAL, 
                     variable=self.n_estimators_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.estimators_label = ttk.Label(estimators_frame, text="150")
            self.estimators_label.pack(side=tk.RIGHT)
            
            ttk.Label(model_frame, text="Max Depth:").grid(row=1, column=0, sticky='w', pady=5)
            self.max_depth_var = tk.IntVar(value=10)
            depth_frame = ttk.Frame(model_frame)
            depth_frame.grid(row=1, column=1, sticky='ew', padx=10, pady=5)
            ttk.Scale(depth_frame, from_=1, to=30, orient=tk.HORIZONTAL,
                     variable=self.max_depth_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.depth_label = ttk.Label(depth_frame, text="10")
            self.depth_label.pack(side=tk.RIGHT)
        except Exception as e:
            messagebox.showerror("Error", f"Settings tab creation failed: {str(e)}")

    def create_status_bar(self):
        """Create status bar"""
        try:
            self.status_bar = ttk.Frame(self.root, relief=tk.SUNKEN, borderwidth=1)
            self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
            
            self.status_label = ttk.Label(self.status_bar, text="Ready", relief=tk.SUNKEN)
            self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=2)
            
            self.progress_bar = ttk.Progressbar(self.status_bar, length=200, mode='indeterminate')
            self.progress_bar.pack(side=tk.RIGHT, padx=5, pady=2)
            
            # Add Exit button to status bar
            ttk.Button(self.status_bar, text="Exit", command=self.root.quit).pack(side=tk.RIGHT, padx=5, pady=2)
        except Exception as e:
            print(f"Status bar creation error: {e}")

    def update_status(self, message, show_progress=False):
        """Update status bar"""
        try:
            self.status_label.config(text=message)
            if show_progress:
                self.progress_bar.start()
            else:
                self.progress_bar.stop()
            self.root.update_idletasks()
        except Exception as e:
            print(f"Status update error: {e}")

    def check_database_connection(self):
        """Check database connection"""
        try:
            if not IMPORT_SUCCESS:
                self.db_status_label.config(text="Import Error", style='Error.TLabel')
                return
            conn = get_connection()
            if conn:
                self.db_status_label.config(text="Connection Successful", style='Success.TLabel')
                close_connection(conn)
            else:
                self.db_status_label.config(text="Not Connected", style='Error.TLabel')
        except Exception as e:
            self.db_status_label.config(text=f"Error: {str(e)}", style='Error.TLabel')

    def refresh_dashboard(self):
        """Refresh dashboard"""
        try:
            self.update_status("Refreshing dashboard...", True)
            
            # Check database
            self.check_database_connection()
            
            # Update data summary
            self.update_data_summary()
            
            # Update charts
            self.update_dashboard_charts()
            
            self.update_status("Dashboard refreshed")
        except Exception as e:
            self.update_status(f"Error refreshing dashboard: {str(e)}")
            messagebox.showerror("Error", f"Cannot refresh dashboard:\n{str(e)}")

    def update_data_summary(self):
        """Update data statistics"""
        try:
            self.data_summary_text.delete(1.0, tk.END)
            
            if not IMPORT_SUCCESS:
                self.data_summary_text.insert(tk.END, "Import error - cannot load data")
                return
            
            conn = get_connection()
            if not conn:
                self.data_summary_text.insert(tk.END, "Cannot connect to database")
                return
            
            cursor = conn.cursor()
            
            # Count records
            try:
                cursor.execute("SELECT COUNT(*) FROM rainfall_data")
                weather_count = cursor.fetchone()[0]
            except:
                weather_count = 0
                
            try:
                cursor.execute("SELECT COUNT(*) FROM river_level_data")
                river_count = cursor.fetchone()[0]
            except:
                river_count = 0
                
            try:
                cursor.execute("SELECT COUNT(*) FROM flood_predictions")
                prediction_count = cursor.fetchone()[0]
            except:
                prediction_count = 0
            
            # Latest data
            try:
                cursor.execute("SELECT MAX(created_at) FROM rainfall_data")
                latest_weather = cursor.fetchone()[0]
            except:
                latest_weather = "N/A"
            
            summary = f"""Weather Data: {weather_count} records
River Data: {river_count} records  
Predictions Made: {prediction_count} records

Latest Data: {latest_weather}

Status: {'Complete' if river_count > 0 else 'Weather Only'}
Model: {'3 Levels' if river_count > 0 else '2 Levels'}
"""
            
            self.data_summary_text.insert(tk.END, summary)
            
            cursor.close()
            close_connection(conn)
            
        except Exception as e:
            self.data_summary_text.insert(tk.END, f"Error: {str(e)}")

    def update_dashboard_charts(self):
        """Update dashboard charts"""
        try:
            # Clear previous plots
            for ax in self.dashboard_axes.flat:
                ax.clear()
            
            if not IMPORT_SUCCESS:
                self.dashboard_axes[0,0].text(0.5, 0.5, 'Import error', 
                                            ha='center', va='center', transform=self.dashboard_axes[0,0].transAxes)
                self.dashboard_canvas.draw()
                return
            
            # Load data
            df = load_combined_data()
            if df is None or len(df) == 0:
                df = load_data_from_db()
            
            if df is None or len(df) == 0:
                # No data available
                self.dashboard_axes[0,0].text(0.5, 0.5, 'No data', 
                                            ha='center', va='center', transform=self.dashboard_axes[0,0].transAxes)
                self.dashboard_canvas.draw()
                return
            
            # Chart 1: Temperature trend (Top-Left)
            if len(df) > 0 and 'temperature' in df.columns:
                temps = df['temperature'].tail(20).values
                self.dashboard_axes[0,0].plot(temps, 'b-o', markersize=3)
                self.dashboard_axes[0,0].set_title('Temperature Trend (Last 20 Samples)')
                self.dashboard_axes[0,0].set_ylabel('°C')
                self.dashboard_axes[0,0].grid(True, alpha=0.3)
            
            # Chart 2: Rainfall distribution (Top-Right)
            if 'rainfall_1h' in df.columns:
                rainfall_data = df['rainfall_1h'].values
                rainfall_data = rainfall_data[rainfall_data >= 0]  # Remove negative values
                self.dashboard_axes[0,1].hist(rainfall_data, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
                self.dashboard_axes[0,1].set_title('Rainfall Distribution')
                self.dashboard_axes[0,1].set_xlabel('mm/h')
                self.dashboard_axes[0,1].set_ylabel('Frequency')
            
            # Chart 3: Risk levels (Bottom-Left)
            if 'flood_risk_level' in df.columns:
                risk_counts = df['flood_risk_level'].value_counts().sort_index()
                labels = ['LOW', 'MODERATE', 'HIGH']
                colors = ['green', 'orange', 'red']
                if len(risk_counts) > 0:
                    self.dashboard_axes[1,0].pie(risk_counts.values, 
                                                labels=[labels[i] for i in risk_counts.index],
                                                colors=[colors[i] for i in risk_counts.index],
                                                autopct='%1.1f%%', startangle=90)
                    self.dashboard_axes[1,0].set_title('Risk Level Distribution')
            elif 'flood_risk' in df.columns:
                risk_counts = df['flood_risk'].value_counts()
                labels = ['No Flood', 'Flood']
                colors = ['green', 'red']
                if len(risk_counts) > 0:
                    self.dashboard_axes[1,0].pie(risk_counts.values, labels=labels, 
                                                colors=colors, autopct='%1.1f%%', startangle=90)
                    self.dashboard_axes[1,0].set_title('Flood Risk Distribution')
            
            # Chart 4: Water level trend (Bottom-Right)
            if 'water_level' in df.columns:
                water_levels = df['water_level'].tail(20).values
                self.dashboard_axes[1,1].plot(water_levels, 'r-o', markersize=3)
                self.dashboard_axes[1,1].set_title('Water Level Trend (Last 20 Samples)')
                self.dashboard_axes[1,1].set_ylabel('cm')
                self.dashboard_axes[1,1].grid(True, alpha=0.3)
                
                # Add alert levels if available
                if 'alert_level_1' in df.columns:
                    alert1 = df['alert_level_1'].iloc[0] if len(df) > 0 else 180
                    alert2 = df['alert_level_2'].iloc[0] if len(df) > 0 else 220
                    alert3 = df['alert_level_3'].iloc[0] if len(df) > 0 else 270
                    
                    self.dashboard_axes[1,1].axhline(y=alert1, color='green', linestyle='--', alpha=0.7, label='Low Alert')
                    self.dashboard_axes[1,1].axhline(y=alert2, color='yellow', linestyle='--', alpha=0.7, label='Moderate Alert')
                    self.dashboard_axes[1,1].axhline(y=alert3, color='red', linestyle='--', alpha=0.7, label='High Alert')
                    self.dashboard_axes[1,1].legend()
            else:
                # Show humidity instead
                if 'humidity' in df.columns:
                    humidity_data = df['humidity'].tail(20).values
                    self.dashboard_axes[1,1].plot(humidity_data, 'g-o', markersize=3)
                    self.dashboard_axes[1,1].set_title('Humidity Trend')
                    self.dashboard_axes[1,1].set_ylabel('%')
                    self.dashboard_axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            self.dashboard_canvas.draw()
            
        except Exception as e:
            print(f"Error updating charts: {e}")
            # Show error in first plot
            try:
                self.dashboard_axes[0,0].text(0.5, 0.5, f'Error: {str(e)}', 
                                            ha='center', va='center', transform=self.dashboard_axes[0,0].transAxes)
                self.dashboard_canvas.draw()
            except:
                pass

    def calculate_alert_level(self, water_level):
        """Calculate alert level based on water level"""
        if water_level < 180:
            return 0
        elif water_level < 220:
            return 1
        elif water_level < 270:
            return 2
        else:
            return 3

    def calculate_alert_level_numeric(self, water_level, risk_level):
        """Calculate numeric alert level (1-3) based on water level and risk"""
        # Base calculation from water level
        if water_level < 180:
            base_level = 1  # Low alert
        elif water_level < 220:
            base_level = 2  # Moderate alert
        else:
            base_level = 3  # High alert
        
        # Adjust based on risk level
        if risk_level == 'HIGH' and base_level < 3:
            base_level += 1
        elif risk_level == 'LOW' and base_level > 1:
            base_level -= 1
        
        # Ensure level is within 1-3 range
        return max(1, min(3, base_level))

    def display_prediction_result(self, result, input_data):
        """Display prediction result in GUI"""
        try:
            # Clear previous result
            self.result_text.delete(1.0, tk.END)
            
            # Calculate probability
            weather_factor = result.get('weather_factor', 0.0)
            river_factor = result.get('river_factor', 0.0)
            combined_score = result.get('combined_score', 0.0)
            risk_level = result.get('risk_level', 'LOW')
            
            # Calculate probability percentage
            if combined_score > 0:
                probability = min(combined_score * 100, 99.99)
            else:
                if risk_level == 'LOW':
                    probability = 15.0
                elif risk_level == 'MODERATE':
                    probability = 50.0
                elif risk_level == 'HIGH':
                    probability = 85.0
                else:
                    probability = 0.0
            
            # Calculate alert level
            water_level = input_data.get('water_level', 0.0)
            alert_level = self.calculate_alert_level_numeric(water_level, risk_level)
            alert_names = {1: "Low Alert", 2: "Moderate Alert", 3: "High Alert"}
            
            # Format result text
            result_str = f"""
{'='*50}
FLOOD RISK PREDICTION RESULT
{'='*50}

Location: {self.location_var.get()}
Prediction Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

INPUT DATA:
-----------
Temperature: {input_data['temperature']:.1f}°C
Humidity: {input_data['humidity']:.1f}%
Pressure: {input_data['pressure']:.1f} hPa
Rainfall (1h): {input_data['rainfall_1h']:.1f} mm
Rainfall (3h): {input_data['rainfall_3h']:.1f} mm
"""
            
            if self.is_advanced and 'water_level' in input_data:
                result_str += f"""
Water Level: {input_data['water_level']:.1f} cm
Flow Rate: {input_data['flow_rate']:.1f} m³/s
Trend: {self.trend_var.get()}
"""
            
            result_str += f"""
PREDICTION RESULT:
------------------
Risk Level: {risk_level}
Flood Probability: {probability:.2f}%
Alert Level: Level {alert_level} - {alert_names[alert_level]}
"""
            
            if weather_factor > 0 or river_factor > 0:
                result_str += f"""
RISK FACTORS:
-------------
Weather Risk Factor: {weather_factor:.4f}
River Risk Factor: {river_factor:.4f}
Combined Risk Score: {combined_score:.4f}
"""
            
            if 'recommendations' in result:
                result_str += f"""
RECOMMENDATIONS:
----------------
{result['recommendations']}
"""
            
            result_str += f"\n{'='*50}"
            
            self.result_text.insert(tk.END, result_str)
            
            # Update visual display with probability
            result['flood_probability'] = probability
            self.update_risk_visualization(result)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error displaying result: {str(e)}")

    def update_risk_visualization(self, result):
        """Update risk level visualization"""
        try:
            # Clear previous widgets
            for widget in self.risk_display_frame.winfo_children():
                widget.destroy()
            
            risk_level = result.get('risk_level', 'LOW')
            probability = result.get('probability_flood', 0)
            
            # Color mapping
            colors = {'LOW': '#27ae60', 'MODERATE': '#f39c12', 'HIGH': '#e74c3c'}
            color = colors.get(risk_level, '#95a5a6')
            
            # Create risk level label
            risk_label = tk.Label(self.risk_display_frame, 
                                 text=risk_level,
                                 font=('Arial', 48, 'bold'),
                                 fg=color,
                                 bg='white')
            risk_label.pack(pady=20)
            
            # Create probability bar
            prob_frame = ttk.Frame(self.risk_display_frame)
            prob_frame.pack(fill=tk.X, padx=20, pady=10)
            
            ttk.Label(prob_frame, text="Flood Probability:", 
                     font=('Arial', 12, 'bold')).pack()
            
            progress = ttk.Progressbar(prob_frame, length=300, mode='determinate')
            progress['value'] = probability * 100
            progress.pack(pady=5)
            
            ttk.Label(prob_frame, text=f"{probability:.1%}", 
                     font=('Arial', 14, 'bold')).pack()
            
            # Add icon or additional info based on risk level
            info_text = {
                'LOW': '✓ Low flood risk\nContinue normal activities',
                'MODERATE': '⚠ Moderate flood risk\nStay alert and monitor situation',
                'HIGH': '⚠ High flood risk!\nPrepare for evacuation if needed'
            }
            
            info_label = tk.Label(self.risk_display_frame,
                                 text=info_text.get(risk_level, ''),
                                 font=('Arial', 11),
                                 fg=color,
                                 justify=tk.CENTER)
            info_label.pack(pady=10)
            
        except Exception as e:
            print(f"Error updating risk visualization: {e}")

    def perform_prediction(self):
        """Perform flood prediction"""
        try:
            if not IMPORT_SUCCESS:
                messagebox.showerror("Error", "Import error - cannot perform prediction")
                return
            
            if self.model is None:
                messagebox.showwarning("Warning", "No model available. Please train a model first!")
                return
            
            # Collect input data
            input_data = {
                'temperature': self.temp_var.get(),
                'humidity': self.humidity_var.get(),
                'pressure': self.pressure_var.get(),
                'rainfall_1h': self.rainfall_1h_var.get(),
                'rainfall_3h': self.rainfall_3h_var.get() * 2.5,
                'wind_speed': self.wind_var.get(),
            }
            
            # Add river data if using advanced model
            if self.is_advanced:
                water_level = self.water_level_var.get()
                normal_level = 150.0
                
                input_data.update({
                    'water_level': water_level,
                    'water_level_ratio': water_level / normal_level,
                    'normal_level': normal_level,
                    'alert_level_1': 180.0,
                    'alert_level_2': 220.0,
                    'alert_level_3': 270.0,
                    'flow_rate': self.flow_rate_var.get(),
                    'flow_rate_normal': self.flow_rate_var.get() / 1000.0,
                    'alert_level_exceeded': self.calculate_alert_level(water_level),
                    'trend_rising': 1 if self.trend_var.get() == 'rising' else 0,
                    'trend_falling': 1 if self.trend_var.get() == 'falling' else 0,
                })
            
            # Perform prediction
            result = predict_flood_risk(self.model, self.features, input_data, self.is_advanced)
            
            if result:
                self.display_prediction_result(result, input_data)
                self.save_prediction_to_db(result, input_data)
            else:
                messagebox.showerror("Error", "Unable to perform prediction!")
                
        except Exception as e:
            messagebox.showerror("Error", f"Prediction error: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def save_prediction_to_db(self, result, input_data):
        """Save prediction result to database"""
        try:
            conn = get_connection()
            if not conn:
                return
            
            cursor = conn.cursor()
            
            # Extract values with defaults
            location = self.location_var.get() if hasattr(self, 'location_var') else 'Unknown'
            risk_level = result.get('risk_level', 'LOW')
            
            # Calculate probability based on risk level and factors
            weather_factor = result.get('weather_factor', 0.0)
            river_factor = result.get('river_factor', 0.0)
            combined_score = result.get('combined_score', 0.0)
            
            # Calculate probability from combined score (0-1 range)
            if combined_score > 0:
                probability = min(combined_score, 0.9999)  # Max 99.99%
            else:
                # Fallback: calculate from risk level
                if risk_level == 'LOW':
                    probability = 0.15  # 15%
                elif risk_level == 'MODERATE':
                    probability = 0.50  # 50%
                elif risk_level == 'HIGH':
                    probability = 0.85  # 85%
                else:
                    probability = 0.0
            
            rainfall_1h = input_data.get('rainfall_1h', 0.0)
            rainfall_3h = input_data.get('rainfall_3h', 0.0)
            water_level = input_data.get('water_level', 0.0)
            
            # Calculate alert_level (1-3) based on water level and risk
            alert_level = self.calculate_alert_level_numeric(water_level, risk_level)
            
            recommendations = result.get('recommendations', 'No recommendations available')
            model_version = 'v1.0' if not self.is_advanced else 'v2.0-advanced'
            
            # Insert prediction with all individual columns
            cursor.execute("""
                INSERT INTO flood_predictions 
                (location_name, prediction_time, risk_level, probability, 
                 weather_factor, river_factor, combined_score,
                 rainfall_1h, rainfall_3h, water_level, 
                 alert_level_exceeded, recommendations, model_version)
                VALUES (%s, NOW(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                location,           # 1. location_name
                risk_level,         # 2. risk_level
                probability,        # 3. probability (0-1)
                weather_factor,     # 4. weather_factor
                river_factor,       # 5. river_factor
                combined_score,     # 6. combined_score
                rainfall_1h,        # 7. rainfall_1h
                rainfall_3h,        # 8. rainfall_3h
                water_level,        # 9. water_level
                alert_level,        # 10. alert_level_exceeded (1-3)
                recommendations,    # 11. recommendations
                model_version       # 12. model_version
            ))
            
            conn.commit()
            cursor.close()
            close_connection(conn)
            
            print(f"Prediction saved: {location}, Risk: {risk_level}, Probability: {probability*100:.1f}%, Alert Level: {alert_level}")
            
        except Exception as e:
            print(f"Error saving prediction to DB: {e}")
            import traceback
            print(traceback.format_exc())
            if conn:
                try:
                    conn.rollback()
                    close_connection(conn)
                except:
                    pass

    def train_prediction_model(self):
        """Train prediction model"""
        try:
            if not IMPORT_SUCCESS:
                messagebox.showerror("Error", "Import error - cannot train model")
                return
            
            self.update_status("Training model...", True)
            df = load_combined_data()
            if df is None or len(df) == 0:
                df = load_data_from_db()
            
            if df is None or len(df) == 0:
                self.update_status("Model training failed")
                messagebox.showerror("Error", "No data available for training")
                return
            
            train_data = generate_advanced_training_data(df)
            result = train_model(train_data)
            
            if result is None:
                raise ValueError("train_model returned None")
            if not isinstance(result, (list, tuple)):
                raise ValueError("train_model returned invalid type")
            if len(result) == 2:
                self.model, self.features = result
            elif len(result) >= 3:
                self.model, self.features, *_ = result
            else:
                raise ValueError("train_model returned insufficient values")
            
            if self.model:
                self.is_advanced = True
                self.model_status_label.config(text="Trained (Advanced)", style='Success.TLabel')
                self.update_status("Model trained successfully")
                messagebox.showinfo("Success", "Model trained successfully!")
            else:
                self.update_status("Model training failed")
                messagebox.showerror("Error", "Model training failed")
                
        except Exception as e:
            self.update_status("Model training failed")
            messagebox.showerror("Error", f"Model training failed: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def evaluate_model(self):
        """Evaluate model"""
        try:
            if not IMPORT_SUCCESS:
                messagebox.showerror("Error", "Import error - cannot evaluate model")
                return
            
            if self.model is None:
                messagebox.showwarning("Warning", "No model available. Please train a model first!")
                return
            
            self.update_status("Evaluating model...", True)
            df = load_combined_data()
            if df is None or len(df) == 0:
                df = load_data_from_db()
            
            if df is None or len(df) == 0:
                self.update_status("Model evaluation failed")
                messagebox.showerror("Error", "No data available for evaluation")
                return
            
            train_data = generate_advanced_training_data(df)
            
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report, confusion_matrix
            
            X = train_data[self.features]
            y = train_data['flood_risk_level']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            y_pred = self.model.predict(X_test)
            
            report = classification_report(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            
            # Show results in messagebox
            result_text = f"Model Evaluation Results:\n\n{report}\n\nConfusion Matrix:\n{cm}"
            messagebox.showinfo("Model Evaluation", result_text)
            
            self.update_status("Model evaluated successfully")
            
        except Exception as e:
            self.update_status("Model evaluation failed")
            messagebox.showerror("Error", f"Model evaluation failed: {str(e)}")
            import traceback
            print(traceback.format_exc())

    # Data management methods
    def refresh_rainfall_data(self):
        """Refresh rainfall data display"""
        try:
            if not IMPORT_SUCCESS:
                return
            
            self.update_status("Loading rainfall data...", True)
            
            # Clear existing data
            for item in self.rainfall_tree.get_children():
                self.rainfall_tree.delete(item)
            
            conn = get_connection()
            if not conn:
                messagebox.showerror("Error", "Cannot connect to database")
                return
            
            cursor = conn.cursor()
            
            # Query with JSON extraction
            cursor.execute("""
                SELECT 
                    location_name,
                    created_at,
                    JSON_EXTRACT(precipitation, '$.temperature') as temperature,
                    JSON_EXTRACT(precipitation, '$.humidity') as humidity,
                    JSON_EXTRACT(precipitation, '$.rainfall_1h') as rainfall_1h,
                    JSON_EXTRACT(precipitation, '$.rainfall_3h') as rainfall_3h,
                    JSON_EXTRACT(precipitation, '$.wind_speed') as wind_speed
                FROM rainfall_data 
                ORDER BY created_at DESC 
                LIMIT 100
            """)
            
            rows = cursor.fetchall()
            
            # Insert data into treeview
            for row in rows:
                location, time, temp, humidity, rain_1h, rain_3h, wind = row
                
                # Clean and convert values (remove quotes from JSON extraction)
                def clean_value(val):
                    if val is None or val == 'null':
                        return None
                    # Remove quotes if present
                    if isinstance(val, str):
                        val = val.strip('"')
                    try:
                        return float(val)
                    except:
                        return None
                
                temp = clean_value(temp)
                humidity = clean_value(humidity)
                rain_1h = clean_value(rain_1h)
                rain_3h = clean_value(rain_3h)
                wind = clean_value(wind)
                
                # Convert temperature from Kelvin to Celsius if needed
                if temp and temp > 100:  # Likely in Kelvin
                    temp = temp - 273.15
                
                display_row = (
                    location,
                    str(time)[:19] if time else 'N/A',
                    f"{temp:.1f}°C" if temp else 'N/A',
                    f"{humidity:.0f}%" if humidity else 'N/A',
                    f"{rain_1h:.1f}mm" if rain_1h else '0.0mm',
                    f"{rain_3h:.1f}mm" if rain_3h else '0.0mm',
                    f"{wind:.1f}km/h" if wind else 'N/A'
                )
                self.rainfall_tree.insert('', tk.END, values=display_row)
            
            cursor.close()
            close_connection(conn)
            self.update_status(f"Loaded {len(rows)} rainfall records")
            
        except Exception as e:
            self.update_status("Error loading rainfall data")
            messagebox.showerror("Error", f"Error loading rainfall data: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def refresh_river_data(self):
        """Refresh river data"""
        try:
            if not IMPORT_SUCCESS:
                return
            
            self.update_status("Loading river data...", True)
            
            # Clear existing data
            for item in self.river_tree.get_children():
                self.river_tree.delete(item)
            
            conn = get_connection()
            if not conn:
                messagebox.showerror("Error", "Cannot connect to database")
                return
            
            cursor = conn.cursor()
            cursor.execute("""
                SELECT location_name, created_at, water_level, flow_rate, trend 
                FROM river_level_data 
                ORDER BY created_at DESC 
                LIMIT 100
            """)
            
            rows = cursor.fetchall()
            for row in rows:
                self.river_tree.insert('', tk.END, values=row)
            
            cursor.close()
            close_connection(conn)
            self.update_status(f"Loaded {len(rows)} river records")
            
        except Exception as e:
            self.update_status("Error loading river data")
            messagebox.showerror("Error", f"Error loading river data: {str(e)}")

    def refresh_predictions_data(self):
        """Refresh predictions data"""
        try:
            if not IMPORT_SUCCESS:
                return
            
            self.update_status("Loading predictions...", True)
            
            # Clear existing data
            for item in self.predictions_tree.get_children():
                self.predictions_tree.delete(item)
            
            conn = get_connection()
            if not conn:
                messagebox.showerror("Error", "Cannot connect to database")
                return
            
            cursor = conn.cursor()
            
            # Query predictions from flood_predictions table
            cursor.execute("""
                SELECT 
                    location_name,
                    prediction_time,
                    risk_level,
                    probability,
                    water_level,
                    rainfall_1h,
                    rainfall_3h,
                    alert_level_exceeded,
                    model_version
                FROM flood_predictions 
                ORDER BY prediction_time DESC 
                LIMIT 100
            """)
            
            rows = cursor.fetchall()
            
            # Alert level names
            alert_names = {1: "Low", 2: "Moderate", 3: "High"}
            
            # Insert data into treeview
            for row in rows:
                location, time, risk, prob, water, rain_1h, rain_3h, alert_level, version = row
                
                display_row = (
                    location,
                    str(time)[:19] if time else 'N/A',
                    risk or 'N/A',
                    f"{float(prob)*100:.1f}%" if prob else 'N/A',
                    f"{float(water):.0f}cm" if water else 'N/A',
                    f"{float(rain_1h):.1f}mm" if rain_1h else '0.0mm',
                    f"{float(rain_3h):.1f}mm" if rain_3h else '0.0mm',
                    f"L{alert_level} - {alert_names.get(alert_level, 'N/A')}" if alert_level else 'N/A',
                    version or 'N/A'
                )
                self.predictions_tree.insert('', tk.END, values=display_row)
            
            cursor.close()
            close_connection(conn)
            self.update_status(f"Loaded {len(rows)} prediction records")
            
        except Exception as e:
            self.update_status("Error loading predictions")
            messagebox.showerror("Error", f"Error loading predictions: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def refresh_all_data(self):
        """Refresh all data views"""
        try:
            self.refresh_rainfall_data()
            self.refresh_river_data()
            self.refresh_predictions_data()
            self.refresh_dashboard()
        except Exception as e:
            messagebox.showerror("Error", f"Error refreshing data: {str(e)}")

    def clear_old_predictions(self):
        """Clear old predictions from database"""
        try:
            if not IMPORT_SUCCESS:
                return
            
            response = messagebox.askyesno("Confirm", "Delete predictions older than 30 days?")
            if not response:
                return
            
            conn = get_connection()
            if not conn:
                messagebox.showerror("Error", "Cannot connect to database")
                return
            
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM flood_predictions 
                WHERE prediction_time < DATE_SUB(NOW(), INTERVAL 30 DAY)
            """)
            deleted_count = cursor.rowcount
            conn.commit();
            
            cursor.close()
            close_connection(conn);
            
            messagebox.showinfo("Success", f"Deleted {deleted_count} old predictions")
            self.refresh_predictions_data();
            
        except Exception as e:
            messagebox.showerror("Error", f"Error clearing predictions: {str(e)}")

    # Crawler methods
    def crawl_weather_data(self):
        """Run weather data crawler"""
        try:
            self.update_status("Crawling weather data...", True)
            
            def run_crawler():
                try:
                    # Use the correct filename: rainfall_crawler.py
                    subprocess.run([sys.executable, "rainfall_crawler.py"], check=True)
                    self.root.after(0, lambda: self.update_status("Weather data crawled successfully"))
                    self.root.after(0, self.refresh_rainfall_data)
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Weather crawl failed: {str(e)}"))
                    self.root.after(0, lambda: self.update_status("Weather crawl failed"))
            
            thread = threading.Thread(target=run_crawler)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            self.update_status("Weather crawl failed")
            messagebox.showerror("Error", f"Cannot start weather crawler: {str(e)}")

    def crawl_river_data(self):
        """Run river data crawler"""
        try:
            self.update_status("Crawling river data...", True)
            
            def run_crawler():
                try:
                    # Use the correct filename: river_level_crawler.py
                    subprocess.run([sys.executable, "river_level_crawler.py"], check=True)
                    self.root.after(0, lambda: self.update_status("River data crawled successfully"))
                    self.root.after(0, self.refresh_river_data)
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror("Error", f"River crawl failed: {str(e)}"))
                    self.root.after(0, lambda: self.update_status("River crawl failed"))
            
            thread = threading.Thread(target=run_crawler)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            self.update_status("River crawl failed")
            messagebox.showerror("Error", f"Cannot start river crawler: {str(e)}")

    # Database management methods
    def setup_database(self):
        """Setup database"""
        try:
            if not IMPORT_SUCCESS:
                messagebox.showerror("Error", "Import error - cannot setup database")
                return
            
            response = messagebox.askyesno("Confirm", "This will create/reset database tables. Continue?")
            if not response:
                return
            
            self.update_status("Setting up database...", True)
            setup_database()
            self.update_status("Database setup complete")
            messagebox.showinfo("Success", "Database setup successfully!")
            self.check_database_connection()
            
        except Exception as e:
            self.update_status("Database setup failed")
            messagebox.showerror("Error", f"Database setup failed: {str(e)}")

    def manage_database(self):
        """Open database management window"""
        try:
            manage_window = tk.Toplevel(self.root)
            manage_window.title("Database Management")
            manage_window.geometry("600x400")
            
            ttk.Label(manage_window, text="Database Management", 
                     style='Title.TLabel').pack(pady=10)
            
            btn_frame = ttk.Frame(manage_window)
            btn_frame.pack(pady=20)
            
            ttk.Button(btn_frame, text="Backup Database", 
                      command=self.backup_database).pack(pady=5, fill=tk.X)
            ttk.Button(btn_frame, text="Restore Database", 
                      command=self.restore_database).pack(pady=5, fill=tk.X)
            ttk.Button(btn_frame, text="Clear All Data", 
                      command=self.clear_all_data).pack(pady=5, fill=tk.X)
            ttk.Button(btn_frame, text="Optimize Database", 
                      command=self.optimize_database).pack(pady=5, fill=tk.X)
            
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open management window: {str(e)}")

    def cleanup_database(self):
        """Cleanup old data from database"""
        try:
            if not IMPORT_SUCCESS:
                return
            
            response = messagebox.askyesno("Confirm", "Delete data older than 90 days?")
            if not response:
                return
            
            conn = get_connection()
            if not conn:
                messagebox.showerror("Error", "Cannot connect to database")
                return
            
            cursor = conn.cursor()
            
            # Delete old rainfall data
            cursor.execute("""
                DELETE FROM rainfall_data 
                WHERE created_at < DATE_SUB(NOW(), INTERVAL 90 DAY)
            """)
            rainfall_deleted = cursor.rowcount
            
            # Delete old river data
            cursor.execute("""
                DELETE FROM river_level_data 
                WHERE created_at < DATE_SUB(NOW(), INTERVAL 90 DAY)
            """)
            river_deleted = cursor.rowcount
            
            conn.commit()
            cursor.close()
            close_connection(conn)
            
            messagebox.showinfo("Success", 
                              f"Deleted {rainfall_deleted} rainfall records\n"
                              f"Deleted {river_deleted} river records")
            
        except Exception as e:
            messagebox.showerror("Error", f"Cleanup failed: {str(e)}")

    def backup_database(self):
        """Backup database"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".sql",
                filetypes=[("SQL files", "*.sql"), ("All files", "*.*")]
            )
            if filename:
                messagebox.showinfo("Info", "Backup feature not implemented yet")
        except Exception as e:
            messagebox.showerror("Error", f"Backup failed: {str(e)}")

    def restore_database(self):
        """Restore database"""
        try:
            filename = filedialog.askopenfilename(
                filetypes=[("SQL files", "*.sql"), ("All files", "*.*")]
            )
            if filename:
                messagebox.showinfo("Info", "Restore feature not implemented yet")
        except Exception as e:
            messagebox.showerror("Error", f"Restore failed: {str(e)}")

    def clear_all_data(self):
        """Clear all data from database"""
        try:
            response = messagebox.askyesnocancel("Warning", 
                "This will delete ALL data from database. Are you absolutely sure?")
            if not response:
                return
            
            if not IMPORT_SUCCESS:
                return
            
            conn = get_connection()
            if not conn:
                return
            
            cursor = conn.cursor()
            cursor.execute("TRUNCATE TABLE rainfall_data")
            cursor.execute("TRUNCATE TABLE river_level_data")
            cursor.execute("TRUNCATE TABLE flood_predictions")
            conn.commit()
            
            cursor.close()
            close_connection(conn)
            
            messagebox.showinfo("Success", "All data cleared")
            self.refresh_all_data()
            
        except Exception as e:
            messagebox.showerror("Error", f"Clear data failed: {str(e)}")

    def optimize_database(self):
        """Optimize database tables"""
        try:
            if not IMPORT_SUCCESS:
                return
            
            conn = get_connection()
            if not conn:
                return
            
            cursor = conn.cursor()
            cursor.execute("OPTIMIZE TABLE rainfall_data, river_level_data, flood_predictions")
            conn.commit()
            
            cursor.close()
            close_connection(conn)
            
            messagebox.showinfo("Success", "Database optimized")
            
        except Exception as e:
            messagebox.showerror("Error", f"Optimization failed: {str(e)}")

    # Reports methods
    def generate_reports(self):
        """Generate analysis reports"""
        try:
            if not IMPORT_SUCCESS:
                return
            
            self.update_status("Generating reports...", True)
            
            # Get time range
            start_date = self.report_start_date.get_date()
            end_date = self.report_end_date.get_date()
            
            conn = get_connection()
            if not conn:
                messagebox.showerror("Error", "Cannot connect to database")
                return
            
            cursor = conn.cursor()
            
            # Clear previous plots
            for ax in self.reports_axes.flat:
                ax.clear()
            
            # Chart 1: Daily average rainfall
            cursor.execute(f"""
                SELECT DATE(created_at) as date, 
                       AVG(CAST(JSON_EXTRACT(precipitation, '$.rainfall_1h') AS DECIMAL(10,2))) as avg_rainfall
                FROM rainfall_data 
                WHERE created_at BETWEEN '{start_date}' AND '{end_date}'
                GROUP BY DATE(created_at)
                ORDER BY date
            """)
            rainfall_data = cursor.fetchall()
            
            if rainfall_data:
                dates = [row[0] for row in rainfall_data]
                rainfall = [float(row[1]) if row[1] else 0 for row in rainfall_data]
                self.reports_axes[0, 0].plot(dates, rainfall, marker='o')
                self.reports_axes[0, 0].set_title('Daily Average Rainfall')
                self.reports_axes[0, 0].set_ylabel('Rainfall (mm)')
                self.reports_axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Chart 2: Average water level by location
            cursor.execute(f"""
                SELECT location_name, AVG(water_level) as avg_level
                FROM river_data 
                WHERE created_at BETWEEN '{start_date}' AND '{end_date}'
                GROUP BY location_name
                ORDER BY avg_level DESC
            """)
            level_data = cursor.fetchall()
            
            
            if level_data:
                locations = [row[0] for row in level_data]
                levels = [float(row[1]) if row[1] else 0 for row in level_data]
                self.reports_axes[0, 1].bar(locations, levels)
                self.reports_axes[0, 1].set_title('Average Water Level by Location')
                self.reports_axes[0, 1].set_ylabel('Water Level (cm)')
                self.reports_axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Chart 3: Flood risk distribution
            cursor.execute(f"""
                SELECT risk_level, COUNT(*) as count
                FROM predictions 
                WHERE prediction_time BETWEEN '{start_date}' AND '{end_date}'
                GROUP BY risk_level
            """)
            risk_data = cursor.fetchall()
            
            if risk_data:
                risk_levels = [row[0] for row in risk_data]
                counts = [row[1] for row in risk_data]
                colors = {'Low': 'green', 'Medium': 'yellow', 'High': 'orange', 'Critical': 'red'}
                bar_colors = [colors.get(level, 'gray') for level in risk_levels]
                self.reports_axes[1, 0].bar(risk_levels, counts, color=bar_colors)
                self.reports_axes[1, 0].set_title('Flood Risk Distribution')
                self.reports_axes[1, 0].set_ylabel('Number of Predictions')
            
            # Chart 4: Correlation between rainfall and water level
            cursor.execute(f"""
                SELECT 
                    r.location_name,
                    CAST(JSON_EXTRACT(rf.precipitation, '$.rainfall_1h') AS DECIMAL(10,2)) as rainfall,
                    r.water_level
                FROM river_data r
                LEFT JOIN rainfall_data rf ON r.location_name = rf.location_name 
                    AND DATE(r.created_at) = DATE(rf.created_at)
                WHERE r.created_at BETWEEN '{start_date}' AND '{end_date}'
                    AND JSON_EXTRACT(rf.precipitation, '$.rainfall_1h') IS NOT NULL
                LIMIT 100
            """)
            correlation_data = cursor.fetchall()
            
            if correlation_data:
                rainfall_vals = [float(row[1]) if row[1] else 0 for row in correlation_data]
                water_vals = [float(row[2]) if row[2] else 0 for row in correlation_data]
                self.reports_axes[1, 1].scatter(rainfall_vals, water_vals, alpha=0.5)
                self.reports_axes[1, 1].set_title('Rainfall vs Water Level')
                self.reports_axes[1, 1].set_xlabel('Rainfall (mm)')
                self.reports_axes[1, 1].set_ylabel('Water Level (cm)')
            
            cursor.close()
            close_connection(conn)
            
            # Update display
            self.reports_fig.tight_layout()
            self.reports_canvas.draw()
            
            self.update_status("Reports generated successfully")
            
        except Exception as e:
            self.update_status("Report generation failed")
            messagebox.showerror("Error", f"Report generation failed: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def export_report(self):
        """Export report to PDF"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
            )
            if filename:
                messagebox.showinfo("Info", "PDF export not implemented yet")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {str(e)}")

    def export_to_excel(self):
        """Export data to Excel"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
            )
            if filename:
                if not IMPORT_SUCCESS:
                    return
                
                conn = get_connection()
                if not conn:
                    return
                
                # Create Excel writer
                with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                    # Export rainfall data
                    rainfall_df = pd.read_sql("SELECT * FROM rainfall_data ORDER BY created_at DESC LIMIT 1000", conn)
                    rainfall_df.to_excel(writer, sheet_name='Rainfall Data', index=False)
                    
                    # Export river data
                    river_df = pd.read_sql("SELECT * FROM river_level_data ORDER BY created_at DESC LIMIT 1000", conn)
                    river_df.to_excel(writer, sheet_name='River Data', index=False)
                    
                    # Export predictions
                    pred_df = pd.read_sql("SELECT * FROM flood_predictions ORDER BY prediction_time DESC LIMIT 1000", conn)
                    pred_df.to_excel(writer, sheet_name='Predictions', index=False)
                
                close_connection(conn)
                messagebox.showinfo("Success", f"Data exported to {filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Excel export failed: {str(e)}")

    # Settings methods
    def test_db_connection(self):
        """Test database connection with current settings"""
        try:
            messagebox.showinfo("Info", "Test connection feature uses current settings in setup_db.py")
            self.check_database_connection()
        except Exception as e:
            messagebox.showerror("Error", f"Connection test failed: {str(e)}")

    def save_db_settings(self):
        """Save database settings"""
        try:
            messagebox.showinfo("Info", "Please update settings directly in setup_db.py file")
        except Exception as e:
            messagebox.showerror("Error", f"Save settings failed: {str(e)}")

    def test_api_key(self):
        """Test API key"""
        try:
            messagebox.showinfo("Info", "API test not implemented yet")
        except Exception as e:
            messagebox.showerror("Error", f"API test failed: {str(e)}")

    def save_api_key(self):
        """Save API key"""
        try:
            api_key = self.api_key_var.get()
            if api_key:
                # Save to config file
                config = {'api_key': api_key}
                with open('config.json', 'w') as f:
                    json.dump(config, f)
                messagebox.showinfo("Success", "API key saved")
            else:
                messagebox.showwarning("Warning", "Please enter API key")
        except Exception as e:
            messagebox.showerror("Error", f"Save API key failed: {str(e)}")

       # Help methods
    def show_help(self):
        """Show user guide"""
        help_text = """
FLOOD PREDICTION SYSTEM - USER GUIDE

1. SETUP
   - Click File > Setup Database to initialize database
   - Configure database settings in Settings tab
   
2. DATA COLLECTION
   - Use Data menu to crawl weather and river data
   - View collected data in Data tab
   
3. MODEL TRAINING
   - Click Model > Train Model to train prediction model
   - Model uses collected data for training
   
4. MAKING PREDICTIONS
   - Go to Prediction tab
   - Select location and enter/adjust parameters
   - Click "PREDICT FLOOD" button
   - View results and recommendations
   
5. REPORTS
   - Go to Reports tab
   - Select time range
   - Click "Generate Report" to view analysis
   - Export data to Excel if needed

For more help, contact: support@floodprediction.com
        """
        
        help_window = tk.Toplevel(self.root)
        help_window.title("User Guide")
        help_window.geometry("600x500")
        
        text_widget = tk.Text(help_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert(tk.END, help_text)
        text_widget.config(state=tk.DISABLED)
        
        scrollbar = ttk.Scrollbar(text_widget, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def show_about(self):
        """Show about dialog"""
        about_text = """
FLOOD PREDICTION SYSTEM
Version 1.0

An AI-powered flood prediction system using:
- Machine Learning (Random Forest)
- Real-time weather data
- River level monitoring
- Multi-level risk assessment

Developed by: MSA30DN Team
Year: 2024

© 2024 All Rights Reserved
        """
        messagebox.showinfo("About Software", about_text)


def main():
    """Main function"""
    root = tk.Tk()
    app = FloodPredictionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
