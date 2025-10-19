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

# Import project modules
try:
    from setup_db import get_connection, close_connection
    from predictor import (load_combined_data, load_data_from_db, train_model, 
                          predict_flood_risk, create_flood_labels, 
                          generate_advanced_training_data)
except ImportError as e:
    print(f"Import error: {e}")

class FloodPredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Flood Prediction System")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Storage variables
        self.model = None
        self.features = None
        self.is_advanced = False
        self.current_data = None
        
        # Create interface
        self.setup_styles()
        self.create_menu()
        self.create_main_interface()
        self.create_status_bar()
        
        # Initial database connection check
        self.check_database_connection()

    def setup_styles(self):
        """Set up styles for the interface"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main colors
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), foreground='#2c3e50')
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'), foreground='#34495e')
        style.configure('Success.TLabel', foreground='#27ae60', font=('Arial', 10, 'bold'))
        style.configure('Warning.TLabel', foreground='#f39c12', font=('Arial', 10, 'bold'))
        style.configure('Error.TLabel', foreground='#e74c3c', font=('Arial', 10, 'bold'))
        style.configure('Accent.TButton', foreground='white', font=('Arial', 10, 'bold'))

    def create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Setup Database", command=self.setup_database)
        file_menu.add_command(label="Export Report", command=self.export_report)
        # Removed: file_menu.add_command(label="Exit", command=self.root.quit)
        
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

    def create_main_interface(self):
        """Create main interface"""
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

    def create_dashboard_tab(self):
        """Dashboard Tab - System Overview"""
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
        
        # Refresh button - Di chuyển lên góc phải trên cùng
        ttk.Button(left_frame, text="Refresh", command=self.refresh_dashboard).pack(anchor='ne', padx=10, pady=10)
        
        # Right frame - Charts
        right_frame = ttk.LabelFrame(dashboard_frame, text="Overview Charts", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Matplotlib figure
        self.dashboard_fig, self.dashboard_axes = plt.subplots(2, 2, figsize=(8, 6))
        self.dashboard_fig.suptitle("System Data Statistics")
        
        self.dashboard_canvas = FigureCanvasTkAgg(self.dashboard_fig, right_frame)
        self.dashboard_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_prediction_tab(self):
        """Prediction Tab - Perform flood prediction"""
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
        
        # Rainfall
        ttk.Label(left_input, text="Rainfall 1h (mm):").pack(anchor='w', pady=(10,0))
        self.rainfall_1h_var = tk.DoubleVar(value=0.0)
        rainfall_frame = ttk.Frame(left_input)
        rainfall_frame.pack(fill=tk.X, pady=2)
        rainfall_scale = tk.Scale(rainfall_frame, from_=0, to=100, resolution=0.1, 
                                 orient=tk.HORIZONTAL, variable=self.rainfall_1h_var)
        rainfall_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.rainfall_value_label = ttk.Label(rainfall_frame, text="0.0mm", width=8)
        self.rainfall_value_label.pack(side=tk.RIGHT)
        rainfall_scale.config(command=lambda v: self.rainfall_value_label.config(text=f"{float(v):.1f}mm"))
        
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

    def on_location_selected(self, event):
        """Load latest river data for selected location and update UI"""
        location = self.location_var.get()
        
        try:
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

    def create_rainfall_data_subtab(self):
        """Create Rainfall Data sub-tab"""
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

    def create_river_data_subtab(self):
        """Create River Level Data sub-tab"""
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

    def create_predictions_data_subtab(self):
        """Create Predictions Data sub-tab"""
        predictions_frame = ttk.Frame(self.data_notebook)
        self.data_notebook.add(predictions_frame, text="Predictions")
        
        # Control frame for predictions
        predictions_control = ttk.Frame(predictions_frame)
        predictions_control.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(predictions_control, text="Refresh Predictions", 
                  command=self.refresh_predictions_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(predictions_control, text="Clear Old Predictions", 
                  command=self.clear_old_predictions).pack(side=tk.LEFT, padx=5)
        
        # Treeview frame
        tree_frame = ttk.Frame(predictions_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Treeview for predictions data
        columns = ['Location', 'Time', 'Risk Level', 'Probability', 'Weather Factor', 
                  'River Factor', 'Combined Score', 'Rainfall 1h', 'Water Level', 'Recommendations']
        self.predictions_tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=20)
        
        # Set column headings and widths
        column_widths = {'Location': 100, 'Time': 150, 'Risk Level': 80, 'Probability': 80, 
                        'Weather Factor': 100, 'River Factor': 100, 'Combined Score': 100, 
                        'Rainfall 1h': 100, 'Water Level': 100, 'Recommendations': 200}
        
        for col in columns:
            self.predictions_tree.heading(col, text=col)
            self.predictions_tree.column(col, width=column_widths.get(col, 100))
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.predictions_tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.predictions_tree.xview)
        self.predictions_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack treeview and scrollbars
        self.predictions_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_reports_tab(self):
        """Reports Tab - Statistics and charts"""
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

    def create_settings_tab(self):
        """Settings Tab - System configuration"""
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
        
    def create_status_bar(self):
        """Create status bar"""
        self.status_bar = ttk.Frame(self.root, relief=tk.SUNKEN, borderwidth=1)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = ttk.Label(self.status_bar, text="Ready", relief=tk.SUNKEN)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=2)
        
        self.progress_bar = ttk.Progressbar(self.status_bar, length=200, mode='indeterminate')
        self.progress_bar.pack(side=tk.RIGHT, padx=5, pady=2)
        
        # Add Exit button to status bar
        ttk.Button(self.status_bar, text="Exit", command=self.root.quit).pack(side=tk.RIGHT, padx=5, pady=2)

    def update_status(self, message, show_progress=False):
        """Update status bar"""
        self.status_label.config(text=message)
        if show_progress:
            self.progress_bar.start()
        else:
            self.progress_bar.stop()
        self.root.update_idletasks()

    def check_database_connection(self):
        """Check database connection"""
        try:
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
        self.update_status("Refreshing dashboard...", True)
        
        try:
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
                # Add description
                self.dashboard_axes[0,0].text(0.02, 0.98, 'Shows recent temperature changes\nover time (blue line)', 
                                             transform=self.dashboard_axes[0,0].transAxes, fontsize=8, 
                                             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Chart 2: Rainfall distribution (Top-Right)
            if 'rainfall_1h' in df.columns:
                rainfall_data = df['rainfall_1h'].values
                rainfall_data = rainfall_data[rainfall_data >= 0]  # Remove negative values
                self.dashboard_axes[0,1].hist(rainfall_data, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
                self.dashboard_axes[0,1].set_title('Rainfall Distribution')
                self.dashboard_axes[0,1].set_xlabel('mm/h')
                self.dashboard_axes[0,1].set_ylabel('Frequency')
                # Add description
                self.dashboard_axes[0,1].text(0.02, 0.98, 'Shows how often different\nrainfall amounts occur\n(blue bars)', 
                                             transform=self.dashboard_axes[0,1].transAxes, fontsize=8, 
                                             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            
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
                # Add description
                self.dashboard_axes[1,0].text(0.02, 0.02, 'Shows percentage of flood risk levels\n(Green=Low, Orange=Moderate, Red=High)', 
                                             transform=self.dashboard_axes[1,0].transAxes, fontsize=8, 
                                             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
            elif 'flood_risk' in df.columns:
                risk_counts = df['flood_risk'].value_counts()
                labels = ['No Flood', 'Flood']
                colors = ['green', 'red']
                if len(risk_counts) > 0:
                    self.dashboard_axes[1,0].pie(risk_counts.values, labels=labels, 
                                                colors=colors, autopct='%1.1f%%', startangle=90)
                    self.dashboard_axes[1,0].set_title('Flood Risk Distribution')
                # Add description
                self.dashboard_axes[1,0].text(0.02, 0.02, 'Shows percentage of flood vs no-flood\n(Green=No Flood, Red=Flood)', 
                                             transform=self.dashboard_axes[1,0].transAxes, fontsize=8, 
                                             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
            
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
                    # Add description
                    self.dashboard_axes[1,1].text(0.02, 0.98, 'Shows recent humidity changes\n(green line)', 
                                                 transform=self.dashboard_axes[1,1].transAxes, fontsize=8, 
                                                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
            
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

    def perform_prediction(self):
        """Perform flood prediction"""
        if self.model is None:
            messagebox.showwarning("Warning", "No model available. Please train a model first!")
            return
        
        try:
            # Collect input data
            input_data = {
                'temperature': self.temp_var.get(),
                'humidity': self.humidity_var.get(),
                'pressure': self.pressure_var.get(),
                'rainfall_1h': self.rainfall_1h_var.get(),
                'rainfall_3h': self.rainfall_1h_var.get() * 2.5,  # Approximation
                'wind_speed': self.wind_var.get(),
            }
            
            # Add river data if using advanced model
            if self.is_advanced:
                water_level = self.water_level_var.get()
                normal_level = 150.0  # Default normal level
                
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
                # Save prediction to database
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
                print("Cannot connect to database for saving prediction")
                return
            
            cursor = conn.cursor()
            
            # Prepare data for insertion
            location_name = self.location_var.get()
            risk_level = result['risk_level'] if self.is_advanced else ('HIGH' if result['probability_flood'] > 0.7 else 'MODERATE' if result['probability_flood'] > 0.4 else 'LOW')
            probability = result['probabilities']['HIGH'] if self.is_advanced else result['probability_flood']
            confidence = result['confidence']
            
            # Calculate factors (simplified)
            weather_factor = (input_data['rainfall_1h'] / 50.0 + input_data['humidity'] / 100.0) / 2.0
            river_factor = (input_data.get('water_level', 150) / 300.0) if self.is_advanced else 0.0
            combined_score = (weather_factor + river_factor) / 2.0
            
            alert_exceeded = input_data.get('alert_level_exceeded', 0) if self.is_advanced else 0
            
            # Generate recommendations
            if risk_level == 'HIGH':
                recommendations = "Immediate evacuation required. Activate emergency response."
            elif risk_level == 'MODERATE':
                recommendations = "Monitor closely. Prepare evacuation if conditions worsen."
            else:
                recommendations = "Continue normal monitoring. No immediate action required."
            
            # Insert into database
            query = """
            INSERT INTO flood_predictions 
            (location_name, risk_level, probability, weather_factor, river_factor, 
             combined_score, rainfall_1h, rainfall_3h, water_level, alert_level_exceeded, 
             recommendations, model_version)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(query, (
                location_name, risk_level, probability, weather_factor, river_factor,
                combined_score, input_data['rainfall_1h'], input_data['rainfall_3h'], 
                input_data.get('water_level'), alert_exceeded, recommendations, 'v1.0'
            ))
            
            conn.commit()
            cursor.close()
            close_connection(conn)
            
            print(f"Prediction saved for {location_name}: {risk_level}")
            
        except Exception as e:
            print(f"Error saving prediction to database: {str(e)}")

    def refresh_all_data(self):
        """Refresh both rainfall and river data"""
        self.refresh_rainfall_data()
        self.refresh_river_data()
        self.refresh_predictions_data()

    def refresh_rainfall_data(self):
        """Refresh rainfall data display"""
        try:
            self.update_status("Loading rainfall data...", True)
            
            # Clear treeview
            for item in self.rainfall_tree.get_children():
                self.rainfall_tree.delete(item)
            
            # Load rainfall data
            conn = get_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT location_name, created_at, 
                           JSON_EXTRACT(precipitation, '$.temperature') as temperature,
                           JSON_EXTRACT(precipitation, '$.humidity') as humidity,
                           JSON_EXTRACT(precipitation, '$.rainfall_1h') as rainfall_1h,
                           JSON_EXTRACT(precipitation, '$.rainfall_3h') as rainfall_3h,
                           JSON_EXTRACT(precipitation, '$.wind_speed') as wind_speed
                    FROM rainfall_data 
                    ORDER BY created_at DESC LIMIT 100
                """)
                rows = cursor.fetchall()
                cursor.close()
                close_connection(conn)
                
                for row in rows:
                    values = [
                        row[0],  # location_name
                        str(row[1])[:19] if row[1] else 'N/A',  # created_at
                        f"{float(row[2]):.1f}°C" if row[2] and row[2] != 'null' else 'N/A',  # temperature
                        f"{float(row[3]):.0f}%" if row[3] and row[3] != 'null' else 'N/A',  # humidity
                        f"{float(row[4]):.1f}mm" if row[4] and row[4] != 'null' else 'N/A',  # rainfall_1h
                        f"{float(row[5]):.1f}mm" if row[5] and row[5] != 'null' else 'N/A',  # rainfall_3h
                        f"{float(row[6]):.1f}km/h" if row[6] and row[6] != 'null' else 'N/A'  # wind_speed
                    ]
                    self.rainfall_tree.insert('', 'end', values=values)
            
            self.update_status("Loaded rainfall data")
            
        except Exception as e:
            self.update_status("Error loading rainfall data")
            messagebox.showerror("Error", f"Error refreshing rainfall data: {str(e)}")

    def refresh_river_data(self):
        """Refresh river level data display"""
        try:
            self.update_status("Loading river data...", True)
            
            # Clear treeview
            for item in self.river_tree.get_children():
                self.river_tree.delete(item)
            
            # Load river data
            conn = get_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT location_name, created_at, water_level, flow_rate, trend 
                    FROM river_level_data 
                    ORDER BY created_at DESC LIMIT 100
                """)
                rows = cursor.fetchall()
                cursor.close()
                close_connection(conn)
                
                for row in rows:
                    values = [
                        row[0],  # location_name
                        str(row[1])[:19] if row[1] else 'N/A',  # created_at
                        f"{float(row[2]):.0f}cm" if row[2] else 'N/A',  # water_level
                        f"{float(row[3]):.0f}m³/s" if row[3] else 'N/A',  # flow_rate
                        row[4] if row[4] else 'N/A'  # trend
                    ]
                    self.river_tree.insert('', 'end', values=values)
            
            self.update_status("Loaded river data")
            
        except Exception as e:
            self.update_status("Error loading river data")
            messagebox.showerror("Error", f"Error refreshing river data: {str(e)}")

    def refresh_predictions_data(self):
        """Refresh predictions data display"""
        try:
            self.update_status("Loading predictions data...", True)
            
            # Clear treeview
            for item in self.predictions_tree.get_children():
                self.predictions_tree.delete(item)
            
            # Load predictions data
            conn = get_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT location_name, prediction_time, risk_level, probability, 
                           weather_factor, river_factor, combined_score, rainfall_1h, 
                           water_level, recommendations 
                    FROM flood_predictions 
                    ORDER BY prediction_time DESC LIMIT 100
                """)
                rows = cursor.fetchall()
                cursor.close()
                close_connection(conn)
                
                for row in rows:
                    values = [
                        row[0],  # location_name
                        str(row[1])[:19] if row[1] else 'N/A',  # prediction_time
                        row[2],  # risk_level
                        f"{float(row[3]):.1%}" if row[3] else 'N/A',  # probability
                        f"{float(row[4]):.3f}" if row[4] else 'N/A',  # weather_factor
                        f"{float(row[5]):.3f}" if row[5] else 'N/A',  # river_factor
                        f"{float(row[6]):.3f}" if row[6] else 'N/A',  # combined_score
                        f"{float(row[7]):.1f}mm" if row[7] else 'N/A',  # rainfall_1h
                        f"{float(row[8]):.0f}cm" if row[8] else 'N/A',  # water_level
                        row[9] if row[9] else 'N/A'  # recommendations
                    ]
                    self.predictions_tree.insert('', 'end', values=values)
            
            self.update_status("Loaded predictions data")
            
        except Exception as e:
            self.update_status("Error loading predictions data")
            messagebox.showerror("Error", f"Error refreshing predictions data: {str(e)}")

    def clear_old_predictions(self):
        """Clear old predictions (keep only last 500)"""
        if messagebox.askyesno("Confirmation", "Are you sure you want to clear old predictions?\nThis will keep only the 500 most recent predictions."):
            try:
                self.update_status("Clearing old predictions...", True)
                
                conn = get_connection()
                if conn:
                    cursor = conn.cursor()
                    
                    # Keep only last 500 predictions
                    cursor.execute("""
                        DELETE FROM flood_predictions 
                        WHERE id NOT IN (
                            SELECT id FROM (
                                SELECT id FROM flood_predictions 
                                ORDER BY prediction_time DESC LIMIT 500
                            ) AS temp
                        )
                    """)
                    
                    deleted_count = cursor.rowcount
                    conn.commit()
                    cursor.close()
                    close_connection(conn)
                    
                    self.update_status(f"Cleared {deleted_count} old predictions")
                    messagebox.showinfo("Success", f"Cleared {deleted_count} old predictions!")
                    self.refresh_predictions_data()
                else:
                    messagebox.showerror("Error", "Cannot connect to database")
                    
            except Exception as e:
                self.update_status("Error clearing predictions")
                messagebox.showerror("Error", f"Error clearing predictions: {str(e)}")

    # Placeholder methods for other functionalities
    def setup_database(self):
        messagebox.showinfo("Info", "Database setup functionality not implemented yet.")

    def export_report(self):
        messagebox.showinfo("Info", "Export report functionality not implemented yet.")

    def crawl_weather_data(self):
        messagebox.showinfo("Info", "Crawl weather data functionality not implemented yet.")

    def crawl_river_data(self):
        messagebox.showinfo("Info", "Crawl river data functionality not implemented yet.")

    def manage_database(self):
        messagebox.showinfo("Info", "Manage database functionality not implemented yet.")

    def train_prediction_model(self):
        messagebox.showinfo("Info", "Train model functionality not implemented yet.")

    def evaluate_model(self):
        messagebox.showinfo("Info", "Evaluate model functionality not implemented yet.")

    def show_help(self):
        messagebox.showinfo("Help", "Help functionality not implemented yet.")

    def show_about(self):
        messagebox.showinfo("About", "About functionality not implemented yet.")

    def run_auto_system(self):
        messagebox.showinfo("Info", "Auto system functionality not implemented yet.")

    def generate_reports(self):
        messagebox.showinfo("Info", "Generate reports functionality not implemented yet.")

    def export_to_excel(self):
        messagebox.showinfo("Info", "Export to Excel functionality not implemented yet.")

    def cleanup_database(self):
        messagebox.showinfo("Info", "Cleanup database functionality not implemented yet.")

    def test_db_connection(self):
        messagebox.showinfo("Info", "Test DB connection functionality not implemented yet.")

    def save_db_settings(self):
        messagebox.showinfo("Info", "Save DB settings functionality not implemented yet.")

    def test_api_key(self):
        messagebox.showinfo("Info", "Test API key functionality not implemented yet.")

    def save_api_key(self):
        messagebox.showinfo("Info", "Save API key functionality not implemented yet.")

    def display_prediction_result(self, result, input_data):
        """Display prediction result in the text area"""
        self.result_text.delete(1.0, tk.END)
        
        result_text = f"Flood Prediction Result\n{'='*30}\n\n"
        result_text += f"Risk Level: {result['risk_level']}\n"
        result_text += f"Probability: {result['probabilities']['HIGH']:.1%}\n"
        result_text += f"Confidence: {result['confidence']:.1%}\n\n"
        
        result_text += "Input Data:\n"
        for key, value in input_data.items():
            result_text += f"  {key}: {value}\n"
        
        self.result_text.insert(tk.END, result_text)

    def calculate_alert_level(self, water_level):
        """Calculate alert level based on water level"""
        if water_level >= 270:
            return 3
        elif water_level >= 220:
            return 2
        elif water_level >= 180:
            return 1
        else:
            return 0

def main():
    root = tk.Tk()
    app = FloodPredictionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()