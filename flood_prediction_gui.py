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
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
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
        self.data_summary_text.pack(fill=tk.BOTH, expand=True, pady=(5,10))
        
        # Buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, text="Refresh", command=self.refresh_dashboard).pack(side=tk.LEFT, padx=(0,5))
        ttk.Button(button_frame, text="Run Auto", command=self.run_auto_system).pack(side=tk.LEFT, padx=5)
        
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

    def create_data_tab(self):
        """Data Tab - Manage and view data"""
        data_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_frame, text="Data")
        
        # Control panel
        control_frame = ttk.LabelFrame(data_frame, text="Data Control", padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Buttons row 1
        btn_frame1 = ttk.Frame(control_frame)
        btn_frame1.pack(fill=tk.X, pady=2)
        
        ttk.Button(btn_frame1, text="Crawl Weather", 
                  command=self.crawl_weather_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame1, text="Crawl River", 
                  command=self.crawl_river_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame1, text="Refresh Data", 
                  command=self.refresh_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame1, text="Cleanup DB", 
                  command=self.cleanup_database).pack(side=tk.LEFT, padx=5)
        
        # Data display
        data_display_frame = ttk.LabelFrame(data_frame, text="Current Data", padding=10)
        data_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Treeview frame
        tree_frame = ttk.Frame(data_display_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        # Treeview for data display
        columns = ['Location', 'Time', 'Temperature', 'Humidity', 'Rainfall', 'Water_Level', 'Risk']
        self.data_tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=20)
        
        # Set column headings and widths
        column_widths = {'Location': 120, 'Time': 150, 'Temperature': 100, 'Humidity': 80, 
                        'Rainfall': 80, 'Water_Level': 100, 'Risk': 80}
        
        for col in columns:
            self.data_tree.heading(col, text=col.replace('_', ' '))
            self.data_tree.column(col, width=column_widths.get(col, 100))
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.data_tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.data_tree.xview)
        self.data_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack treeview and scrollbars
        self.data_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
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
        ttk.Scale(depth_frame, from_=5, to=20, orient=tk.HORIZONTAL,
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
            
            # Chart 1: Temperature trend
            if len(df) > 0 and 'temperature' in df.columns:
                temps = df['temperature'].tail(20).values
                self.dashboard_axes[0,0].plot(temps, 'b-o', markersize=3)
                self.dashboard_axes[0,0].set_title('Temperature Trend (Last 20 Samples)')
                self.dashboard_axes[0,0].set_ylabel('°C')
                self.dashboard_axes[0,0].grid(True, alpha=0.3)
            
            # Chart 2: Rainfall distribution
            if 'rainfall_1h' in df.columns:
                rainfall_data = df['rainfall_1h'].values
                rainfall_data = rainfall_data[rainfall_data >= 0]  # Remove negative values
                self.dashboard_axes[0,1].hist(rainfall_data, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
                self.dashboard_axes[0,1].set_title('Rainfall Distribution')
                self.dashboard_axes[0,1].set_xlabel('mm/h')
                self.dashboard_axes[0,1].set_ylabel('Frequency')
            
            # Chart 3: Risk levels (if available)
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
            
            # Chart 4: Water level trend (if available)
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
                    
                    self.dashboard_axes[1,1].axhline(y=alert1, color='yellow', linestyle='--', alpha=0.7, label='Alert 1')
                    self.dashboard_axes[1,1].axhline(y=alert2, color='orange', linestyle='--', alpha=0.7, label='Alert 2')
                    self.dashboard_axes[1,1].axhline(y=alert3, color='red', linestyle='--', alpha=0.7, label='Alert 3')
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
            else:
                messagebox.showerror("Error", "Unable to perform prediction!")
                
        except Exception as e:
            messagebox.showerror("Error", f"Prediction error: {str(e)}")
            import traceback
            print(traceback.format_exc())

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

    def display_prediction_result(self, result, input_data):
        """Display prediction result"""
        # Clear previous results
        self.result_text.delete(1.0, tk.END)
        
        # Clear risk display frame
        for widget in self.risk_display_frame.winfo_children():
            widget.destroy()
        
        # Display text result
        result_text = f"""{'='*50}
FLOOD PREDICTION RESULT
{'='*50}

Location: {self.location_var.get()}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: {'Advanced (3 Levels)' if self.is_advanced else 'Basic (2 Levels)'}

{'='*50}
INPUT DATA:
{'='*50}
- Temperature: {input_data['temperature']:.1f}°C
- Humidity: {input_data['humidity']:.0f}%
- Pressure: {input_data['pressure']:.0f} hPa
- Rainfall 1h: {input_data['rainfall_1h']:.1f} mm
- Rainfall 3h: {input_data['rainfall_3h']:.1f} mm
- Wind Speed: {input_data['wind_speed']:.0f} km/h
"""
        
        if self.is_advanced and 'water_level' in input_data:
            result_text += f"""- Water Level: {input_data['water_level']:.0f} cm
- Flow Rate: {input_data.get('flow_rate', 0):.0f} m³/s
- Trend: {self.trend_var.get()}
- Alert Level: {input_data.get('alert_level_exceeded', 0)}
"""
        
        result_text += f"""\n{'='*50}
PREDICTION RESULT:
{'='*50}
"""
        
        if self.is_advanced:
            risk_level = result['risk_level']
            confidence = result['confidence']
            
            result_text += f"""Risk Level: {risk_level}
Confidence: {confidence:.1%}

Probability per Level:
"""
            
            for level, prob in result['probabilities'].items():
                result_text += f"  {level}: {prob:.1%}\n"
            
            # Create visual risk display
            self.create_risk_display(risk_level, confidence, result['probabilities'])
            
        else:
            flood_prob = result['probability_flood']
            confidence = result['confidence']
            
            result_text += f"""Flood Probability: {flood_prob:.1%}
Confidence: {confidence:.1%}
"""
            
            # Determine risk level for basic model
            if flood_prob < 0.3:
                risk_level = "LOW"
                color = "green"
            elif flood_prob < 0.7:
                risk_level = "MODERATE"
                color = "orange"
            else:
                risk_level = "HIGH"
                color = "red"
            
            result_text += f"Risk Level: {risk_level}\n"
            
            # Create simple risk display for basic model
            self.create_simple_risk_display(risk_level, flood_prob, color)
        
        # Add recommendations
        result_text += f"""\n{'='*50}
RECOMMENDATIONS:
{'='*50}
"""
        if self.is_advanced:
            if result['risk_level'] == 'HIGH':
                result_text += """HIGH RISK - Emergency response required!
- Evacuate residents in danger zones
- Prepare emergency relief supplies
- Continuously monitor water levels
- Activate emergency response team
- Issue warnings to all areas
"""
            elif result['risk_level'] == 'MODERATE':
                result_text += """MODERATE RISK - Closely monitor
- Prepare response measures
- Notify residents in low-lying areas
- Check drainage systems
- Continuously monitor weather forecasts
"""
            else:
                result_text += """LOW RISK - Continue monitoring
- Maintain normal operations
- Periodically check weather updates
- Check warning systems
"""
        else:
            if flood_prob > 0.7:
                result_text += """HIGH RISK - Take note!
- Closely monitor weather developments
- Prepare response measures
- Check drainage systems
"""
            elif flood_prob > 0.4:
                result_text += """MODERATE RISK
- Continue monitoring
- Prepare if necessary
"""
            else:
                result_text += """LOW RISK
- Normal operations
- Monitor weather updates
"""
        
        result_text += f"\n{'='*50}\nReport generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*50}"
        
        self.result_text.insert(tk.END, result_text)

    def create_risk_display(self, risk_level, confidence, probabilities):
        """Create visual risk display for advanced model"""
        # Risk level indicator
        risk_colors = {'LOW': '#27ae60', 'MODERATE': '#f39c12', 'HIGH': '#e74c3c'}
        color = risk_colors.get(risk_level, '#95a5a6')
        
        # Large risk indicator
        risk_frame = tk.Frame(self.risk_display_frame, bg=color, relief=tk.RAISED, bd=3)
        risk_frame.pack(fill=tk.X, pady=10)
        
        risk_label = tk.Label(risk_frame, text=f"RISK {risk_level}", 
                             font=('Arial', 18, 'bold'), fg='white', bg=color)
        risk_label.pack(pady=15)
        
        confidence_label = tk.Label(risk_frame, text=f"Confidence: {confidence:.1%}",
                                   font=('Arial', 12), fg='white', bg=color)
        confidence_label.pack(pady=(0, 15))
        
        # Probability bars
        prob_frame = ttk.LabelFrame(self.risk_display_frame, text="Probability per Level")
        prob_frame.pack(fill=tk.X, pady=10)
        
        colors = {'LOW': '#27ae60', 'MODERATE': '#f39c12', 'HIGH': '#e74c3c'}
        
        for level, prob in probabilities.items():
            level_frame = tk.Frame(prob_frame)
            level_frame.pack(fill=tk.X, padx=10, pady=5)
            
            # Label
            level_label = tk.Label(level_frame, text=f"{level}:", width=10, anchor='w', font=('Arial', 10, 'bold'))
            level_label.pack(side=tk.LEFT)
            
            # Progress bar frame
            bar_frame = tk.Frame(level_frame, height=20, relief=tk.SUNKEN, bd=1)
            bar_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            # Filled portion
            fill_width = int(200 * prob)
            if fill_width > 0:
                fill_frame = tk.Frame(bar_frame, bg=colors[level], height=18)
                fill_frame.place(x=0, y=0, width=fill_width, height=18)
            
            # Percentage label
            percent_label = tk.Label(level_frame, text=f"{prob:.1%}", width=8, font=('Arial', 10))
            percent_label.pack(side=tk.RIGHT)

    def create_simple_risk_display(self, risk_level, probability, color):
        """Create simple display for basic model"""
        # Risk indicator
        risk_frame = tk.Frame(self.risk_display_frame, bg=color, relief=tk.RAISED, bd=3)
        risk_frame.pack(fill=tk.X, pady=20)
        
        risk_label = tk.Label(risk_frame, text=f"RISK {risk_level}", 
                             font=('Arial', 18, 'bold'), fg='white', bg=color)
        risk_label.pack(pady=20)
        
        prob_label = tk.Label(risk_frame, text=f"Flood Probability: {probability:.1%}",
                             font=('Arial', 12), fg='white', bg=color)
        prob_label.pack(pady=(0, 20))

    def train_prediction_model(self):
        """Train prediction model"""
        self.update_status("Training model...", True)
        
        def train_in_thread():
            try:
                # Load data
                combined_df = load_combined_data()
                
                if combined_df is not None and len(combined_df) > 0:
                    print("Using combined data")
                    real_df = combined_df
                    use_advanced = True
                    real_df = create_flood_labels(real_df)
                    synthetic_df = generate_advanced_training_data(real_df)
                else:
                    print("Using basic data")
                    real_df = load_data_from_db()
                    if real_df is None:
                        real_df = pd.DataFrame()
                    use_advanced = False
                    if len(real_df) > 0:
                        real_df = create_flood_labels(real_df)
                    
                    # Generate basic synthetic data
                    synthetic_data = []
                    for i in range(100):
                        sample = {
                            'location_name': f'Sample_{i}',
                            'latitude': 10.0 + np.random.uniform(-5, 5),
                            'longitude': 106.0 + np.random.uniform(-5, 5),
                            'temperature': 26.0 + np.random.uniform(-5, 5),
                            'humidity': np.random.uniform(40, 98),
                            'pressure': 1013.0 + np.random.uniform(-20, 20),
                            'rainfall_1h': np.random.uniform(0, 50),
                            'rainfall_3h': np.random.uniform(0, 100),
                            'wind_speed': np.random.uniform(0, 50),
                            'flood_risk': np.random.choice([0, 1], p=[0.6, 0.4])
                        }
                        synthetic_data.append(sample)
                    
                    synthetic_df = pd.DataFrame(synthetic_data)
                
                # Combine data
                if len(real_df) > 0:
                    df = pd.concat([real_df, synthetic_df], ignore_index=True)
                else:
                    df = synthetic_df
                
                # Train model
                result = train_model(df)
                if len(result) == 3:
                    model, features, is_advanced = result
                else:
                    model, features = result
                    is_advanced = use_advanced
                
                # Update UI in main thread
                self.root.after(0, lambda: self.on_training_complete(model, features, is_advanced))
                
            except Exception as e:
                import traceback
                error_msg = f"{str(e)}\n{traceback.format_exc()}"
                self.root.after(0, lambda: self.on_training_error(error_msg))
        
        # Start training in background thread
        threading.Thread(target=train_in_thread, daemon=True).start()

    def on_training_complete(self, model, features, is_advanced):
        """Callback when training completes"""
        self.model = model
        self.features = features
        self.is_advanced = is_advanced
        
        self.update_status("Model training completed!")
        
        if self.model is not None:
            self.model_status_label.config(
                text=f"Trained ({'Advanced' if self.is_advanced else 'Basic'})",
                style='Success.TLabel'
            )
            messagebox.showinfo("Success", 
                              f"Model {'advanced (3 levels)' if self.is_advanced else 'basic (2 levels)'} trained successfully!")
        else:
            self.model_status_label.config(text="Training failed", style='Error.TLabel')

    def on_training_error(self, error_msg):
        """Callback when training error occurs"""
        self.update_status(f"Error training model")
        messagebox.showerror("Error", f"Training error:\n{error_msg}")

    def run_auto_system(self):
        """Run auto system"""
        def run_in_thread():
            try:
                self.root.after(0, lambda: self.update_status("Running auto system...", True))
                
                # Run crawlers
                commands = [
                    ["python", "rainfall_crawler.py"],
                    ["python", "river_level_crawler.py"]
                ]
                
                for cmd in commands:
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                        if result.returncode != 0:
                            self.root.after(0, lambda e=result.stderr, c=cmd: 
                                          messagebox.showerror("Error", f"Error running {' '.join(c)}:\n{e}"))
                            return
                    except subprocess.TimeoutExpired:
                        self.root.after(0, lambda c=cmd: 
                                      messagebox.showerror("Error", f"Timeout running {' '.join(c)}"))
                        return
                
                # Train model after getting data
                self.root.after(0, lambda: self.update_status("Training model after crawl...", True))
                self.root.after(1000, self.train_prediction_model)  # Delay to allow UI update
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Auto run error: {str(e)}"))
        
        threading.Thread(target=run_in_thread, daemon=True).start()

    # Database management methods
    def setup_database(self):
        """Set up database"""
        try:
            # Import setup function
            from setup_db import setup_database
            result = setup_database()
            if result:
                messagebox.showinfo("Success", "Database setup successful!")
                self.check_database_connection()
            else:
                messagebox.showerror("Error", "Unable to set up database!")
        except Exception as e:
            messagebox.showerror("Error", f"Database setup error: {str(e)}")

    def crawl_weather_data(self):
        """Crawl weather data"""
        self.update_status("Crawling weather data...", True)
        
        def crawl_in_thread():
            try:
                result = subprocess.run(["python", "rainfall_crawler.py"], 
                                      capture_output=True, text=True, timeout=120)
                if result.returncode == 0:
                    self.root.after(0, lambda: self.update_status("Weather crawl completed!"))
                    self.root.after(0, lambda: messagebox.showinfo("Success", "Weather data crawled!"))
                    self.root.after(0, self.refresh_dashboard)
                else:
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Weather crawl error:\n{result.stderr}"))
            except subprocess.TimeoutExpired:
                self.root.after(0, lambda: messagebox.showerror("Error", "Timeout crawling weather data"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Error: {str(e)}"))
        
        threading.Thread(target=crawl_in_thread, daemon=True).start()

    def crawl_river_data(self):
        """Crawl river water level data"""
        self.update_status("Crawling river data...", True)
        
        def crawl_in_thread():
            try:
                result = subprocess.run(["python", "river_level_crawler.py"], 
                                      capture_output=True, text=True, timeout=120)
                if result.returncode == 0:
                    self.root.after(0, lambda: self.update_status("River crawl completed!"))
                    self.root.after(0, lambda: messagebox.showinfo("Success", "River data crawled!"))
                    self.root.after(0, self.refresh_dashboard)
                else:
                    self.root.after(0, lambda: messagebox.showerror("Error", f"River crawl error:\n{result.stderr}"))
            except subprocess.TimeoutExpired:
                self.root.after(0, lambda: messagebox.showerror("Error", "Timeout crawling river data"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Error: {str(e)}"))
        
        threading.Thread(target=crawl_in_thread, daemon=True).start()

    def refresh_data(self):
        """Refresh displayed data"""
        try:
            self.update_status("Loading data...", True)
            
            # Clear treeview
            for item in self.data_tree.get_children():
                self.data_tree.delete(item)
            
            # Load and display data
            df = load_combined_data()
            if df is None or len(df) == 0:
                df = load_data_from_db()
            
            if df is not None and len(df) > 0:
                # Sort by time descending
                if 'created_at' in df.columns:
                    df = df.sort_values('created_at', ascending=False)
                
                for idx, row in df.head(100).iterrows():  # Show latest 100 records
                    values = [
                        row.get('location_name', 'N/A'),
                        str(row.get('created_at', ''))[:19] if row.get('created_at') else 'N/A',
                        f"{row.get('temperature', 0):.1f}°C",
                        f"{row.get('humidity', 0):.0f}%",
                        f"{row.get('rainfall_1h', 0):.1f}mm",
                        f"{row.get('water_level', 0):.0f}cm" if 'water_level' in row and pd.notna(row.get('water_level')) else 'N/A',
                        str(row.get('flood_risk_level', row.get('flood_risk', 'N/A')))
                    ]
                    self.data_tree.insert('', 'end', values=values)
            
            self.update_status(f"Loaded {len(df) if df is not None else 0} records")
            
        except Exception as e:
            self.update_status("Error loading data")
            messagebox.showerror("Error", f"Error refreshing data: {str(e)}")

    def cleanup_database(self):
        """Clean up database"""
        if messagebox.askyesno("Confirmation", "Are you sure you want to clean up the database?\nThis will remove old and duplicate data."):
            try:
                self.update_status("Cleaning up database...", True)
                
                conn = get_connection()
                if conn:
                    cursor = conn.cursor()
                    
                    # Remove duplicates from rainfall_data
                    cursor.execute("""
                        DELETE r1 FROM rainfall_data r1
                        INNER JOIN rainfall_data r2 
                        WHERE r1.id > r2.id 
                        AND r1.location_name = r2.location_name 
                        AND DATE(r1.created_at) = DATE(r2.created_at)
                    """)
                    
                    # Remove old data (keep only last 1000 records)
                    cursor.execute("""
                        DELETE FROM rainfall_data 
                        WHERE id NOT IN (
                            SELECT id FROM (
                                SELECT id FROM rainfall_data 
                                ORDER BY created_at DESC LIMIT 1000
                            ) AS temp
                        )
                    """)
                    
                    conn.commit()
                    cursor.close()
                    close_connection(conn)
                    
                    self.update_status("Database cleanup completed")
                    messagebox.showinfo("Success", "Database cleaned up!")
                    self.refresh_dashboard()
                else:
                    messagebox.showerror("Error", "Cannot connect to database")
                    
            except Exception as e:
                self.update_status("Error cleaning database")
                messagebox.showerror("Error", f"Database cleanup error: {str(e)}")

    def manage_database(self):
        """Open database management tool"""
        try:
            subprocess.Popen(["python", "database_manager.py"])
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open database manager: {str(e)}")

    def evaluate_model(self):
        """Evaluate model"""
        if self.model is None:
            messagebox.showwarning("Warning", "No model to evaluate!\nPlease train a model first.")
            return
        
        info_text = f"""MODEL INFORMATION:

Type: {'Advanced (3 Levels)' if self.is_advanced else 'Basic (2 Levels)'}
Number of Features: {len(self.features) if self.features else 0}
Features Used: {', '.join(self.features) if self.features else 'N/A'}

Status: Ready to use
Algorithm: Random Forest Classifier

Prediction Capability:
- {'LOW, MODERATE, HIGH' if self.is_advanced else 'No Flood, Flood'}
- High confidence with complete data
- Automatically adjusts to available data type
"""
        
        messagebox.showinfo("Model Evaluation", info_text)

    def generate_reports(self):
        """Generate report"""
        try:
            self.update_status("Generating report...", True)
            
            # Clear previous plots
            for ax in self.reports_axes.flat:
                ax.clear()
            
            # Load data based on date range
            df = load_combined_data()
            if df is None or len(df) == 0:
                df = load_data_from_db()
            
            if df is None or len(df) == 0:
                messagebox.showwarning("Warning", "No data to generate report!")
                return
            
            # Filter by date range
            date_range = self.date_range_var.get()
            if date_range != "All" and 'created_at' in df.columns:
                days_map = {"1 day": 1, "7 days": 7, "30 days": 30}
                days = days_map.get(date_range, 7)
                
                cutoff_date = datetime.now() - timedelta(days=days)
                df = df[pd.to_datetime(df['created_at']) >= cutoff_date]
            
            if len(df) == 0:
                messagebox.showwarning("Warning", f"No data in the {date_range} time range!")
                return
            
            # Chart 1: Temperature vs Rainfall correlation
            if 'temperature' in df.columns and 'rainfall_1h' in df.columns:
                self.reports_axes[0,0].scatter(df['temperature'], df['rainfall_1h'], alpha=0.6, c='blue')
                self.reports_axes[0,0].set_title('Temperature vs Rainfall Correlation')
                self.reports_axes[0,0].set_xlabel('Temperature (°C)')
                self.reports_axes[0,0].set_ylabel('Rainfall (mm/h)')
                self.reports_axes[0,0].grid(True, alpha=0.3)
            
            # Chart 2: Daily rainfall trend
            if 'created_at' in df.columns and 'rainfall_1h' in df.columns:
                df['date'] = pd.to_datetime(df['created_at']).dt.date
                daily_rainfall = df.groupby('date')['rainfall_1h'].mean()
                
                self.reports_axes[0,1].plot(daily_rainfall.index, daily_rainfall.values, 'g-o', markersize=4)
                self.reports_axes[0,1].set_title('Daily Rainfall Trend')
                self.reports_axes[0,1].set_ylabel('Average Rainfall (mm/h)')
                self.reports_axes[0,1].tick_params(axis='x', rotation=45)
                self.reports_axes[0,1].grid(True, alpha=0.3)
            
            # Chart 3: Risk distribution
            if 'flood_risk_level' in df.columns:
                risk_counts = df['flood_risk_level'].value_counts().sort_index()
                labels = ['LOW', 'MODERATE', 'HIGH']
                colors = ['#27ae60', '#f39c12', '#e74c3c']
                
                bars = self.reports_axes[1,0].bar(range(len(risk_counts)), risk_counts.values, 
                                                 color=[colors[i] for i in risk_counts.index])
                self.reports_axes[1,0].set_title('Risk Level Distribution')
                self.reports_axes[1,0].set_xlabel('Risk Level')
                self.reports_axes[1,0].set_ylabel('Occurrences')
                self.reports_axes[1,0].set_xticks(range(len(risk_counts)))
                self.reports_axes[1,0].set_xticklabels([labels[i] for i in risk_counts.index])
                
                # Add value labels on bars
                for bar, value in zip(bars, risk_counts.values):
                    self.reports_axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                                               str(value), ha='center', va='bottom')
            
            # Chart 4: Water level vs Alert levels (if available)
            if 'water_level' in df.columns:
                water_levels = df['water_level'].values
                self.reports_axes[1,1].hist(water_levels, bins=20, alpha=0.7, color='cyan', edgecolor='black')
                self.reports_axes[1,1].set_title('Water Level Distribution')
                self.reports_axes[1,1].set_xlabel('Water Level (cm)')
                self.reports_axes[1,1].set_ylabel('Frequency')
                
                # Add alert level lines
                self.reports_axes[1,1].axvline(x=180, color='yellow', linestyle='--', alpha=0.8, label='Alert 1')
                self.reports_axes[1,1].axvline(x=220, color='orange', linestyle='--', alpha=0.8, label='Alert 2')
                self.reports_axes[1,1].axvline(x=270, color='red', linestyle='--', alpha=0.8, label='Alert 3')
                self.reports_axes[1,1].legend()
            else:
                # Show humidity distribution instead
                if 'humidity' in df.columns:
                    humidity_data = df['humidity'].values
                    self.reports_axes[1,1].hist(humidity_data, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
                    self.reports_axes[1,1].set_title('Humidity Distribution')
                    self.reports_axes[1,1].set_xlabel('Humidity (%)')
                    self.reports_axes[1,1].set_ylabel('Frequency')
            
            plt.tight_layout()
            self.reports_canvas.draw()
            
            self.update_status(f"Report generated for {len(df)} records")
            messagebox.showinfo("Success", f"Report generated successfully!\nUsed {len(df)} records in {date_range}.")
            
        except Exception as e:
            self.update_status("Error generating report")
            messagebox.showerror("Error", f"Report generation error: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def export_to_excel(self):
        """Export data to Excel"""
        try:
            # Load data
            df = load_combined_data()
            if df is None or len(df) == 0:
                df = load_data_from_db()
            
            if df is None or len(df) == 0:
                messagebox.showwarning("Warning", "No data to export!")
                return
            
            # Ask for save location
            filename = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv"), ("All files", "*.*")],
                title="Save Report"
            )
            
            if filename:
                if filename.endswith('.xlsx'):
                    df.to_excel(filename, index=False)
                else:
                    df.to_csv(filename, index=False, encoding='utf-8-sig')
                
                messagebox.showinfo("Success", f"Exported {len(df)} records to:\n{filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Export error: {str(e)}")

    def export_report(self):
        """Export detailed report"""
        try:
            # Create comprehensive report
            df = load_combined_data()
            if df is None or len(df) == 0:
                df = load_data_from_db()
            
            if df is None or len(df) == 0:
                messagebox.showwarning("Warning", "No data to generate report!")
                return
            
            # Generate report text
            report_text = f"""FLOOD PREDICTION SYSTEM REPORT
{'='*60}

Report Generation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Records: {len(df)}

DATA STATISTICS:
{'='*30}
"""
            
            if 'temperature' in df.columns:
                report_text += f"""
Temperature:
- Average: {df['temperature'].mean():.1f}°C
- Min: {df['temperature'].min():.1f}°C
- Max: {df['temperature'].max():.1f}°C
"""
            
            if 'humidity' in df.columns:
                report_text += f"""
Humidity:
- Average: {df['humidity'].mean():.1f}%
- Min: {df['humidity'].min():.1f}%
- Max: {df['humidity'].max():.1f}%
"""
            
            if 'rainfall_1h' in df.columns:
                report_text += f"""
Rainfall:
- Average: {df['rainfall_1h'].mean():.1f}mm/h
- Min: {df['rainfall_1h'].min():.1f}mm/h
- Max: {df['rainfall_1h'].max():.1f}mm/h
- Number of times rainfall > 10mm/h: {len(df[df['rainfall_1h'] > 10])}
"""
            
            if 'water_level' in df.columns:
                report_text += f"""
River Water Level:
- Average: {df['water_level'].mean():.1f}cm
- Min: {df['water_level'].min():.1f}cm
- Max: {df['water_level'].max():.1f}cm
- Times exceeding Alert Level 1 (>180cm): {len(df[df['water_level'] > 180])}
- Times exceeding Alert Level 2 (>220cm): {len(df[df['water_level'] > 220])}
- Times exceeding Alert Level 3 (>270cm): {len(df[df['water_level'] > 270])}
"""
            
            # Risk analysis
            if 'flood_risk_level' in df.columns:
                risk_counts = df['flood_risk_level'].value_counts()
                report_text += f"""
RISK ANALYSIS (3 LEVELS):
{'='*35}
- LOW Risk: {risk_counts.get(0, 0)} times ({risk_counts.get(0, 0)/len(df)*100:.1f}%)
- MODERATE Risk: {risk_counts.get(1, 0)} times ({risk_counts.get(1, 0)/len(df)*100:.1f}%)
- HIGH Risk: {risk_counts.get(2, 0)} times ({risk_counts.get(2, 0)/len(df)*100:.1f}%)
"""
            elif 'flood_risk' in df.columns:
                risk_counts = df['flood_risk'].value_counts()
                report_text += f"""
RISK ANALYSIS (2 LEVELS):
{'='*35}
- No Risk: {risk_counts.get(0, 0)} times ({risk_counts.get(0, 0)/len(df)*100:.1f}%)
- Flood Risk: {risk_counts.get(1, 0)} times ({risk_counts.get(1, 0)/len(df)*100:.1f}%)
"""
            
            report_text += f"""
RECOMMENDATIONS:
{'='*15}
1. Continue monitoring weather and river data
2. Periodically update prediction model
3. Check and maintain warning systems
4. Train staff on emergency procedures

{'='*60}
Report generated by Flood Prediction System
"""
            
            # Save report
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Save Detailed Report"
            )
            
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(report_text)
                
                messagebox.showinfo("Success", f"Detailed report created:\n{filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Report creation error: {str(e)}")

    # Settings methods
    def test_db_connection(self):
        """Test database connection"""
        try:
            # Temporarily update connection parameters (this is just for testing)
            conn = get_connection()
            if conn:
                close_connection(conn)
                messagebox.showinfo("Success", "Database connection successful!")
            else:
                messagebox.showerror("Error", "Cannot connect to database!")
        except Exception as e:
            messagebox.showerror("Error", f"Connection error: {str(e)}")

    def save_db_settings(self):
        """Save database settings"""
        try:
            # Create or update .env file
            env_content = f"""MYSQL_HOST={self.db_host_var.get()}
MYSQL_PORT={self.db_port_var.get()}
MYSQL_USER={self.db_user_var.get()}
MYSQL_PASSWORD={self.db_pass_var.get()}
MYSQL_DATABASE=flood_prediction_db
WINDY_API_KEY={self.api_key_var.get()}
"""
            
            with open('.env', 'w') as f:
                f.write(env_content)
            
            messagebox.showinfo("Success", "Database settings saved!\nRestart the application to apply changes.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Settings save error: {str(e)}")

    def test_api_key(self):
        """Test API key"""
        api_key = self.api_key_var.get()
        if not api_key:
            messagebox.showwarning("Warning", "Please enter an API key!")
            return
        
        try:
            import requests
            # Test with a simple API call
            url = f"https://api.windy.com/api/point-forecast/v2"
            headers = {"key": api_key}
            
            # Simple test request
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                messagebox.showinfo("Success", "API key is valid!")
            else:
                messagebox.showerror("Error", f"Invalid API key!\nStatus code: {response.status_code}")
                
        except Exception as e:
            messagebox.showerror("Error", f"API test error: {str(e)}")

    def save_api_key(self):
        """Save API key"""
        try:
            # Read existing .env file
            env_content = ""
            try:
                with open('.env', 'r') as f:
                    env_content = f.read()
            except FileNotFoundError:
                pass
            
            # Update API key
            lines = env_content.split('\n')
            updated = False
            
            for i, line in enumerate(lines):
                if line.startswith('WINDY_API_KEY='):
                    lines[i] = f"WINDY_API_KEY={self.api_key_var.get()}"
                    updated = True
                    break
            
            if not updated:
                lines.append(f"WINDY_API_KEY={self.api_key_var.get()}")
            
            # Write back to file
            with open('.env', 'w') as f:
                f.write('\n'.join(lines))
            
            messagebox.showinfo("Success", "API key saved!")
            
        except Exception as e:
            messagebox.showerror("Error", f"API key save error: {str(e)}")

    # Help methods
    def show_help(self):
        """Show user guide"""
        help_text = """FLOOD PREDICTION SYSTEM USER GUIDE

1. INITIAL SETUP:
   - Go to File > Setup Database to create the database
   - Go to Settings to configure database and API key
   - Crawl weather and river data

2. TRAIN MODEL:
   - Go to Dashboard > Run Auto (recommended)
   - Or go to Model > Train Model

3. PREDICTION:
   - Go to Prediction tab
   - Enter weather and water level parameters
   - Click "PREDICT FLOOD"

4. VIEW DATA:
   - Data tab: View collected data
   - Reports tab: Generate charts and detailed reports

5. MANAGEMENT:
   - Periodically clean up database
   - Export reports when needed
   - Monitor system status on Dashboard

NOTES:
- System automatically selects appropriate model
- Advanced model (3 levels) with water level data
- Basic model (2 levels) with only weather data
"""
        
        # Create help window
        help_window = tk.Toplevel(self.root)
        help_window.title("User Guide")
        help_window.geometry("700x500")
        
        text_widget = tk.Text(help_window, wrap=tk.WORD, padx=10, pady=10)
        scrollbar = ttk.Scrollbar(help_window, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_widget.insert(tk.END, help_text)
        text_widget.config(state=tk.DISABLED)

    def show_about(self):
        """Show software information"""
        about_text = """FLOOD PREDICTION SYSTEM

Version: 1.0
Release Date: 2024

Main Features:
- Multi-level flood prediction (2-3 levels)
- Integration of weather and river data
- User-friendly interface
- Detailed reports and charts
- Automated data collection

Technologies Used:
- Python 3.x
- Tkinter (GUI)
- Scikit-learn (Machine Learning)
- MySQL (Database)
- Matplotlib (Visualization)

Developed by: AI Research Team
Email: support@floodprediction.com
Website: www.floodprediction.com
"""
        messagebox.showinfo("About Software", about_text)


def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = FloodPredictionGUI(root)
    
    # Set window icon (if available)
    try:
        root.iconbitmap('icon.ico')
    except:
        pass
    
    # Center window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    # Start the GUI
    root.mainloop()


if __name__ == "__main__":
    main()