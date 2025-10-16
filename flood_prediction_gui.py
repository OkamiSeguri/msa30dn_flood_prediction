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

# Import các module của dự án
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
        self.root.title("Hệ Thống Dự Báo Lũ Lụt")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Biến lưu trữ
        self.model = None
        self.features = None
        self.is_advanced = False
        self.current_data = None
        
        # Tạo giao diện
        self.setup_styles()
        self.create_menu()
        self.create_main_interface()
        self.create_status_bar()
        
        # Kiểm tra kết nối database ban đầu
        self.check_database_connection()

    def setup_styles(self):
        """Thiết lập styles cho giao diện"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Màu sắc chủ đạo
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), foreground='#2c3e50')
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'), foreground='#34495e')
        style.configure('Success.TLabel', foreground='#27ae60', font=('Arial', 10, 'bold'))
        style.configure('Warning.TLabel', foreground='#f39c12', font=('Arial', 10, 'bold'))
        style.configure('Error.TLabel', foreground='#e74c3c', font=('Arial', 10, 'bold'))
        style.configure('Accent.TButton', foreground='white', font=('Arial', 10, 'bold'))

    def create_menu(self):
        """Tạo menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Thiết lập Database", command=self.setup_database)
        file_menu.add_command(label="Xuất báo cáo", command=self.export_report)
        file_menu.add_separator()
        file_menu.add_command(label="Thoát", command=self.root.quit)
        
        # Data menu
        data_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Dữ liệu", menu=data_menu)
        data_menu.add_command(label="Crawl dữ liệu thời tiết", command=self.crawl_weather_data)
        data_menu.add_command(label="Crawl dữ liệu mực nước", command=self.crawl_river_data)
        data_menu.add_command(label="Quản lý Database", command=self.manage_database)
        
        # Model menu
        model_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Mô hình", menu=model_menu)
        model_menu.add_command(label="Huấn luyện mô hình", command=self.train_prediction_model)
        model_menu.add_command(label="Đánh giá mô hình", command=self.evaluate_model)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Trợ giúp", menu=help_menu)
        help_menu.add_command(label="Hướng dẫn sử dụng", command=self.show_help)
        help_menu.add_command(label="Về phần mềm", command=self.show_about)

    def create_main_interface(self):
        """Tạo giao diện chính"""
        # Tạo notebook để chia tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Tab 1: Dashboard
        self.create_dashboard_tab()
        
        # Tab 2: Dự báo
        self.create_prediction_tab()
        
        # Tab 3: Dữ liệu
        self.create_data_tab()
        
        # Tab 4: Báo cáo
        self.create_reports_tab()
        
        # Tab 5: Cài đặt
        self.create_settings_tab()

    def create_dashboard_tab(self):
        """Tab Dashboard - Tổng quan hệ thống"""
        dashboard_frame = ttk.Frame(self.notebook)
        self.notebook.add(dashboard_frame, text="Dashboard")
        
        # Frame trái - Thông tin hệ thống
        left_frame = ttk.LabelFrame(dashboard_frame, text="Tình trạng hệ thống", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Database status
        ttk.Label(left_frame, text="Trạng thái Database:", style='Header.TLabel').pack(anchor='w')
        self.db_status_label = ttk.Label(left_frame, text="Đang kiểm tra...", style='Warning.TLabel')
        self.db_status_label.pack(anchor='w', pady=(0,10))
        
        # Model status
        ttk.Label(left_frame, text="Trạng thái Mô hình:", style='Header.TLabel').pack(anchor='w')
        self.model_status_label = ttk.Label(left_frame, text="Chưa huấn luyện", style='Error.TLabel')
        self.model_status_label.pack(anchor='w', pady=(0,10))
        
        # Data summary
        ttk.Label(left_frame, text="Thống kê dữ liệu:", style='Header.TLabel').pack(anchor='w')
        self.data_summary_text = tk.Text(left_frame, height=8, wrap=tk.WORD)
        self.data_summary_text.pack(fill=tk.BOTH, expand=True, pady=(5,10))
        
        # Buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, text="Làm mới", command=self.refresh_dashboard).pack(side=tk.LEFT, padx=(0,5))
        ttk.Button(button_frame, text="Chạy tự động", command=self.run_auto_system).pack(side=tk.LEFT, padx=5)
        
        # Frame phải - Biểu đồ
        right_frame = ttk.LabelFrame(dashboard_frame, text="Biểu đồ tổng quan", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Matplotlib figure
        self.dashboard_fig, self.dashboard_axes = plt.subplots(2, 2, figsize=(8, 6))
        self.dashboard_fig.suptitle("Thống kê dữ liệu hệ thống")
        
        self.dashboard_canvas = FigureCanvasTkAgg(self.dashboard_fig, right_frame)
        self.dashboard_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_prediction_tab(self):
        """Tab Dự báo - Thực hiện dự báo lũ lụt"""
        prediction_frame = ttk.Frame(self.notebook)
        self.notebook.add(prediction_frame, text="Dự báo")
        
        # Frame trên - Input parameters
        input_frame = ttk.LabelFrame(prediction_frame, text="Nhập dữ liệu dự báo", padding=10)
        input_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Tạo 2 cột cho input
        left_input = ttk.Frame(input_frame)
        left_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        right_input = ttk.Frame(input_frame)
        right_input.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        
        # Left column - Weather data
        ttk.Label(left_input, text="Dữ liệu thời tiết:", style='Header.TLabel').pack(anchor='w')
        
        # Temperature
        ttk.Label(left_input, text="Nhiệt độ (°C):").pack(anchor='w', pady=(10,0))
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
        ttk.Label(left_input, text="Độ ẩm (%):").pack(anchor='w', pady=(10,0))
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
        ttk.Label(left_input, text="Áp suất (hPa):").pack(anchor='w', pady=(10,0))
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
        ttk.Label(left_input, text="Lượng mưa 1h (mm):").pack(anchor='w', pady=(10,0))
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
        ttk.Label(left_input, text="Tốc độ gió (km/h):").pack(anchor='w', pady=(10,0))
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
        ttk.Label(right_input, text="Dữ liệu mực nước sông:", style='Header.TLabel').pack(anchor='w')
        
        # Water level
        ttk.Label(right_input, text="Mực nước (cm):").pack(anchor='w', pady=(10,0))
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
        ttk.Label(right_input, text="Lưu lượng (m³/s):").pack(anchor='w', pady=(10,0))
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
        ttk.Label(right_input, text="Xu hướng mực nước:").pack(anchor='w', pady=(10,0))
        self.trend_var = tk.StringVar(value="stable")
        trend_combo = ttk.Combobox(right_input, textvariable=self.trend_var, 
                                  values=["stable", "rising", "falling"], state="readonly")
        trend_combo.pack(fill=tk.X, pady=2)
        
        # Location
        ttk.Label(right_input, text="Địa điểm:").pack(anchor='w', pady=(10,0))
        self.location_var = tk.StringVar(value="Hanoi")
        location_combo = ttk.Combobox(right_input, textvariable=self.location_var,
                                     values=["Hanoi", "Ho_Chi_Minh_City", "Da_Nang", 
                                            "Hue", "Can_Tho", "Hai_Phong", "Nha_Trang"])
        location_combo.pack(fill=tk.X, pady=2)
        
        # Predict button
        predict_btn = ttk.Button(right_input, text="DỰ BÁO LŨ LỤT", 
                                command=self.perform_prediction, style='Accent.TButton')
        predict_btn.pack(pady=20, ipadx=20, ipady=10)
        
        # Frame dưới - Kết quả
        result_frame = ttk.LabelFrame(prediction_frame, text="Kết quả dự báo", padding=10)
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
        right_result = ttk.LabelFrame(result_frame, text="Hiển thị trực quan", padding=10)
        right_result.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Risk level display
        self.risk_display_frame = ttk.Frame(right_result)
        self.risk_display_frame.pack(fill=tk.BOTH, expand=True)

    def create_data_tab(self):
        """Tab Dữ liệu - Quản lý và xem dữ liệu"""
        data_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_frame, text="Dữ liệu")
        
        # Control panel
        control_frame = ttk.LabelFrame(data_frame, text="Điều khiển dữ liệu", padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Buttons row 1
        btn_frame1 = ttk.Frame(control_frame)
        btn_frame1.pack(fill=tk.X, pady=2)
        
        ttk.Button(btn_frame1, text="Crawl thời tiết", 
                  command=self.crawl_weather_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame1, text="Crawl mực nước", 
                  command=self.crawl_river_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame1, text="Làm mới dữ liệu", 
                  command=self.refresh_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame1, text="Dọn dẹp DB", 
                  command=self.cleanup_database).pack(side=tk.LEFT, padx=5)
        
        # Data display
        data_display_frame = ttk.LabelFrame(data_frame, text="Dữ liệu hiện tại", padding=10)
        data_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Treeview frame
        tree_frame = ttk.Frame(data_display_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        # Treeview cho hiển thị dữ liệu
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
        """Tab Báo cáo - Thống kê và biểu đồ"""
        reports_frame = ttk.Frame(self.notebook)
        self.notebook.add(reports_frame, text="Báo cáo")
        
        # Control frame
        control_frame = ttk.LabelFrame(reports_frame, text="Tùy chọn báo cáo", padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Date range selection
        ttk.Label(control_frame, text="Thời gian:").pack(side=tk.LEFT)
        self.date_range_var = tk.StringVar(value="7 ngày")
        date_combo = ttk.Combobox(control_frame, textvariable=self.date_range_var,
                                 values=["1 ngày", "7 ngày", "30 ngày", "Tất cả"],
                                 state="readonly", width=15)
        date_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Tạo báo cáo", 
                  command=self.generate_reports).pack(side=tk.LEFT, padx=10)
        ttk.Button(control_frame, text="Xuất Excel", 
                  command=self.export_to_excel).pack(side=tk.LEFT, padx=5)
        
        # Charts frame
        charts_frame = ttk.LabelFrame(reports_frame, text="Biểu đồ phân tích", padding=10)
        charts_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Matplotlib figure for reports
        self.reports_fig, self.reports_axes = plt.subplots(2, 2, figsize=(12, 8))
        self.reports_fig.suptitle("Báo cáo phân tích dữ liệu lũ lụt")
        
        self.reports_canvas = FigureCanvasTkAgg(self.reports_fig, charts_frame)
        self.reports_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_settings_tab(self):
        """Tab Cài đặt - Cấu hình hệ thống"""
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="Cài đặt")
        
        # Database settings
        db_frame = ttk.LabelFrame(settings_frame, text="Cài đặt Database", padding=15)
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
        ttk.Button(btn_frame, text="Test kết nối", command=self.test_db_connection).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Lưu cài đặt", command=self.save_db_settings).pack(side=tk.LEFT, padx=5)
        
        # API settings
        api_frame = ttk.LabelFrame(settings_frame, text="Cài đặt API", padding=15)
        api_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(api_frame, text="Windy API Key:").grid(row=0, column=0, sticky='w', pady=5)
        self.api_key_var = tk.StringVar()
        ttk.Entry(api_frame, textvariable=self.api_key_var, width=50, show="*").grid(row=0, column=1, padx=10, pady=5)
        
        api_btn_frame = ttk.Frame(api_frame)
        api_btn_frame.grid(row=1, column=0, columnspan=2, pady=10)
        ttk.Button(api_btn_frame, text="Test API", command=self.test_api_key).pack(side=tk.LEFT, padx=5)
        ttk.Button(api_btn_frame, text="Lưu API Key", command=self.save_api_key).pack(side=tk.LEFT, padx=5)
        
        # Model settings
        model_frame = ttk.LabelFrame(settings_frame, text="Cài đặt Mô hình", padding=15)
        model_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(model_frame, text="Số cây Random Forest:").grid(row=0, column=0, sticky='w', pady=5)
        self.n_estimators_var = tk.IntVar(value=150)
        estimators_frame = ttk.Frame(model_frame)
        estimators_frame.grid(row=0, column=1, sticky='ew', padx=10, pady=5)
        ttk.Scale(estimators_frame, from_=50, to=500, orient=tk.HORIZONTAL, 
                 variable=self.n_estimators_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.estimators_label = ttk.Label(estimators_frame, text="150")
        self.estimators_label.pack(side=tk.RIGHT)
        
        ttk.Label(model_frame, text="Độ sâu tối đa:").grid(row=1, column=0, sticky='w', pady=5)
        self.max_depth_var = tk.IntVar(value=10)
        depth_frame = ttk.Frame(model_frame)
        depth_frame.grid(row=1, column=1, sticky='ew', padx=10, pady=5)
        ttk.Scale(depth_frame, from_=5, to=20, orient=tk.HORIZONTAL,
                 variable=self.max_depth_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.depth_label = ttk.Label(depth_frame, text="10")
        self.depth_label.pack(side=tk.RIGHT)

    def create_status_bar(self):
        """Tạo status bar"""
        self.status_bar = ttk.Frame(self.root, relief=tk.SUNKEN, borderwidth=1)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = ttk.Label(self.status_bar, text="Sẵn sàng", relief=tk.SUNKEN)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=2)
        
        self.progress_bar = ttk.Progressbar(self.status_bar, length=200, mode='indeterminate')
        self.progress_bar.pack(side=tk.RIGHT, padx=5, pady=2)

    def update_status(self, message, show_progress=False):
        """Cập nhật status bar"""
        self.status_label.config(text=message)
        if show_progress:
            self.progress_bar.start()
        else:
            self.progress_bar.stop()
        self.root.update_idletasks()

    def check_database_connection(self):
        """Kiểm tra kết nối database"""
        try:
            conn = get_connection()
            if conn:
                self.db_status_label.config(text="Kết nối thành công", style='Success.TLabel')
                close_connection(conn)
            else:
                self.db_status_label.config(text="Không kết nối được", style='Error.TLabel')
        except Exception as e:
            self.db_status_label.config(text=f"Lỗi: {str(e)}", style='Error.TLabel')

    def refresh_dashboard(self):
        """Làm mới dashboard"""
        self.update_status("Đang làm mới dashboard...", True)
        
        try:
            # Kiểm tra database
            self.check_database_connection()
            
            # Cập nhật thống kê dữ liệu
            self.update_data_summary()
            
            # Cập nhật biểu đồ
            self.update_dashboard_charts()
            
            self.update_status("Dashboard đã được làm mới")
        except Exception as e:
            self.update_status(f"Lỗi làm mới dashboard: {str(e)}")
            messagebox.showerror("Lỗi", f"Không thể làm mới dashboard:\n{str(e)}")

    def update_data_summary(self):
        """Cập nhật thống kê dữ liệu"""
        try:
            self.data_summary_text.delete(1.0, tk.END)
            
            conn = get_connection()
            if not conn:
                self.data_summary_text.insert(tk.END, "Không thể kết nối database")
                return
            
            cursor = conn.cursor()
            
            # Đếm records
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
            
            # Dữ liệu mới nhất
            try:
                cursor.execute("SELECT MAX(created_at) FROM rainfall_data")
                latest_weather = cursor.fetchone()[0]
            except:
                latest_weather = "N/A"
            
            summary = f"""Dữ liệu thời tiết: {weather_count} records
Dữ liệu mực nước: {river_count} records  
Dự báo đã tạo: {prediction_count} records

Dữ liệu mới nhất: {latest_weather}

Trạng thái: {'Đầy đủ' if river_count > 0 else 'Chỉ có thời tiết'}
Mô hình: {'3 cấp độ' if river_count > 0 else '2 cấp độ'}
"""
            
            self.data_summary_text.insert(tk.END, summary)
            
            cursor.close()
            close_connection(conn)
            
        except Exception as e:
            self.data_summary_text.insert(tk.END, f"Lỗi: {str(e)}")

    def update_dashboard_charts(self):
        """Cập nhật biểu đồ dashboard"""
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
                self.dashboard_axes[0,0].text(0.5, 0.5, 'Không có dữ liệu', 
                                            ha='center', va='center', transform=self.dashboard_axes[0,0].transAxes)
                self.dashboard_canvas.draw()
                return
            
            # Chart 1: Temperature trend
            if len(df) > 0 and 'temperature' in df.columns:
                temps = df['temperature'].tail(20).values
                self.dashboard_axes[0,0].plot(temps, 'b-o', markersize=3)
                self.dashboard_axes[0,0].set_title('Xu hướng nhiệt độ (20 mẫu gần nhất)')
                self.dashboard_axes[0,0].set_ylabel('°C')
                self.dashboard_axes[0,0].grid(True, alpha=0.3)
            
            # Chart 2: Rainfall distribution
            if 'rainfall_1h' in df.columns:
                rainfall_data = df['rainfall_1h'].values
                rainfall_data = rainfall_data[rainfall_data >= 0]  # Remove negative values
                self.dashboard_axes[0,1].hist(rainfall_data, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
                self.dashboard_axes[0,1].set_title('Phân bố lượng mưa')
                self.dashboard_axes[0,1].set_xlabel('mm/h')
                self.dashboard_axes[0,1].set_ylabel('Tần suất')
            
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
                    self.dashboard_axes[1,0].set_title('Phân bố mức độ nguy cơ')
            elif 'flood_risk' in df.columns:
                risk_counts = df['flood_risk'].value_counts()
                labels = ['Không lũ', 'Có lũ']
                colors = ['green', 'red']
                if len(risk_counts) > 0:
                    self.dashboard_axes[1,0].pie(risk_counts.values, labels=labels, 
                                                colors=colors, autopct='%1.1f%%', startangle=90)
                    self.dashboard_axes[1,0].set_title('Phân bố nguy cơ lũ')
            
            # Chart 4: Water level trend (if available)
            if 'water_level' in df.columns:
                water_levels = df['water_level'].tail(20).values
                self.dashboard_axes[1,1].plot(water_levels, 'r-o', markersize=3)
                self.dashboard_axes[1,1].set_title('Xu hướng mực nước (20 mẫu gần nhất)')
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
                    self.dashboard_axes[1,1].set_title('Xu hướng độ ẩm')
                    self.dashboard_axes[1,1].set_ylabel('%')
                    self.dashboard_axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            self.dashboard_canvas.draw()
            
        except Exception as e:
            print(f"Lỗi cập nhật biểu đồ: {e}")
            # Show error in first plot
            try:
                self.dashboard_axes[0,0].text(0.5, 0.5, f'Lỗi: {str(e)}', 
                                            ha='center', va='center', transform=self.dashboard_axes[0,0].transAxes)
                self.dashboard_canvas.draw()
            except:
                pass

    def perform_prediction(self):
        """Thực hiện dự báo lũ lụt"""
        if self.model is None:
            messagebox.showwarning("Cảnh báo", "Chưa có mô hình. Vui lòng huấn luyện mô hình trước!")
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
                messagebox.showerror("Lỗi", "Không thể thực hiện dự báo!")
                
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi dự báo: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def calculate_alert_level(self, water_level):
        """Tính toán mức báo động dựa trên mực nước"""
        if water_level >= 270:
            return 3
        elif water_level >= 220:
            return 2
        elif water_level >= 180:
            return 1
        else:
            return 0

    def display_prediction_result(self, result, input_data):
        """Hiển thị kết quả dự báo"""
        # Clear previous results
        self.result_text.delete(1.0, tk.END)
        
        # Clear risk display frame
        for widget in self.risk_display_frame.winfo_children():
            widget.destroy()
        
        # Display text result
        result_text = f"""{'='*50}
KẾT QUỢ DỰ BÁO LŨ LỤT
{'='*50}

Địa điểm: {self.location_var.get()}
Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Mô hình: {'Nâng cao (3 cấp độ)' if self.is_advanced else 'Cơ bản (2 cấp độ)'}

{'='*50}
DỮ LIỆU ĐẦU VÀO:
{'='*50}
- Nhiệt độ: {input_data['temperature']:.1f}°C
- Độ ẩm: {input_data['humidity']:.0f}%
- Áp suất: {input_data['pressure']:.0f} hPa
- Lượng mưa 1h: {input_data['rainfall_1h']:.1f} mm
- Lượng mưa 3h: {input_data['rainfall_3h']:.1f} mm
- Tốc độ gió: {input_data['wind_speed']:.0f} km/h
"""
        
        if self.is_advanced and 'water_level' in input_data:
            result_text += f"""- Mực nước: {input_data['water_level']:.0f} cm
- Lưu lượng: {input_data.get('flow_rate', 0):.0f} m³/s
- Xu hướng: {self.trend_var.get()}
- Mức báo động: {input_data.get('alert_level_exceeded', 0)}
"""
        
        result_text += f"""\n{'='*50}
KẾT QUẢ DỰ BÁO:
{'='*50}
"""
        
        if self.is_advanced:
            risk_level = result['risk_level']
            confidence = result['confidence']
            
            result_text += f"""Mức độ nguy cơ: {risk_level}
Độ tin cậy: {confidence:.1%}

Xác suất từng cấp độ:
"""
            
            for level, prob in result['probabilities'].items():
                result_text += f"  {level}: {prob:.1%}\n"
            
            # Create visual risk display
            self.create_risk_display(risk_level, confidence, result['probabilities'])
            
        else:
            flood_prob = result['probability_flood']
            confidence = result['confidence']
            
            result_text += f"""Xác suất lũ lụt: {flood_prob:.1%}
Độ tin cậy: {confidence:.1%}
"""
            
            # Determine risk level for basic model
            if flood_prob < 0.3:
                risk_level = "THẤP"
                color = "green"
            elif flood_prob < 0.7:
                risk_level = "TRUNG BÌNH"
                color = "orange"
            else:
                risk_level = "CAO"
                color = "red"
            
            result_text += f"Mức độ nguy cơ: {risk_level}\n"
            
            # Create simple risk display for basic model
            self.create_simple_risk_display(risk_level, flood_prob, color)
        
        # Add recommendations
        result_text += f"""\n{'='*50}
KHUYẾN NGHỊ:
{'='*50}
"""
        if self.is_advanced:
            if result['risk_level'] == 'HIGH':
                result_text += """NGUY CƠ CAO - Cần có biện pháp ứng phó khẩn cấp!
- Sơ tán dân cư vùng nguy hiểm
- Chuẩn bị vật tư cứu trợ khẩn cấp
- Theo dõi mực nước liên tục
- Kích hoạt đội ứng phó khẩn cấp
- Thông báo cảnh báo đến tất cả khu vực
"""
            elif result['risk_level'] == 'MODERATE':
                result_text += """NGUY CƠ TRUNG BÌNH - Theo dõi chặt chẽ
- Chuẩn bị sẵn sàng biện pháp ứng phó
- Thông báo người dân ở vùng trũng thấp
- Kiểm tra hệ thống thoát nước
- Theo dõi dự báo thời tiết liên tục
"""
            else:
                result_text += """NGUY CƠ THẤP - Tiếp tục theo dõi
- Duy trì hoạt động bình thường
- Theo dõi thông tin thời tiết định kỳ
- Kiểm tra hệ thống cảnh báo
"""
        else:
            if flood_prob > 0.7:
                result_text += """NGUY CƠ CAO - Cần chú ý!
- Theo dõi chặt chẽ diễn biến thời tiết
- Chuẩn bị biện pháp ứng phó
- Kiểm tra hệ thống thoát nước
"""
            elif flood_prob > 0.4:
                result_text += """NGUY CƠ TRUNG BÌNH
- Tiếp tục theo dõi tình hình
- Chuẩn bị sẵn sàng nếu cần thiết
"""
            else:
                result_text += """NGUY CƠ THẤP
- Hoạt động bình thường
- Theo dõi thông tin thời tiết
"""
        
        result_text += f"\n{'='*50}\nBáo cáo được tạo lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*50}"
        
        self.result_text.insert(tk.END, result_text)

    def create_risk_display(self, risk_level, confidence, probabilities):
        """Tạo hiển thị trực quan mức độ nguy cơ cho mô hình nâng cao"""
        # Risk level indicator
        risk_colors = {'LOW': '#27ae60', 'MODERATE': '#f39c12', 'HIGH': '#e74c3c'}
        color = risk_colors.get(risk_level, '#95a5a6')
        
        # Large risk indicator
        risk_frame = tk.Frame(self.risk_display_frame, bg=color, relief=tk.RAISED, bd=3)
        risk_frame.pack(fill=tk.X, pady=10)
        
        risk_label = tk.Label(risk_frame, text=f"NGUY CƠ {risk_level}", 
                             font=('Arial', 18, 'bold'), fg='white', bg=color)
        risk_label.pack(pady=15)
        
        confidence_label = tk.Label(risk_frame, text=f"Độ tin cậy: {confidence:.1%}",
                                   font=('Arial', 12), fg='white', bg=color)
        confidence_label.pack(pady=(0, 15))
        
        # Probability bars
        prob_frame = ttk.LabelFrame(self.risk_display_frame, text="Xác suất các cấp độ")
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
        """Tạo hiển thị đơn giản cho mô hình cơ bản"""
        # Risk indicator
        risk_frame = tk.Frame(self.risk_display_frame, bg=color, relief=tk.RAISED, bd=3)
        risk_frame.pack(fill=tk.X, pady=20)
        
        risk_label = tk.Label(risk_frame, text=f"NGUY CƠ {risk_level}", 
                             font=('Arial', 18, 'bold'), fg='white', bg=color)
        risk_label.pack(pady=20)
        
        prob_label = tk.Label(risk_frame, text=f"Xác suất lũ: {probability:.1%}",
                             font=('Arial', 12), fg='white', bg=color)
        prob_label.pack(pady=(0, 20))

    def train_prediction_model(self):
        """Huấn luyện mô hình dự báo"""
        self.update_status("Đang huấn luyện mô hình...", True)
        
        def train_in_thread():
            try:
                # Load data
                combined_df = load_combined_data()
                
                if combined_df is not None and len(combined_df) > 0:
                    print("Sử dụng dữ liệu kết hợp")
                    real_df = combined_df
                    use_advanced = True
                    real_df = create_flood_labels(real_df)
                    synthetic_df = generate_advanced_training_data(real_df)
                else:
                    print("Sử dụng dữ liệu cơ bản")
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
        """Callback khi huấn luyện hoàn thành"""
        self.model = model
        self.features = features
        self.is_advanced = is_advanced
        
        self.update_status("Huấn luyện mô hình hoàn thành!")
        
        if self.model is not None:
            self.model_status_label.config(
                text=f"Đã huấn luyện ({'Nâng cao' if self.is_advanced else 'Cơ bản'})",
                style='Success.TLabel'
            )
            messagebox.showinfo("Thành công", 
                              f"Mô hình {'nâng cao (3 cấp độ)' if self.is_advanced else 'cơ bản (2 cấp độ)'} đã được huấn luyện thành công!")
        else:
            self.model_status_label.config(text="Huấn luyện thất bại", style='Error.TLabel')

    def on_training_error(self, error_msg):
        """Callback khi có lỗi huấn luyện"""
        self.update_status(f"Lỗi huấn luyện mô hình")
        messagebox.showerror("Lỗi", f"Lỗi khi huấn luyện mô hình:\n{error_msg}")

    def run_auto_system(self):
        """Chạy hệ thống tự động"""
        def run_in_thread():
            try:
                self.root.after(0, lambda: self.update_status("Đang chạy hệ thống tự động...", True))
                
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
                                          messagebox.showerror("Lỗi", f"Lỗi chạy {' '.join(c)}:\n{e}"))
                            return
                    except subprocess.TimeoutExpired:
                        self.root.after(0, lambda c=cmd: 
                                      messagebox.showerror("Lỗi", f"Timeout chạy {' '.join(c)}"))
                        return
                
                # Train model after getting data
                self.root.after(0, lambda: self.update_status("Đang huấn luyện mô hình sau khi crawl dữ liệu...", True))
                self.root.after(1000, self.train_prediction_model)  # Delay to allow UI update
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Lỗi", f"Lỗi chạy tự động: {str(e)}"))
        
        threading.Thread(target=run_in_thread, daemon=True).start()

    # Database management methods
    def setup_database(self):
        """Thiết lập database"""
        try:
            # Import setup function
            from setup_db import setup_database
            result = setup_database()
            if result:
                messagebox.showinfo("Thành công", "Database đã được thiết lập thành công!")
                self.check_database_connection()
            else:
                messagebox.showerror("Lỗi", "Không thể thiết lập database!")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi thiết lập database: {str(e)}")

    def crawl_weather_data(self):
        """Crawl dữ liệu thời tiết"""
        self.update_status("Đang crawl dữ liệu thời tiết...", True)
        
        def crawl_in_thread():
            try:
                result = subprocess.run(["python", "rainfall_crawler.py"], 
                                      capture_output=True, text=True, timeout=120)
                if result.returncode == 0:
                    self.root.after(0, lambda: self.update_status("Crawl thời tiết hoàn thành!"))
                    self.root.after(0, lambda: messagebox.showinfo("Thành công", "Đã crawl dữ liệu thời tiết!"))
                    self.root.after(0, self.refresh_dashboard)
                else:
                    self.root.after(0, lambda: messagebox.showerror("Lỗi", f"Lỗi crawl thời tiết:\n{result.stderr}"))
            except subprocess.TimeoutExpired:
                self.root.after(0, lambda: messagebox.showerror("Lỗi", "Timeout khi crawl dữ liệu thời tiết"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Lỗi", f"Lỗi: {str(e)}"))
        
        threading.Thread(target=crawl_in_thread, daemon=True).start()

    def crawl_river_data(self):
        """Crawl dữ liệu mực nước sông"""
        self.update_status("Đang crawl dữ liệu mực nước...", True)
        
        def crawl_in_thread():
            try:
                result = subprocess.run(["python", "river_level_crawler.py"], 
                                      capture_output=True, text=True, timeout=120)
                if result.returncode == 0:
                    self.root.after(0, lambda: self.update_status("Crawl mực nước hoàn thành!"))
                    self.root.after(0, lambda: messagebox.showinfo("Thành công", "Đã crawl dữ liệu mực nước!"))
                    self.root.after(0, self.refresh_dashboard)
                else:
                    self.root.after(0, lambda: messagebox.showerror("Lỗi", f"Lỗi crawl mực nước:\n{result.stderr}"))
            except subprocess.TimeoutExpired:
                self.root.after(0, lambda: messagebox.showerror("Lỗi", "Timeout khi crawl dữ liệu mực nước"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Lỗi", f"Lỗi: {str(e)}"))
        
        threading.Thread(target=crawl_in_thread, daemon=True).start()

    def refresh_data(self):
        """Làm mới dữ liệu hiển thị"""
        try:
            self.update_status("Đang tải dữ liệu...", True)
            
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
            
            self.update_status(f"Đã tải {len(df) if df is not None else 0} records")
            
        except Exception as e:
            self.update_status("Lỗi tải dữ liệu")
            messagebox.showerror("Lỗi", f"Lỗi làm mới dữ liệu: {str(e)}")

    def cleanup_database(self):
        """Dọn dẹp database"""
        if messagebox.askyesno("Xác nhận", "Bạn có chắc muốn dọn dẹp database?\nThao tác này sẽ xóa dữ liệu cũ và trùng lặp."):
            try:
                self.update_status("Đang dọn dẹp database...", True)
                
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
                    
                    self.update_status("Dọn dẹp database hoàn thành")
                    messagebox.showinfo("Thành công", "Đã dọn dẹp database!")
                    self.refresh_dashboard()
                else:
                    messagebox.showerror("Lỗi", "Không thể kết nối database")
                    
            except Exception as e:
                self.update_status("Lỗi dọn dẹp database")
                messagebox.showerror("Lỗi", f"Lỗi dọn dẹp database: {str(e)}")

    def manage_database(self):
        """Mở công cụ quản lý database"""
        try:
            subprocess.Popen(["python", "database_manager.py"])
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể mở database manager: {str(e)}")

    def evaluate_model(self):
        """Đánh giá mô hình"""
        if self.model is None:
            messagebox.showwarning("Cảnh báo", "Chưa có mô hình để đánh giá!\nVui lòng huấn luyện mô hình trước.")
            return
        
        info_text = f"""THÔNG TIN MÔ HÌNH:

Loại: {'Nâng cao (3 cấp độ)' if self.is_advanced else 'Cơ bản (2 cấp độ)'}
Số đặc trưng: {len(self.features) if self.features else 0}
Đặc trưng sử dụng: {', '.join(self.features) if self.features else 'N/A'}

Trạng thái: Sẵn sàng sử dụng
Thuật toán: Random Forest Classifier

Khả năng dự báo:
- {'LOW, MODERATE, HIGH' if self.is_advanced else 'No Flood, Flood'}
- Độ tin cậy cao với dữ liệu đầy đủ
- Tự động điều chỉnh theo loại dữ liệu có sẵn
"""
        
        messagebox.showinfo("Đánh giá mô hình", info_text)

    def generate_reports(self):
        """Tạo báo cáo"""
        try:
            self.update_status("Đang tạo báo cáo...", True)
            
            # Clear previous plots
            for ax in self.reports_axes.flat:
                ax.clear()
            
            # Load data based on date range
            df = load_combined_data()
            if df is None or len(df) == 0:
                df = load_data_from_db()
            
            if df is None or len(df) == 0:
                messagebox.showwarning("Cảnh báo", "Không có dữ liệu để tạo báo cáo!")
                return
            
            # Filter by date range
            date_range = self.date_range_var.get()
            if date_range != "Tất cả" and 'created_at' in df.columns:
                days_map = {"1 ngày": 1, "7 ngày": 7, "30 ngày": 30}
                days = days_map.get(date_range, 7)
                
                cutoff_date = datetime.now() - timedelta(days=days)
                df = df[pd.to_datetime(df['created_at']) >= cutoff_date]
            
            if len(df) == 0:
                messagebox.showwarning("Cảnh báo", f"Không có dữ liệu trong khoảng thời gian {date_range}!")
                return
            
            # Chart 1: Temperature vs Rainfall correlation
            if 'temperature' in df.columns and 'rainfall_1h' in df.columns:
                self.reports_axes[0,0].scatter(df['temperature'], df['rainfall_1h'], alpha=0.6, c='blue')
                self.reports_axes[0,0].set_title('Mối quan hệ Nhiệt độ - Lượng mưa')
                self.reports_axes[0,0].set_xlabel('Nhiệt độ (°C)')
                self.reports_axes[0,0].set_ylabel('Lượng mưa (mm/h)')
                self.reports_axes[0,0].grid(True, alpha=0.3)
            
            # Chart 2: Daily rainfall trend
            if 'created_at' in df.columns and 'rainfall_1h' in df.columns:
                df['date'] = pd.to_datetime(df['created_at']).dt.date
                daily_rainfall = df.groupby('date')['rainfall_1h'].mean()
                
                self.reports_axes[0,1].plot(daily_rainfall.index, daily_rainfall.values, 'g-o', markersize=4)
                self.reports_axes[0,1].set_title('Xu hướng lượng mưa theo ngày')
                self.reports_axes[0,1].set_ylabel('Lượng mưa TB (mm/h)')
                self.reports_axes[0,1].tick_params(axis='x', rotation=45)
                self.reports_axes[0,1].grid(True, alpha=0.3)
            
            # Chart 3: Risk distribution
            if 'flood_risk_level' in df.columns:
                risk_counts = df['flood_risk_level'].value_counts().sort_index()
                labels = ['LOW', 'MODERATE', 'HIGH']
                colors = ['#27ae60', '#f39c12', '#e74c3c']
                
                bars = self.reports_axes[1,0].bar(range(len(risk_counts)), risk_counts.values, 
                                                 color=[colors[i] for i in risk_counts.index])
                self.reports_axes[1,0].set_title('Phân bố mức độ nguy cơ')
                self.reports_axes[1,0].set_xlabel('Mức độ nguy cơ')
                self.reports_axes[1,0].set_ylabel('Số lần')
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
                self.reports_axes[1,1].set_title('Phân bố mực nước')
                self.reports_axes[1,1].set_xlabel('Mực nước (cm)')
                self.reports_axes[1,1].set_ylabel('Tần suất')
                
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
                    self.reports_axes[1,1].set_title('Phân bố độ ẩm')
                    self.reports_axes[1,1].set_xlabel('Độ ẩm (%)')
                    self.reports_axes[1,1].set_ylabel('Tần suất')
            
            plt.tight_layout()
            self.reports_canvas.draw()
            
            self.update_status(f"Đã tạo báo cáo cho {len(df)} records")
            messagebox.showinfo("Thành công", f"Đã tạo báo cáo thành công!\nSử dụng {len(df)} records trong {date_range}.")
            
        except Exception as e:
            self.update_status("Lỗi tạo báo cáo")
            messagebox.showerror("Lỗi", f"Lỗi tạo báo cáo: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def export_to_excel(self):
        """Xuất dữ liệu ra Excel"""
        try:
            # Load data
            df = load_combined_data()
            if df is None or len(df) == 0:
                df = load_data_from_db()
            
            if df is None or len(df) == 0:
                messagebox.showwarning("Cảnh báo", "Không có dữ liệu để xuất!")
                return
            
            # Ask for save location
            filename = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv"), ("All files", "*.*")],
                title="Lưu báo cáo"
            )
            
            if filename:
                if filename.endswith('.xlsx'):
                    df.to_excel(filename, index=False)
                else:
                    df.to_csv(filename, index=False, encoding='utf-8-sig')
                
                messagebox.showinfo("Thành công", f"Đã xuất {len(df)} records vào:\n{filename}")
                
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi xuất file: {str(e)}")

    def export_report(self):
        """Xuất báo cáo chi tiết"""
        try:
            # Create comprehensive report
            df = load_combined_data()
            if df is None or len(df) == 0:
                df = load_data_from_db()
            
            if df is None or len(df) == 0:
                messagebox.showwarning("Cảnh báo", "Không có dữ liệu để tạo báo cáo!")
                return
            
            # Generate report text
            report_text = f"""BÁO CÁO HỆ THỐNG DỰ BÁO LŨ LỤT
{'='*60}

Thời gian tạo báo cáo: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Tổng số records: {len(df)}

THỐNG KÊ DỮ LIỆU:
{'='*30}
"""
            
            if 'temperature' in df.columns:
                report_text += f"""
Nhiệt độ:
- Trung bình: {df['temperature'].mean():.1f}°C
- Min: {df['temperature'].min():.1f}°C
- Max: {df['temperature'].max():.1f}°C
"""
            
            if 'humidity' in df.columns:
                report_text += f"""
Độ ẩm:
- Trung bình: {df['humidity'].mean():.1f}%
- Min: {df['humidity'].min():.1f}%
- Max: {df['humidity'].max():.1f}%
"""
            
            if 'rainfall_1h' in df.columns:
                report_text += f"""
Lượng mưa:
- Trung bình: {df['rainfall_1h'].mean():.1f}mm/h
- Min: {df['rainfall_1h'].min():.1f}mm/h
- Max: {df['rainfall_1h'].max():.1f}mm/h
- Tổng số lần mưa > 10mm/h: {len(df[df['rainfall_1h'] > 10])}
"""
            
            if 'water_level' in df.columns:
                report_text += f"""
Mực nước sông:
- Trung bình: {df['water_level'].mean():.1f}cm
- Min: {df['water_level'].min():.1f}cm
- Max: {df['water_level'].max():.1f}cm
- Số lần vượt báo động cấp 1 (>180cm): {len(df[df['water_level'] > 180])}
- Số lần vượt báo động cấp 2 (>220cm): {len(df[df['water_level'] > 220])}
- Số lần vượt báo động cấp 3 (>270cm): {len(df[df['water_level'] > 270])}
"""
            
            # Risk analysis
            if 'flood_risk_level' in df.columns:
                risk_counts = df['flood_risk_level'].value_counts()
                report_text += f"""
PHÂN TÍCH NGUY CƠ (3 CẤP ĐỘ):
{'='*35}
- Nguy cơ THẤP: {risk_counts.get(0, 0)} lần ({risk_counts.get(0, 0)/len(df)*100:.1f}%)
- Nguy cơ TRUNG BÌNH: {risk_counts.get(1, 0)} lần ({risk_counts.get(1, 0)/len(df)*100:.1f}%)
- Nguy cơ CAO: {risk_counts.get(2, 0)} lần ({risk_counts.get(2, 0)/len(df)*100:.1f}%)
"""
            elif 'flood_risk' in df.columns:
                risk_counts = df['flood_risk'].value_counts()
                report_text += f"""
PHÂN TÍCH NGUY CƠ (2 CẤP ĐỘ):
{'='*35}
- Không có nguy cơ: {risk_counts.get(0, 0)} lần ({risk_counts.get(0, 0)/len(df)*100:.1f}%)
- Có nguy cơ lũ: {risk_counts.get(1, 0)} lần ({risk_counts.get(1, 0)/len(df)*100:.1f}%)
"""
            
            report_text += f"""
KHUYẾN NGHỊ:
{'='*15}
1. Tiếp tục theo dõi dữ liệu thời tiết và mực nước sông
2. Cập nhật mô hình dự báo định kỳ
3. Kiểm tra và bảo trì hệ thống cảnh báo
4. Đào tạo nhân viên về quy trình ứng phó khẩn cấp

{'='*60}
Báo cáo được tạo bởi Hệ thống Dự báo Lũ lụt
"""
            
            # Save report
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Lưu báo cáo chi tiết"
            )
            
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(report_text)
                
                messagebox.showinfo("Thành công", f"Đã tạo báo cáo chi tiết:\n{filename}")
                
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi tạo báo cáo: {str(e)}")

    # Settings methods
    def test_db_connection(self):
        """Test kết nối database"""
        try:
            # Temporarily update connection parameters (this is just for testing)
            conn = get_connection()
            if conn:
                close_connection(conn)
                messagebox.showinfo("Thành công", "Kết nối database thành công!")
            else:
                messagebox.showerror("Lỗi", "Không thể kết nối database!")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi kết nối: {str(e)}")

    def save_db_settings(self):
        """Lưu cài đặt database"""
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
            
            messagebox.showinfo("Thành công", "Đã lưu cài đặt database!\nKhởi động lại ứng dụng để áp dụng.")
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi lưu cài đặt: {str(e)}")

    def test_api_key(self):
        """Test API key"""
        api_key = self.api_key_var.get()
        if not api_key:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập API key!")
            return
        
        try:
            import requests
            # Test with a simple API call
            url = f"https://api.windy.com/api/point-forecast/v2"
            headers = {"key": api_key}
            
            # Simple test request
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                messagebox.showinfo("Thành công", "API key hợp lệ!")
            else:
                messagebox.showerror("Lỗi", f"API key không hợp lệ!\nStatus code: {response.status_code}")
                
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi test API: {str(e)}")

    def save_api_key(self):
        """Lưu API key"""
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
            
            messagebox.showinfo("Thành công", "Đã lưu API key!")
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi lưu API key: {str(e)}")

    # # Help methods
    def show_help(self):
        """Hiển thị hướng dẫn sử dụng"""
        help_text = """HƯỚNG DẪN SỬ DỤNG HỆ THỐNG DỰ BÁO LŨ LỤT

1. THIẾT LẬP BAN ĐẦU:
   - Vào File > Thiết lập Database để tạo database
   - Vào Cài đặt để cấu hình database và API key
   - Crawl dữ liệu thời tiết và mực nước sông

2. HUẤN LUYỆN MÔ HÌNH:
   - Vào tab Dashboard > Chạy tự động (khuyến nghị)
   - Hoặc vào Mô hình > Huấn luyện mô hình

3. DỰ BÁO:
   - Vào tab Dự báo
   - Nhập các thông số thời tiết và mực nước
   - Nhấn "DỰ BÁO LŨ LỤT"

4. XEM DỮ LIỆU:
   - Tab Dữ liệu: Xem dữ liệu đã thu thập
   - Tab Báo cáo: Tạo biểu đồ và báo cáo chi tiết

5. QUẢN LÝ:
   - Dọn dẹp database định kỳ
   - Xuất báo cáo khi cần thiết
   - Theo dõi trạng thái hệ thống ở Dashboard

LƯU Ý:
- Hệ thống tự động chọn mô hình phù hợp
- Mô hình nâng cao (3 cấp độ) khi có dữ liệu mực nước
- Mô hình cơ bản (2 cấp độ) khi chỉ có dữ liệu thời tiết
"""
        
        # Create help window
        help_window = tk.Toplevel(self.root)
        help_window.title("Hướng dẫn sử dụng")
        help_window.geometry("700x500")
        
        text_widget = tk.Text(help_window, wrap=tk.WORD, padx=10, pady=10)
        scrollbar = ttk.Scrollbar(help_window, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_widget.insert(tk.END, help_text)
        text_widget.config(state=tk.DISABLED)

    def show_about(self):
        """Hiển thị thông tin về phần mềm"""
        about_text = """HỆ THỐNG DỰ BÁO LŨ LỤT

Phiên bản: 1.0
Ngày phát hành: 2024

Tính năng chính:
- Dự báo lũ lụt đa cấp độ (2-3 cấp)
- Tích hợp dữ liệu thời tiết và mực nước sông
- Giao diện trực quan, dễ sử dụng
- Báo cáo và biểu đồ chi tiết
- Tự động hóa thu thập dữ liệu

Công nghệ sử dụng:
- Python 3.x
- Tkinter (GUI)
- Scikit-learn (Machine Learning)
- MySQL (Database)
- Matplotlib (Visualization)

Phát triển bởi: Nhóm nghiên cứu AI
Email: support@floodprediction.com
Website: www.floodprediction.com
"""
        messagebox.showinfo("Về phần mềm", about_text)


def main():
    """Hàm main để chạy ứng dụng"""
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