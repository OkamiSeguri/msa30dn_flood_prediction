#!/usr/bin/env python3
"""
Khởi động GUI cho Hệ thống Dự báo Lũ lụt
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox

def check_dependencies():
    """Kiểm tra các thư viện cần thiết"""
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
            "Thiếu thư viện",
            f"Các thư viện sau chưa được cài đặt:\n{', '.join(missing_packages)}\n\n"
            f"Vui lòng chạy lệnh:\npip install {' '.join(missing_packages)}"
        )
        return False
    
    return True

def main():
    """Khởi động ứng dụng"""
    print("Đang khởi động Hệ thống Dự báo Lũ lụt...")
    
    # Kiểm tra dependencies
    if not check_dependencies():
        return
    
    # Import và chạy GUI
    try:
        from flood_prediction_gui import main as run_gui
        run_gui()
    except ImportError as e:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Lỗi import",
            f"Không thể import module GUI:\n{str(e)}\n\n"
            "Vui lòng kiểm tra file flood_prediction_gui.py"
        )
    except Exception as e:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Lỗi", f"Lỗi khởi động ứng dụng:\n{str(e)}")

if __name__ == "__main__":
    main()