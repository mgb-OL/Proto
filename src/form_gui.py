import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime, date, time
import calendar


class DataCollectionForm(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Data Collection Form")
        self.geometry("560x300")
        self.resizable(False, False)

        # State variables
        self.file_path = tk.StringVar()
        self.serial_number = tk.StringVar()
        self.user_id = tk.StringVar()
        
        # Result variable
        self.result = None

        # Build UI
        self._build_file_row()
        self._build_text_rows()
        self._build_datetime_rows()
        self._build_actions()

    def _build_file_row(self):
        frm = ttk.Frame(self, padding=(12, 10, 12, 4))
        frm.grid(row=0, column=0, sticky="ew")
        frm.columnconfigure(1, weight=1)

        ttk.Label(frm, text="File:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        file_entry = ttk.Entry(frm, textvariable=self.file_path)
        file_entry.grid(row=0, column=1, sticky="ew", padx=(0, 8))

        ttk.Button(frm, text="Browse…", command=self._browse_file).grid(row=0, column=2, sticky="e")

    def _build_text_rows(self):
        frm = ttk.Frame(self, padding=(12, 4, 12, 4))
        frm.grid(row=1, column=0, sticky="ew")
        frm.columnconfigure(1, weight=1)

        ttk.Label(frm, text="Serial number:").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(frm, textvariable=self.serial_number).grid(row=0, column=1, sticky="ew", pady=4)

        ttk.Label(frm, text="User ID:").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(frm, textvariable=self.user_id).grid(row=1, column=1, sticky="ew", pady=4)

    def _build_datetime_rows(self):
        today = date.today()

        # Start datetime
        frm1 = ttk.LabelFrame(self, text="Start datetime", padding=(12, 8, 12, 8))
        frm1.grid(row=2, column=0, sticky="ew", padx=12, pady=(6, 4))

        # Date: YYYY-MM-DD via Spinboxes
        ttk.Label(frm1, text="Date (YYYY-MM-DD)").grid(row=0, column=0, padx=(0, 6))
        self.start_year_var = tk.StringVar(value=str(today.year))
        self.start_month_var = tk.StringVar(value=f"{today.month:02d}")
        self.start_day_var = tk.StringVar(value=f"{today.day:02d}")

        self.start_year = ttk.Spinbox(frm1, from_=1900, to=2100, width=5, wrap=True, textvariable=self.start_year_var)
        self.start_month = ttk.Spinbox(frm1, from_=1, to=12, width=3, wrap=True, format="%02.0f", textvariable=self.start_month_var)
        self.start_day = ttk.Spinbox(frm1, from_=1, to=31, width=3, wrap=True, format="%02.0f", textvariable=self.start_day_var)

        self.start_year.grid(row=0, column=1)
        ttk.Label(frm1, text="-").grid(row=0, column=2)
        self.start_month.grid(row=0, column=3)
        ttk.Label(frm1, text="-").grid(row=0, column=4)
        self.start_day.grid(row=0, column=5, padx=(0, 12))

        # Time: HH:MM
        ttk.Label(frm1, text="Time (HH:MM)").grid(row=0, column=6, padx=(0, 6))
        self.start_hour = ttk.Spinbox(frm1, from_=0, to=23, width=3, wrap=True, format="%02.0f")
        self.start_min = ttk.Spinbox(frm1, from_=0, to=59, width=3, increment=1, wrap=True, format="%02.0f")
        self.start_hour.grid(row=0, column=7)
        ttk.Label(frm1, text=":").grid(row=0, column=8)
        self.start_min.grid(row=0, column=9)

        # Default time values
        self.start_hour.delete(0, tk.END); self.start_hour.insert(0, "00")
        self.start_min.delete(0, tk.END); self.start_min.insert(0, "00")

        # Keep day range in sync with year/month
        self.start_year_var.trace_add('write', lambda *_: self._sync_day_range(self.start_year_var, self.start_month_var, self.start_day, self.start_day_var))
        self.start_month_var.trace_add('write', lambda *_: self._sync_day_range(self.start_year_var, self.start_month_var, self.start_day, self.start_day_var))
        self._sync_day_range(self.start_year_var, self.start_month_var, self.start_day, self.start_day_var)

        # End datetime
        frm2 = ttk.LabelFrame(self, text="End datetime", padding=(12, 8, 12, 12))
        frm2.grid(row=3, column=0, sticky="ew", padx=12, pady=(4, 6))

        ttk.Label(frm2, text="Date (YYYY-MM-DD)").grid(row=0, column=0, padx=(0, 6))
        self.end_year_var = tk.StringVar(value=str(today.year))
        self.end_month_var = tk.StringVar(value=f"{today.month:02d}")
        self.end_day_var = tk.StringVar(value=f"{today.day:02d}")

        self.end_year = ttk.Spinbox(frm2, from_=1900, to=2100, width=5, wrap=True, textvariable=self.end_year_var)
        self.end_month = ttk.Spinbox(frm2, from_=1, to=12, width=3, wrap=True, format="%02.0f", textvariable=self.end_month_var)
        self.end_day = ttk.Spinbox(frm2, from_=1, to=31, width=3, wrap=True, format="%02.0f", textvariable=self.end_day_var)

        self.end_year.grid(row=0, column=1)
        ttk.Label(frm2, text="-").grid(row=0, column=2)
        self.end_month.grid(row=0, column=3)
        ttk.Label(frm2, text="-").grid(row=0, column=4)
        self.end_day.grid(row=0, column=5, padx=(0, 12))

        ttk.Label(frm2, text="Time (HH:MM)").grid(row=0, column=6, padx=(0, 6))
        self.end_hour = ttk.Spinbox(frm2, from_=0, to=23, width=3, wrap=True, format="%02.0f")
        self.end_min = ttk.Spinbox(frm2, from_=0, to=59, width=3, increment=1, wrap=True, format="%02.0f")
        self.end_hour.grid(row=0, column=7)
        ttk.Label(frm2, text=":").grid(row=0, column=8)
        self.end_min.grid(row=0, column=9)

        self.end_hour.delete(0, tk.END); self.end_hour.insert(0, "23")
        self.end_min.delete(0, tk.END); self.end_min.insert(0, "59")

        self.end_year_var.trace_add('write', lambda *_: self._sync_day_range(self.end_year_var, self.end_month_var, self.end_day, self.end_day_var))
        self.end_month_var.trace_add('write', lambda *_: self._sync_day_range(self.end_year_var, self.end_month_var, self.end_day, self.end_day_var))
        self._sync_day_range(self.end_year_var, self.end_month_var, self.end_day, self.end_day_var)

    def _sync_day_range(self, year_var: tk.StringVar, month_var: tk.StringVar, day_spin: ttk.Spinbox, day_var: tk.StringVar):
        try:
            y = int(year_var.get())
            m = int(month_var.get())
            if not (1 <= m <= 12):
                return
            max_day = calendar.monthrange(y, m)[1]
            # Update spinbox upper bound and clamp current value
            day_spin.configure(to=max_day)
            try:
                d = int(day_var.get())
            except ValueError:
                d = 1
            if d > max_day:
                day_var.set(f"{max_day:02d}")
            elif d < 1:
                day_var.set("01")
        except ValueError:
            # Ignore until valid integers are present
            pass

    def _build_actions(self):
        frm = ttk.Frame(self, padding=(12, 0, 12, 12))
        frm.grid(row=4, column=0, sticky="e")
        ttk.Button(frm, text="Submit", command=self._on_submit).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(frm, text="Cancel", command=self._on_cancel).grid(row=0, column=1)

    def _browse_file(self):
        path = filedialog.askopenfilename(title="Choose a file")
        if path:
            self.file_path.set(path)

    def _combine_dt(self, d: date, hh: str, mm: str) -> datetime:
        try:
            h = int(hh); m = int(mm)
            if not (0 <= h <= 23 and 0 <= m <= 59):
                raise ValueError
            return datetime.combine(d, time(hour=h, minute=m))
        except ValueError:
            raise ValueError("Time must be HH (0-23) and MM (0-59)")

    def _on_submit(self):
        # Basic validation
        if not self.file_path.get():
            messagebox.showerror("Validation", "Please select a file.")
            return
        if not self.serial_number.get().strip():
            messagebox.showerror("Validation", "Please enter a serial number.")
            return
        if not self.user_id.get().strip():
            messagebox.showerror("Validation", "Please enter a user ID.")
            return

        try:
            start_d = self._get_date(self.start_year_var, self.start_month_var, self.start_day_var)
            end_d = self._get_date(self.end_year_var, self.end_month_var, self.end_day_var)
            start_dt = self._combine_dt(start_d, self.start_hour.get(), self.start_min.get())
            end_dt = self._combine_dt(end_d, self.end_hour.get(), self.end_min.get())
        except ValueError as e:
            messagebox.showerror("Validation", str(e))
            return

        if end_dt < start_dt:
            messagebox.showerror("Validation", "End datetime must be after start datetime.")
            return

        # Store result
        self.result = {
            'file_path': self.file_path.get(),
            'serial_number': self.serial_number.get().strip(),
            'user_id': self.user_id.get().strip(),
            'start_datetime': start_dt,
            'end_datetime': end_dt
        }
        
        # Close the window
        self.destroy()
    
    def _on_cancel(self):
        """Handle cancel button - set result to None and close."""
        self.result = None
        self.destroy()

    def _get_date(self, y_var: tk.StringVar, m_var: tk.StringVar, d_var: tk.StringVar) -> date:
        try:
            y = int(y_var.get()); m = int(m_var.get()); d = int(d_var.get())
            return date(y, m, d)
        except Exception:
            raise ValueError("Please enter a valid date (YYYY-MM-DD).")
    
    @staticmethod
    def show_form():
        """
        Show the form and return collected data.
        
        Returns:
            dict or None: Dictionary with form data if submitted, None if canceled.
                Keys: 'file_path', 'serial_number', 'user_id', 'start_datetime', 'end_datetime'
        """
        app = DataCollectionForm()
        app.mainloop()
        return app.result


if __name__ == "__main__":
    result = DataCollectionForm.show_form()
    if result:
        print("Form submitted with:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    else:
        print("Form canceled")
