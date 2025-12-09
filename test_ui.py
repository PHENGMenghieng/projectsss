import tkinter as tk

root = tk.Tk()
root.title("UI TEST")
root.geometry("300x200")

label = tk.Label(root, text="If you see this, UI works!", font=("Arial", 14))
label.pack(pady=20)

root.mainloop()
