#!/usr/bin/env python3
"""Tkinter GUI; the fields are generated from photo_to_svg.build_parser()."""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText

from PIL import Image, ImageTk
import photo_to_svg

# Only slider *ranges* are configured here. Option names/defaults/types/choices
# always come from photo_to_svg.build_parser(), not from a second CLI definition.
SLIDERS = {
    "accent_local_density": (1, 6, .1), "accent_density": (0, 2, .05),
    "accent_clusters": (1, 20, 1), "accent_spacing": (5, 100, 1),
    "accent_length": (3, 70, 1), "accent_jitter": (0, 1, .01),
    "accent_inner_lines": (0, 4, 1), "outline_cut_density": (.2, 4, .1),
    "outline_cut_length": (3, 80, 1), "outline_jitter": (0, 1, .01),
    "stroke_width": (.2, 5, .1), "grabcut_margin": (.02, .4, .01),
    "max_side": (200, 3000, 100),
}


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Stylus Effect — Engraving Studio")
        self.geometry("1160x810")
        self.parser = photo_to_svg.build_parser()
        self.controls: dict[str, tk.Variable] = {}
        self.preview_photo = None
        self.busy = False
        self._create_ui()

    def _create_ui(self):
        top = ttk.Frame(self, padding=12)
        top.pack(fill="x")
        ttk.Label(top, text="Stylus Effect", font=("Segoe UI", 19, "bold")).pack(side="left")
        ttk.Label(top, text=" ·  Pencil / contour engraving", foreground="gray").pack(side="left")
        ttk.Button(top, text="Select photo", command=self._choose_photo).pack(side="right")

        split = ttk.Panedwindow(self, orient="horizontal")
        split.pack(fill="both", expand=True, padx=10, pady=5)
        left = ttk.Frame(split)
        right = ttk.Frame(split, padding=(10, 2))
        split.add(left, weight=3)
        split.add(right, weight=2)

        notebook = ttk.Notebook(left)
        notebook.pack(fill="both", expand=True)
        for group in self.parser._action_groups:
            actions = [a for a in group._group_actions if a.dest != "help"]
            if not actions:
                continue
            page = ttk.Frame(notebook, padding=8)
            notebook.add(page, text=group.title)
            canvas = tk.Canvas(page, highlightthickness=0)
            scroll = ttk.Scrollbar(page, orient="vertical", command=canvas.yview)
            inner = ttk.Frame(canvas, padding=5)
            inner.bind("<Configure>", lambda e, c=canvas: c.configure(scrollregion=c.bbox("all")))
            canvas.create_window((0, 0), window=inner, anchor="nw")
            canvas.configure(yscrollcommand=scroll.set)
            canvas.pack(side="left", fill="both", expand=True)
            scroll.pack(side="right", fill="y")
            inner.columnconfigure(1, weight=1)
            for row, action in enumerate(actions):
                self._add_control(inner, row, action)

        ttk.Label(right, text="Preview", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.preview = ttk.Label(right, text="Generate an image to see the preview", anchor="center")
        self.preview.pack(fill="both", expand=True, pady=(5, 12))

        toolbar = ttk.Frame(right)
        toolbar.pack(fill="x", pady=(0, 8))
        self.run_button = ttk.Button(toolbar, text="Generate SVG + PNG", command=self._run)
        self.run_button.pack(side="left")
        ttk.Button(toolbar, text="Reset", command=self._reset).pack(side="left", padx=5)
        ttk.Button(toolbar, text="Open output", command=self._open_output).pack(side="left")

        ttk.Label(right, text="Command (updates automatically)").pack(anchor="w")
        self.command = tk.StringVar()
        ttk.Entry(right, textvariable=self.command, state="readonly").pack(fill="x", pady=(3, 8))
        self.log = ScrolledText(right, height=8, wrap="word", state="disabled")
        self.log.pack(fill="x")
        self._update_command()

    def _add_control(self, parent, row, action):
        dest = action.dest
        label = dest.replace("_", " ")
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=8)
        if isinstance(action, (argparse._StoreTrueAction, argparse.BooleanOptionalAction)):
            var = tk.BooleanVar(value=bool(action.default))
            ttk.Checkbutton(parent, variable=var).grid(row=row, column=1, sticky="w")
        elif action.choices:
            var = tk.StringVar(value=str(action.default))
            ttk.Combobox(parent, textvariable=var, values=list(action.choices),
                         state="readonly").grid(row=row, column=1, sticky="ew")
        elif dest in SLIDERS:
            lo, hi, step = SLIDERS[dest]
            var = tk.IntVar(value=int(action.default)) if action.type is int else tk.DoubleVar(value=float(action.default))
            frame = ttk.Frame(parent)
            frame.grid(row=row, column=1, sticky="ew")
            frame.columnconfigure(0, weight=1)
            tk.Scale(frame, from_=lo, to=hi, resolution=step, variable=var, orient="horizontal",
                     showvalue=False).grid(row=0, column=0, sticky="ew")
            ttk.Entry(frame, textvariable=var, width=9).grid(row=0, column=1, padx=5)
        else:
            var = tk.StringVar(value="" if action.default is None else str(action.default))
            frame = ttk.Frame(parent)
            frame.grid(row=row, column=1, sticky="ew")
            frame.columnconfigure(0, weight=1)
            ttk.Entry(frame, textvariable=var).grid(row=0, column=0, sticky="ew")
            if action.type is Path:
                ttk.Button(frame, text="...", width=3,
                           command=lambda d=dest, v=var: self._browse(d, v)).grid(row=0, column=1, padx=4)
        self.controls[dest] = var
        var.trace_add("write", lambda *_: self._update_command())

    def _browse(self, dest, var):
        if dest == "output":
            result = filedialog.askdirectory()
        elif dest == "input":
            result = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff")])
            if not result:
                result = filedialog.askdirectory()
        else:
            result = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg"), ("All", "*.*")])
        if result:
            var.set(result)

    def _choose_photo(self):
        self._browse("input", self.controls["input"])

    def _arguments(self) -> list[str]:
        args = []
        for action in self.parser._actions:
            if action.dest == "help" or action.dest not in self.controls:
                continue
            raw = self.controls[action.dest].get()
            opt = next((s for s in action.option_strings if s.startswith("--") and not s.startswith("--no-")), None)
            if opt is None:
                continue
            if isinstance(action, argparse._StoreTrueAction):
                if raw:
                    args.append(opt)
            elif isinstance(action, argparse.BooleanOptionalAction):
                if raw != action.default:
                    args.append(opt if raw else "--no-" + action.dest.replace("_", "-"))
            elif raw != "" and raw is not None:
                # Pass all current numeric/string settings; argparse validates them.
                args.extend((opt, str(raw)))
        return args

    def _update_command(self):
        if not hasattr(self, "command"):
            return
        try:
            parts = ["python", "photo_to_svg.py", *self._arguments()]
            self.command.set(subprocess.list2cmdline(parts) if os.name == "nt" else shlex.join(parts))
        except (ValueError, tk.TclError):
            self.command.set("Finish editing a value to generate the command")

    def _append_log(self, message):
        self.log.configure(state="normal")
        self.log.insert("end", message + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _run(self):
        if self.busy:
            return
        try:
            args = self._arguments()
        except (ValueError, tk.TclError) as exc:
            self._append_log(f"Invalid setting: {exc}")
            return
        self.busy = True
        self.run_button.configure(state="disabled")
        self._append_log("Running…")
        thread = threading.Thread(target=self._worker, args=(args,), daemon=True)
        thread.start()

    def _worker(self, args):
        script = Path(__file__).resolve().with_name("photo_to_svg.py")
        try:
            result = subprocess.run([sys.executable, str(script), *args], cwd=script.parent,
                                    text=True, capture_output=True, errors="replace")
            output = (result.stdout + "\n" + result.stderr).strip()
            self.after(0, self._finish, result.returncode, output)
        except OSError as exc:
            self.after(0, self._finish, 1, str(exc))

    def _finish(self, code, output):
        self.busy = False
        self.run_button.configure(state="normal")
        self._append_log(output or ("Done" if code == 0 else "No output"))
        if code != 0:
            return
        source = Path(str(self.controls["input"].get()))
        target = Path(str(self.controls["output"].get()))
        if not target.is_absolute():
            target = Path(__file__).resolve().parent / target
        if source.is_file():
            path = target / (source.stem + "_preview.png")
            if path.is_file():
                try:
                    with Image.open(path) as img:
                        img.thumbnail((470, 420))
                        self.preview_photo = ImageTk.PhotoImage(img.copy())
                    self.preview.configure(image=self.preview_photo, text="")
                except OSError as exc:
                    self._append_log(f"Preview unavailable: {exc}")

    def _open_output(self):
        target = Path(str(self.controls["output"].get())).resolve()
        if not target.exists():
            self._append_log("Generate artwork first to create the output folder")
            return
        if os.name == "nt":
            os.startfile(str(target))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])

    def _reset(self):
        for action in self.parser._actions:
            if action.dest in self.controls:
                self.controls[action.dest].set(action.default if action.default is not None else "")


if __name__ == "__main__":
    App().mainloop()