#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

import photo_to_svg as core

RANGES = {
    'max_side': (0, 4000, 50),
    'grabcut_margin': (0.02, 0.4, 0.01),
    'stroke_width': (0.1, 6.0, 0.1),
    'opacity': (0.0, 1.0, 0.01),
    'detail_width': (0.1, 4.0, 0.05),
    'detail_opacity': (0.0, 1.0, 0.01),
    'highlight_width': (0.1, 4.0, 0.05),
    'highlight_opacity': (0.0, 1.0, 0.01),
    'hatch_angle': (-90.0, 90.0, 1.0),
    'hatch_spacing': (3.0, 80.0, 0.5),
    'hatch_length': (2.0, 120.0, 0.5),
    'hatch_density': (0.0, 3.0, 0.05),
    'hatch_width': (0.1, 4.0, 0.05),
    'hatch_opacity': (0.0, 1.0, 0.01),
    'accent_density': (0.0, 2.0, 0.05),
    'accent_spacing': (5.0, 100.0, 0.5),
    'accent_local_density': (1.0, 6.0, 0.1),
    'accent_length': (3.0, 70.0, 0.5),
    'accent_jitter': (0.0, 1.0, 0.01),
    'accent_border_weight': (0.0, 1.0, 0.01),
    'accent_inner_lines': (0, 4, 1),
    'accent_clusters': (1, 12, 1),
    'accent_width': (0.1, 4.0, 0.05),
    'outline_cut_density': (0.2, 4.0, 0.05),
    'outline_cut_length': (3.0, 80.0, 0.5),
    'outline_jitter': (0.0, 1.0, 0.01),
}

class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title('Photo to SVG — GUI')
        self.geometry('1160x860')
        self.parser = core.build_parser()
        self.defaults = vars(core.parse_args([]))
        self.widgets = {}
        self.action_map = {}
        self._build()

    def _build(self) -> None:
        top = ttk.Frame(self, padding=10)
        top.pack(fill='x')
        ttk.Label(top, text='Photo to SVG GUI', font=('Segoe UI', 16, 'bold')).pack(anchor='w')
        ttk.Label(top, text='Controlli generati automaticamente dagli argomenti di photo_to_svg.py').pack(anchor='w', pady=(2, 8))

        main = ttk.Panedwindow(self, orient='horizontal')
        main.pack(fill='both', expand=True)

        left_wrap = ttk.Frame(main)
        right = ttk.Frame(main, padding=10)
        main.add(left_wrap, weight=3)
        main.add(right, weight=2)

        canvas = tk.Canvas(left_wrap, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_wrap, orient='vertical', command=canvas.yview)
        self.form = ttk.Frame(canvas, padding=8)
        self.form.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=self.form, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        sections = {
            'Input / Output': {'input', 'output', 'preset', 'mode', 'style', 'max_side', 'preview', 'debug'},
            'Mask & segmentation': {'mask', 'grabcut', 'grabcut_margin', 'auto_mask'},
            'Linee base': {'stroke', 'stroke_width', 'opacity', 'detail_width', 'detail_opacity', 'highlight_width', 'highlight_opacity'},
            'Adaptive hatching': {'hatching', 'hatch_tone', 'hatch_region', 'hatch_flow', 'hatch_angle', 'hatch_spacing', 'hatch_length', 'hatch_density', 'hatch_width', 'hatch_opacity', 'hatch_seed'},
            'Accent style': {'accent_region', 'accent_follow', 'accent_density', 'accent_spacing', 'accent_local_density', 'accent_length', 'accent_jitter', 'accent_border_weight', 'accent_inner_lines', 'accent_clusters', 'accent_width', 'accent_seed'},
            'Outline / export': {'outline_style', 'outline_cut_density', 'outline_cut_length', 'outline_jitter', 'export_transparent', 'svg_background', 'background'},
        }
        used = set()
        row = 0
        for title, names in sections.items():
            lf = ttk.LabelFrame(self.form, text=title, padding=8)
            lf.grid(row=row, column=0, sticky='ew', pady=6)
            lf.columnconfigure(1, weight=1)
            self._populate_section(lf, names)
            used |= names
            row += 1
        remaining = [a for a in self.parser._actions if getattr(a, 'dest', None) not in used and a.dest not in ('help',)]
        if remaining:
            lf = ttk.LabelFrame(self.form, text='Other', padding=8)
            lf.grid(row=row, column=0, sticky='ew', pady=6)
            lf.columnconfigure(1, weight=1)
            self._populate_actions(lf, remaining)

        cmd_frame = ttk.LabelFrame(right, text='Comando generato', padding=8)
        cmd_frame.pack(fill='x')
        self.command_var = tk.StringVar()
        ttk.Entry(cmd_frame, textvariable=self.command_var).pack(fill='x')

        btns = ttk.Frame(right)
        btns.pack(fill='x', pady=8)
        ttk.Button(btns, text='Aggiorna comando', command=self.update_command).pack(side='left')
        ttk.Button(btns, text='Esegui', command=self.run_command).pack(side='left', padx=6)
        ttk.Button(btns, text='Reset default', command=self.reset_defaults).pack(side='left')

        info = ttk.LabelFrame(right, text='Output', padding=8)
        info.pack(fill='both', expand=True)
        self.log = ScrolledText(info, height=30)
        self.log.pack(fill='both', expand=True)

        self.update_command()

    def _populate_section(self, frame: ttk.LabelFrame, names: set[str]) -> None:
        actions = [a for a in self.parser._actions if getattr(a, 'dest', None) in names]
        self._populate_actions(frame, actions)

    def _populate_actions(self, frame, actions):
        row = 0
        for action in actions:
            dest = action.dest
            if dest == 'help':
                continue
            self.action_map[dest] = action
            ttk.Label(frame, text=dest.replace('_', ' ')).grid(row=row, column=0, sticky='w', padx=(0, 8), pady=4)
            widget = self._make_widget(frame, action, row)
            self.widgets[dest] = widget
            row += 1

    def _make_widget(self, frame, action, row):
        dest = action.dest
        default = self.defaults.get(dest)
        if isinstance(action, argparse._StoreTrueAction):
            var = tk.BooleanVar(value=bool(default))
            w = ttk.Checkbutton(frame, variable=var, command=self.update_command)
            w.var = var
            w.grid(row=row, column=1, sticky='w')
            return w
        if isinstance(action, argparse.BooleanOptionalAction):
            var = tk.BooleanVar(value=bool(default))
            w = ttk.Checkbutton(frame, variable=var, command=self.update_command)
            w.var = var
            w.grid(row=row, column=1, sticky='w')
            return w
        if action.choices:
            var = tk.StringVar(value=str(default))
            w = ttk.Combobox(frame, textvariable=var, values=list(action.choices), state='readonly')
            w.bind('<<ComboboxSelected>>', lambda e: self.update_command())
            w.grid(row=row, column=1, sticky='ew')
            w.var = var
            return w
        if action.type in (int, float):
            holder = ttk.Frame(frame)
            holder.grid(row=row, column=1, sticky='ew')
            holder.columnconfigure(1, weight=1)
            var = tk.DoubleVar(value=float(default) if default is not None else 0.0)
            if action.type is int:
                var = tk.IntVar(value=int(default) if default is not None else 0)
            if dest in RANGES:
                lo, hi, step = RANGES[dest]
                scale = tk.Scale(holder, from_=lo, to=hi, orient='horizontal', resolution=step,
                                 variable=var, command=lambda _=None: self.update_command())
                scale.grid(row=0, column=0, sticky='ew')
            entry = ttk.Entry(holder, textvariable=var, width=10)
            entry.grid(row=0, column=1, padx=(6, 0))
            entry.bind('<KeyRelease>', lambda e: self.update_command())
            holder.var = var
            return holder
        var = tk.StringVar(value='' if default is None else str(default))
        holder = ttk.Frame(frame)
        holder.grid(row=row, column=1, sticky='ew')
        holder.columnconfigure(0, weight=1)
        entry = ttk.Entry(holder, textvariable=var)
        entry.grid(row=0, column=0, sticky='ew')
        entry.bind('<KeyRelease>', lambda e: self.update_command())
        if action.type is Path or dest in ('input', 'output', 'mask'):
            def browse():
                current = var.get().strip()
                if dest == 'output':
                    chosen = filedialog.askdirectory(initialdir=current or '.')
                elif dest == 'mask':
                    chosen = filedialog.askopenfilename(initialdir=str(Path(current).parent) if current else '.', filetypes=[('Images', '*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff'), ('All files', '*.*')])
                else:
                    chosen = filedialog.askopenfilename(initialdir=str(Path(current).parent) if current else '.', filetypes=[('Images', '*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff'), ('All files', '*.*')])
                    if not chosen:
                        chosen = filedialog.askdirectory(initialdir=current or '.')
                if chosen:
                    var.set(chosen)
                    self.update_command()
            ttk.Button(holder, text='Browse', command=browse).grid(row=0, column=1, padx=(6, 0))
        holder.var = var
        return holder

    def _value_of(self, dest):
        w = self.widgets[dest]
        return w.var.get()

    def build_argv(self):
        argv = []
        for action in self.parser._actions:
            dest = getattr(action, 'dest', None)
            if not dest or dest == 'help' or dest not in self.widgets:
                continue
            value = self._value_of(dest)
            default = self.defaults.get(dest)
            option = action.option_strings[0] if action.option_strings else None
            if isinstance(action, argparse._StoreTrueAction):
                if value:
                    argv.append(option)
                continue
            if isinstance(action, argparse.BooleanOptionalAction):
                if value != default:
                    argv.append(option if value else '--no-' + dest.replace('_', '-'))
                continue
            if value in ('', None):
                continue
            normalized = str(value)
            if action.type is int:
                normalized = str(int(float(value)))
            elif action.type is float:
                normalized = str(float(value))
            if str(default) != normalized:
                argv.extend([option, normalized])
        return argv

    def update_command(self):
        argv = self.build_argv()
        self.command_var.set('python photo_to_svg.py ' + ' '.join(argv))

    def reset_defaults(self):
        self.defaults = vars(core.parse_args([]))
        for dest, widget in self.widgets.items():
            value = self.defaults.get(dest)
            try:
                widget.var.set(value)
            except Exception:
                pass
        self.update_command()

    def run_command(self):
        script = Path(__file__).with_name('photo_to_svg.py')
        argv = [sys.executable, str(script), *self.build_argv()]
        self.update_command()
        self.log.delete('1.0', 'end')
        self.log.insert('end', '$ ' + ' '.join(argv) + '\n\n')
        self.update()
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, cwd=script.parent)
        except Exception as exc:
            messagebox.showerror('Errore', str(exc))
            return
        if proc.stdout:
            self.log.insert('end', proc.stdout)
        if proc.stderr:
            self.log.insert('end', '\n[stderr]\n' + proc.stderr)
        self.log.see('end')
        if proc.returncode == 0:
            messagebox.showinfo('Fatto', 'Elaborazione completata.')
        else:
            messagebox.showwarning('Attenzione', f'Comando terminato con codice {proc.returncode}.')

if __name__ == '__main__':
    App().mainloop()
