import sys
import tkinter as tk
from matching import match_mesh
from tkinterhtml import HtmlFrame
from urllib.request import urlopen
from PyQt5.QtCore import *
from PyQt5.QtWebEngineWidgets import *
from PyQt5.QtWidgets import QApplication


def main():

    def run_whole_database():
        run_database = int(no_mesh_database.get())
        match_mesh('y', run_database)

    def run_single_mesh():
        run_mesh = int(single_mesh.get())
        mesh = mesh_id.get()
        print(run_mesh)
        match_mesh('n', run_mesh, mesh)
        frame = HtmlFrame(root, horizontal_scrollbar="auto")
        frame.set_content(urlopen("https://duckduckgo.com").read().decode())
    root = tk.Tk()

    width = 500
    height = 750
    canvas1 = tk.Canvas(root, width=width * 2, height=height, relief='raised')
    canvas1.pack()

    label = tk.Label(root, text='CBSR system',font=('helvetica', 20, 'bold'))
    canvas1.create_window(width, height/12, window=label)

    label1_1 = tk.Label(root, text='Running the whole database')
    label1_1.config(font=('helvetica', 14))
    canvas1.create_window(width, height/6, window=label1_1)

    label1_2 = tk.Label(root, text='Comparing to how many mashes?')
    label1_2.config(font=('helvetica', 12))
    canvas1.create_window(width, height/5, window=label1_2)

    no_mesh_database = tk.Entry(root)
    canvas1.create_window(width, height/4, window=no_mesh_database)

    button1 = tk.Button(text='Accuracy of the CBSR system', command=run_whole_database, bg='brown', fg='white',
                        font=('helvetica', 10, 'bold'))
    canvas1.create_window(width, height/3.5, window=button1)

    label2 = tk.Label(root, text='Running one single mesh')
    label2.config(font=('helvetica', 14))
    canvas1.create_window(width, height/2.5, window=label2)

    label2_1 = tk.Label(root, text='What is the mesh id you want to compare?')
    label2_1.config(font=('helvetica', 12))
    canvas1.create_window(width, height/2.25, window=label2_1)

    mesh_id = tk.Entry(root)
    canvas1.create_window(width, height/2, window=mesh_id)

    label2_2 = tk.Label(root, text='Comparing to how many mashes?')
    label2_2.config(font=('helvetica', 12))
    canvas1.create_window(width, height/1.85, window=label2_2)

    single_mesh = tk.Entry(root)
    canvas1.create_window(width, height/1.75, window=single_mesh)

    button2 = tk.Button(text='Compare mesh', command=run_single_mesh, bg='brown', fg='white',
                        font=('helvetica', 10, 'bold'))
    canvas1.create_window(width, height/1.6, window=button2)

    root.mainloop()

