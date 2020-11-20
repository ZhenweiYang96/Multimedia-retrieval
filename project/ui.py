import os
import glob
import tkinter as tk
from tkinter import *
from matching import match_mesh
from urllib.request import urlopen
from PIL import Image, ImageTk
import webbrowser
from xhtml2pdf import pisa
from pdf2image import convert_from_path
import tempfile
from matplotlib import pyplot as plt
from evaluation import *

def convert_pdf(file_path, output_path):
    # save temp image files in temp dir, delete them after we are finished
    # convert pdf to multiple image
    for i in glob.glob("process_image\\*.png"):
        os.remove(i)

    folder = r'C:\Users\Admin\Documents\GitHub\Multimedia-retrieval\proj\process_image'
    images = convert_from_path(file_path, output_folder=folder, poppler_path=r'poppler-\bin')
    # save images to temporary directory
    temp_images = []
    for i in range(len(images)):
        image_path = f'{folder}/{i}.jpg'
        images[i].save(image_path, 'JPEG')
        temp_images.append(image_path)
    # # read images into pillow.Image
    # imgs = list(map(Image.open, temp_images))
    # # find minimum width of images
    # min_img_width = min(i.width for i in imgs)
    # # find total height of all images
    # total_height = 0
    # for i, img in enumerate(imgs):
    #     total_height += imgs[i].height
    # # create new image object with width and total height
    # merged_image = Image.new(imgs[0].mode, (min_img_width, total_height))
    # # paste images together one by one
    # y = 0
    # for img in imgs:
    #     merged_image.paste(img, (0, y))
    #     y += img.height
    # # save merged image
    # merged_image.save(output_path)
    # return output_path


def convertHtmlToPdf(sourceHtml, outputFilename):
    resultFile = open(outputFilename, "w+b")
    pisaStatus = pisa.CreatePDF(sourceHtml, resultFile)
    resultFile.close()
    return pisaStatus.err


def main():
    def run_whole_database():
        for i in glob.glob("process_image\\*.pdf"):
            os.remove(i)
        run_database = int(no_mesh_ev_nm.get())
        match_mesh('y', run_database)
        # webbrowser.open(url='file:///C:/Users/Admin/Documents/GitHub/Multimedia-retrieval/proj/database.pdf')

    def run_single_mesh():
        run_mesh = int(no_mesh_nm.get())
        mesh = mesh_id_nm.get()
        mesh = mesh.split(',')
        match_mesh('n', run_mesh, mesh)
        webbrowser.open(url='file:///C:/Users/Admin/Documents/GitHub/MR/image.html')
        for i in glob.glob("process_image\\*.html"):
            os.remove(i)

    def run_one_mesh_sc():
        run_mesh = int(no_mesh_sc.get())
        print(run_mesh)
        mesh_id = mesh_id_sc.get()
        print(mesh_id)
        mesh_id_list = mesh_id.strip(' ').split(",")
        mesh_id_list = map(int, mesh_id_list)
        scalability(mesh_id_list, run_mesh)
        webbrowser.open("file:///C:/Users/Admin/Documents/GitHub/MR/image_sc.html")

    def run_whole_database_sc():
        run_database = int(no_mesh_ev_sc.get())
        evaluate_scalability(run_database)

    root = tk.Tk()
    width = 500
    height = 1500

    canvas1 = tk.Canvas(root, width=width * 2, height=height, relief='raised')
    canvas1.pack()

    label = tk.Label(root, text='CONTENT-BASED SHAPE RETRIEVAL (CBSR) SYSTEM', font=('helvetica', 20, 'bold'),
                     bg="light grey")
    canvas1.create_window(width, height / 24, window=label)

    label_inst = tk.Label(root, text="Instruction: in this CBSR system, we present two methods for searching similar "
                                     "shapes, normal matching and scalability. The former method utilizes the "
                                     "weighted combination of euclidean distance and earth mover's distance. "
                                     "The later one uses k and r nearest neighbour. Users can "
                                     "implement normal matching in the left part while in the right there is "
                                     "scalability. In both parts, user can input the query shape and the number "
                                     "of most similar shapes returned as well as have a glance at the performance of "
                                     "both methods.", font=("helvetica", 12), wraplength=450
                          )
    canvas1.create_window(width, height / 8, window=label_inst)

    method_1 = tk.Label(root, text="NORMAL MATCHING", relief="solid", font=("helvetica", 16, 'bold'))
    canvas1.create_window(width / 2, height / 5, window=method_1)

    nm_subtitle1 = tk.Label(root, text='Running one single mesh', font=("helvetica", 14, 'bold'))
    canvas1.create_window(width / 2, height / 4.5, window=nm_subtitle1)

    nm_mesh_id = tk.Label(root, text='Which mesh do you want to compare? (1 ~ 260 or 281 ~ 400)',
                          font=('helvetica', 12))
    canvas1.create_window(width / 2, height / 4.2, window=nm_mesh_id)
    mesh_id_nm = tk.Entry(root)
    canvas1.create_window(width / 2, height / 3.8, window=mesh_id_nm)

    nm_num_mesh = tk.Label(root, text='How many similar shapes do you want to return?', font=('helvetica', 12))
    canvas1.create_window(width / 2, height / 3.5, window=nm_num_mesh)
    no_mesh_nm = tk.Entry(root)
    canvas1.create_window(width / 2, height / 3.3, window=no_mesh_nm)
    nm_one_button = Button(root, text="search", font=('helvetica', 11),
                           bg="brown", fg="white", relief="raised", command=run_single_mesh)
    canvas1.create_window(width / 2, height / 3.1, window=nm_one_button)

    nm_subtitle2 = tk.Label(root, text='Evaluating the matching method', font=("helvetica", 14, 'bold'))
    canvas1.create_window(width / 2, height / 2.7, window=nm_subtitle2)

    nm_num_mesh_ev = tk.Label(root, text='How many similar shapes do you want to return?', font=('helvetica', 12))
    canvas1.create_window(width / 2, height / 2.5, window=nm_num_mesh_ev)
    no_mesh_ev_nm = tk.Entry(root)
    canvas1.create_window(width / 2, height / 2.4, window=no_mesh_ev_nm)
    nm_all_button = Button(root, text="Evaluation", font=('helvetica', 11),
                           bg="brown", fg="white", relief="raised", command=run_whole_database)
    canvas1.create_window(width / 2, height / 2.28, window=nm_all_button)

    method_2 = tk.Label(root, text="SCALABILITY", relief="solid", font=("helvetica", 16, 'bold'))
    canvas1.create_window(width * 1.5, height / 5, window=method_2)

    sc_subtitle1 = tk.Label(root, text='Running one single mesh', font=("helvetica", 14, 'bold'))
    canvas1.create_window(width * 1.5, height / 4.5, window=sc_subtitle1)

    sc_mesh_id = tk.Label(root, text='Which mesh do you want to compare? (1 ~ 260 or 281 ~ 400)',
                          font=('helvetica', 12))
    canvas1.create_window(width * 1.5, height / 4.2, window=sc_mesh_id)
    mesh_id_sc = tk.Entry(root)
    canvas1.create_window(width * 1.5, height / 3.8, window=mesh_id_sc)
    sc_num_mesh = tk.Label(root, text='How many similar shapes do you want to return?', font=('helvetica', 12))
    canvas1.create_window(width * 1.5, height / 3.5, window=sc_num_mesh)
    no_mesh_sc = tk.Entry(root)
    canvas1.create_window(width * 1.5, height / 3.3, window=no_mesh_sc)
    sc_one_button = Button(root, text="search", font=('helvetica', 11),
                           bg="brown", fg="white", relief="raised", command=run_one_mesh_sc)
    canvas1.create_window(width * 1.5, height / 3.1, window=sc_one_button)

    sc_subtitle2 = tk.Label(root, text='Evaluating the matching method', font=("helvetica", 14, 'bold'))
    canvas1.create_window(width * 1.5, height / 2.7, window=sc_subtitle2)

    sc_num_mesh_ev = tk.Label(root, text='How many similar shapes do you want to return?', font=('helvetica', 12))
    canvas1.create_window(width * 1.5, height / 2.5, window=sc_num_mesh_ev)
    no_mesh_ev_sc = tk.Entry(root)
    canvas1.create_window(width * 1.5, height / 2.4, window=no_mesh_ev_sc)
    sc_all_button = Button(root, text="Evaluation", font=('helvetica', 11),
                           bg="brown", fg="white", relief="raised", command=run_whole_database_sc)
    canvas1.create_window(width * 1.5, height / 2.28, window=sc_all_button)
    # label1_1 = tk.Label(root, text='Running the whole database',font=('helvetica', 14))
    # canvas1.create_window(width, height/6, window=label1_1)
    # button1 = tk.Button(text='Accuracy of the CBSR system', command=run_whole_database, bg='brown', fg='white',
    #                    font=('helvetica', 10, 'bold'))
    # canvas1.create_window(width, height/3.5, window=button1)
    # button2 = tk.Button(text='Compare mesh', command=run_single_mesh, bg='brown', fg='white',
    #                   font=('helvetica', 10, 'bold'))
    # canvas1.create_window(width, height/1.6, window=button2)
    ###To show the image
    root.mainloop()
