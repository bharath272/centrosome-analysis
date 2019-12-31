import tkinter
from tkinter import ttk
from tkinter.filedialog import askopenfilename, asksaveasfilename
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np
import tifffile
import json
import sys
import os
from matplotlib.collections import PatchCollection
import argparse
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import distance_transform_edt
import ml_foci_detect
import ml_cell_segmentation
import torch
from tkinter import messagebox
import centrosome_analysis_backend
import cell_segmentation

def get_colors():
    colors = [[0,1,0],\
    [1,0,0],\
    [0,1,1],\
    [1,0,1],\
    [1,1,0],\
    [1,1,1],\
    [0,0,0.5],\
    [0,0.5,0],\
    [0.5,0,0],\
    [0,1,0.5],\
    [0,0.5,1],\
    [1,0,0.5],\
    [0.5,0,1],\
    [1,0.5,0],\
    [0.5,1,0],\
    [1,1,0.5],\
    [1,0.5,1],\
    [0.5,1,1]]

    colors = np.array(colors)
    colors = np.concatenate((colors, np.random.rand(1000,3)),axis=0)
    return colors



def normalize_for_display(img):
    img = img.astype(float).transpose(1,2,0)
    for i in range(img.shape[2]):
        img[:,:,i] = (img[:,:,i] - np.min(img[:,:,i]))/(np.max(img[:,:,i])-np.min(img[:,:,i])+1e-5)
    return img
def normalize_channel(img):
    img = img.astype(float)
    return (img-np.min(img))/(np.max(img)-np.min(img)+1e-5)
def get_single_channel_img(channel):
    img = np.zeros((channel.shape[0],channel.shape[1],3),dtype=np.uint8)
    for i in range(3):
        img[:,:,i] = (channel*255).astype(np.uint8)
    return img
def false_color_img(colors, channels):
    img = np.zeros((channels[0].shape[0],channels[0].shape[1],3))
    for i, c in enumerate(channels):
        for j in range(3):
            img[:,:,j] = np.minimum(img[:,:,j]+colors[i][j]*channels[i],1)
    return (img*255).astype(np.uint8)
def get_imgs_for_display(img):
    #channels
    channels = [normalize_channel(img[i,:,:]) for i in range(img.shape[0])]
    #single channel images
    imgs = [get_single_channel_img(channel) for channel in channels]
    #false color
    colors = [[1,0,0],[0,1,0],[0,0,1]]
    false_colors = ['c','m','y','w']
    colors.extend([matplotlib.colors.to_rgb(c) for c in false_colors])
    fimg = false_color_img(colors, channels)
    return imgs, fimg





def bresenham_line_h(pt0, d, b):
    max_deltax = int(np.ceil(d[0]*b))
    m = d[1]/d[0]
    coords = np.zeros((max_deltax+1,2), dtype=int)
    coords[0,:] = pt0
    ii = pt0[0]
    jj = pt0[1]
    error = 0
    for i in range(max_deltax):
        coords[i+1,0] = ii
        coords[i+1,1] = jj
        ii = ii+1
        error = error + m
        if error>0.5:
            error = error - 1
            jj = jj + 1
    return coords





def bresenham_line(startpoint, endpoint):
    direction = endpoint-startpoint
    startpoint = np.round(startpoint).astype(int)
    endpoint = np.round(endpoint).astype(int)

    maxval = np.linalg.norm(direction)
    direction = direction / np.linalg.norm(direction)
    swap_i_and_j = negate_i = negate_j = False
    if abs(direction[1])>abs(direction[0]):
        swap_i_and_j = True
    if direction[0]<0:
        negate_i = True
    if direction[1]<0:
        negate_j = True

    startpoint_h = startpoint.copy()
    direction_h = direction.copy()
    if negate_i:
        startpoint_h[0] = -startpoint_h[0]
        direction_h[0] = -direction_h[0]
    if negate_j:
        startpoint_h[1] = -startpoint_h[1]
        direction_h[1] = -direction_h[1]
    if swap_i_and_j:
        startpoint_h = startpoint_h[::-1]
        direction_h = direction_h[::-1]

    coords = bresenham_line_h(startpoint_h, direction_h, maxval)

    if swap_i_and_j:
        coords = coords[:,::-1]
    if negate_i:
        coords[:,0] = -coords[:,0]
    if negate_j:
        coords[:,1] = -coords[:,1]
    return coords


class ChooseChannelOrder:
    def __init__(self, parent):
        self.top = tkinter.Toplevel(parent)
        self.parent = parent

        self.channel1var = tkinter.StringVar()
        self.channel1var.set('1')
        self.channel2var = tkinter.StringVar()
        self.channel2var.set('2')
        self.channel3var = tkinter.StringVar()
        self.channel3var.set('3')
        self.scalevar = tkinter.StringVar()
        self.scalevar.set('0.13')


        self.channel1 = tkinter.Frame(self.top)
        self.channel1.pack(side=tkinter.TOP)
        self.channel1label = tkinter.Label(self.channel1, text='Which channel (1,2 or 3) has pericentriolar material?')
        self.channel1label.pack(side=tkinter.LEFT)
        self.channel1entry = tkinter.Entry(self.channel1, textvariable=self.channel1var)
        self.channel1entry.pack(side=tkinter.LEFT)

        self.channel2 = tkinter.Frame(self.top)
        self.channel2.pack(side=tkinter.TOP)
        self.channel2label = tkinter.Label(self.channel2, text='Which channel (1,2 or 3) has centrin?')
        self.channel2label.pack(side=tkinter.LEFT)
        self.channel2entry = tkinter.Entry(self.channel2,  textvariable=self.channel2var)
        self.channel2entry.pack(side=tkinter.LEFT)


        self.channel3 = tkinter.Frame(self.top)
        self.channel3.pack(side=tkinter.TOP)
        self.channel3label = tkinter.Label(self.channel3, text='Which channel (1,2 or 3) has DAPI?')
        self.channel3label.pack(side=tkinter.LEFT)
        self.channel3entry = tkinter.Entry(self.channel3, textvariable=self.channel3var)

        self.channel3entry.pack(side=tkinter.LEFT)

        self.scaleframe = tkinter.Frame(self.top)
        self.scaleframe.pack(side=tkinter.TOP)
        self.scalelabel = tkinter.Label(self.scaleframe, text='1 pixel = how many microns?')
        self.scalelabel.pack(side=tkinter.LEFT)
        self.scaleentry = tkinter.Entry(self.scaleframe, textvariable=self.scalevar)
        self.scaleentry.pack(side=tkinter.LEFT)

        self.okbutton = tkinter.Button(self.top, text="OK", command=self.done)
        self.okbutton.pack(side=tkinter.TOP)
        self.top.grab_set()
        self.channel1entry.focus_set()
        self.top.protocol("WM_DELETE_WINDOW", self.done)
        self.top.wait_window(self.top)


    def done(self):
        self.result = [int(self.channel1var.get()), int(self.channel2var.get()), int(self.channel3var.get())]
        self.result = [x-1 for x in self.result]
        self.scaleresult = float(self.scalevar.get())
        self.parent.focus_set()
        self.top.destroy()








class AnalysisGUI:
    def __init__(self, master):
        #general parameters
        self.should_load = False
        self.mean = 118
        self.std = 23
        #self.radii = 0.13*np.arange(1,41)
        self.bgradii = np.array([7,8,9])
        #self.eb3radii = 0.13*np.arange(20,121,20)


        self.master = master

        #variables
        self.modevar = tkinter.StringVar()
        self.modevar.set("")
        self.mode = None

        self.det_thresh = 0.7
        self.det_thresh_var = tkinter.StringVar()
        self.det_thresh_var.set(self.det_thresh)

        self.cell_probability_thresh=0.2
        self.cell_thresh_var = tkinter.StringVar()
        self.cell_thresh_var.set(self.cell_probability_thresh)

        self.show_cell_bdry_var = tkinter.IntVar()
        self.show_cell_bdry_var.set(1)
        self.show_cell_boundaries = 1
        self.show_detections_var = tkinter.IntVar()
        self.show_detections_var.set(1)
        self.show_dets=1
        self.show_channel1_var = tkinter.IntVar()
        self.show_channel1_var.set(1)
        self.show_channel1=1
        self.show_channel2_var = tkinter.IntVar()
        self.show_channel2_var.set(1)
        self.show_channel2=1
        self.show_channel3_var = tkinter.IntVar()
        self.show_channel3_var.set(1)
        self.show_channel3 = 1
        self.show_channel4_var = tkinter.IntVar()
        self.show_channel4_var.set(1)
        self.show_channel4 = 1

        self.pcm_channel_var=tkinter.StringVar(self.master)
        self.pcm_channel_var.set("None")
        self.centrin_channel_var=tkinter.StringVar(self.master)
        self.centrin_channel_var.set("None")
        self.mt_channel_var=tkinter.StringVar(self.master)
        self.mt_channel_var.set("None")
        self.eb3_channel_var=tkinter.StringVar(self.master)
        self.eb3_channel_var.set("None")
        self.dapi_channel_var=tkinter.StringVar(self.master)
        self.dapi_channel_var.set("None")
        self.channel_choices=["Absent","1","2","3","4"]

        self.scalevar = tkinter.StringVar(self.master)
        self.scalevar.set('0.13')

        self.eb3radiusvar = tkinter.StringVar(self.master)
        self.eb3radiusvar.set('16')
        self.eb3stepvar = tkinter.StringVar(self.master)
        self.eb3stepvar.set('2.6')

        self.pcmradiusvar = tkinter.StringVar(self.master)
        self.pcmradiusvar.set('6')
        self.pcmstepvar = tkinter.StringVar(self.master)
        self.pcmstepvar.set('0.13')


        self.display_image_var = tkinter.StringVar(self.master)
        self.display_image_var.set("All (False color: Red, Green, Blue, Cyan)")
        self.display_choices=["Channel 1 only", "Channel 2 only", \
                        "Channel 3 only", "Channel 4 only", \
                        "All (False color: Red, Green, Blue, Cyan)",\
                        "Thresholded EB3 (Cell body)",
                        "EB3 Foci"]

        self.analysis_var = tkinter.StringVar(self.master)
        self.analysis_var.set("Density of MT")
        self.analysis_choices=["Density of MT", "Density of PCM", "EB3 foci density"]

        self.rel_thresh_var=tkinter.StringVar(self.master)
        self.rel_thresh_var.set("0.01")
        self.length_thresh_var=tkinter.StringVar(self.master)
        self.length_thresh_var.set("20")
        self.abs_thresh_var=tkinter.StringVar(self.master)
        self.abs_thresh_var.set("0.1")
        self.save_eb3_vis_var = tkinter.IntVar(self.master)
        self.save_eb3_vis_var.set(0)

        #Canvas
        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(side=tkinter.LEFT, expand=1)
        self.zoom_bbox = None
        self.zoom_rect = None
        c1,c2,c3 = np.unravel_index(np.arange(1000), (10,10,10))
        c1 = c1.reshape((-1,1))+10
        c2 = c2.reshape((-1,1))+10
        c3 = c3.reshape((-1,1))+10
        self.colors = get_colors()

        #self.draw(redraw_img=True)
        self.canvas.mpl_connect("button_press_event", self.__getattribute__('on_button_press'))
        self.canvas.mpl_connect("button_release_event", self.__getattribute__('on_button_release'))
        self.canvas.mpl_connect('motion_notify_event', self.__getattribute__('on_mouse_move'))



        self.set_styles()
        self.menubar = tkinter.Menu(master=self.master)

        self.master.config(menu=self.menubar)
        self.create_file_menu()
        self.master_toolbar = ttk.Frame(master=self.master, style="BW.TFrame")
        self.master_toolbar.pack(side=tkinter.RIGHT)
        self.toolbar_count=0

        self.modelabel = ttk.Label(master=self.master_toolbar, textvariable=self.modevar, style="TLabel")
        self.modelabel.grid(row=0, column=1)
        self.run_ml_model_button = ttk.Button(master=self.master_toolbar, text="Run ML models", command=self.run_ml_models, style="TButton")
        self.run_ml_model_button.grid(row=0, column=0, sticky="W")



        #zoom toolbar
        self.create_zoom_toolbar(2,0)


        #cell segmentation toolbar
        self.create_cell_segmentation_toolbar(2,1)

        #foci toolbar
        self.create_foci_toolbar(3,0)

        #amplified toolbar
        self.create_amplified_toolbar(3,1)

        #vis toolbar
        self.create_vis_toolbar(4)

        self.create_channel_toolbar(5)




        self.create_parameter_toolbar(6)


        self.save_file = None
        self.foci_model = None
        self.ml_cell_segmentation_model = None
        self.ml_model_has_run=False
        tkinter.mainloop()


    def set_styles(self):
        self.style=ttk.Style()
        self.style.configure("TFrame",  borderwidth=3, relief=tkinter.GROOVE)
        self.style.configure("BW.TFrame",  bg="white",borderwidth=0, relief=tkinter.GROOVE)

        self.style.configure("Italic.TLabel", font='Helvetica 14 italic' )
        self.style.configure("ItalicU.TLabel", font='Helvetica 14 italic underline' )

        self.style.configure("TLabel", font='Helvetica 14' )

        self.style.configure("TButton", borderwidth=3, relief=tkinter.RAISED )

    def create_file_menu(self):
        self.file_menu = tkinter.Menu(master=self.menubar, tearoff=0)
        self.file_menu.add_command(label="Analyze and save", command=self._save)
        self.file_menu.add_command(label="Save corrections", command=self._save_foci_cellmap)
        self.file_menu.add_command(label="Open image", command=self._open)
        self.file_menu.add_command(label="Load Centrosome Detector", command=self.load_foci_model)
        self.file_menu.add_command(label="Load Cell Model", command=self.load_cell_model)
        self.file_menu.add_command(label="Close",command=self._quit)
        self.menubar.add_cascade(label="File",menu = self.file_menu)

    def create_zoom_toolbar(self, r, c):
        self.zoom_toolbar = ttk.Frame(master=self.master_toolbar, style="TFrame",padding=5)
        self.zoom_toolbar.grid(row=r,column=c, sticky="W,E,N,S")
        self.zoom_label = ttk.Label(master=self.zoom_toolbar, text="Zoom", style="Italic.TLabel")
        self.zoom_label.grid(row=0,column=0,sticky="W")
        self.zoombutton = ttk.Button(master=self.zoom_toolbar, text="Zoom", command=self.set_zoom_mode, style="TButton")
        self.zoombutton.grid(row=1,column=0)
        self.resetbutton = ttk.Button(master=self.zoom_toolbar, text="Reset zoom", command=self.reset_zoom, style="TButton")
        self.resetbutton.grid(row=1,column=1)


    def create_cell_segmentation_toolbar(self,r,c):
        self.cell_segmentation_toolbar = ttk.Frame(master=self.master_toolbar, style="TFrame", padding=5)
        self.cell_segmentation_toolbar.grid(row=r,column=c, sticky="W,E")
        self.cell_seg_label = ttk.Label(master=self.cell_segmentation_toolbar, text="Cell manipulation", style="Italic.TLabel")
        self.cell_seg_label.grid(row=0,column=0,sticky="W")

        self.drawcellbutton = ttk.Button(master=self.cell_segmentation_toolbar, text="Add cell boundary", command=self.set_draw_mode, style="TButton")
        self.drawcellbutton.grid(row=1,column=0, sticky="W")
        self.removecellbutton = ttk.Button(master=self.cell_segmentation_toolbar, text="Remove cell from analysis", command=self.set_remove_cell_mode, style="TButton")
        self.removecellbutton.grid(row=1,column=1,sticky="W")

        self.mergecellbutton = ttk.Button(master=self.cell_segmentation_toolbar, text="Choose cells to merge", command=self.set_merge_mode, style="TButton")
        self.mergecellbutton.grid(row=2,column=0,sticky="W")
        self.mergecellfinishbutton = ttk.Button(master=self.cell_segmentation_toolbar, text="Merge cells", command=self.merge_cells, style="TButton")
        self.mergecellfinishbutton.grid(row=2,column=1, sticky="W")

    def create_foci_toolbar(self,r,c):
        self.foci_toolbar = ttk.Frame(master=self.master_toolbar, style="TFrame", padding=5)
        self.foci_toolbar.grid(row=r,column=c, sticky="W,E")
        self.foci_label = ttk.Label(master=self.foci_toolbar, text="Foci", style="Italic.TLabel")
        self.foci_label.grid(row=0,column=0, sticky="W")

        self.correctbutton = ttk.Button(master=self.foci_toolbar, text="Remove foci", command=self.set_correct_mode, style="TButton")
        self.correctbutton.grid(row=1,column=0, sticky="W")
        self.addbutton = ttk.Button(master=self.foci_toolbar, text="Add centrosomes", command=self.set_add_mode,style="TButton")
        self.addbutton.grid(row=1,column=1)

    def create_amplified_toolbar(self,r,c):
        self.amplified_toolbar = ttk.Frame(master=self.master_toolbar, style="TFrame", padding=5)
        self.amplified_toolbar.grid(row=r,column=c, sticky="W,E")
        self.amplified_label = ttk.Label(master=self.amplified_toolbar, text="Amplified?", style="Italic.TLabel")
        self.amplified_label.grid(row=0,column=0, sticky="W")
        self.amplifiedbutton = ttk.Button(master=self.amplified_toolbar, text="Mark cells as amplified", command=self.set_amplified_mode, style="TButton")
        self.amplifiedbutton.grid(row=1,column=0, sticky="W")
        self.notamplifiedbutton = ttk.Button(master=self.amplified_toolbar, text="Mark cells as not amplified", command=self.set_notamplified_mode, style="TButton")
        self.notamplifiedbutton.grid(row=1, column=1, sticky="W")


    def create_vis_toolbar(self,r):
        self.vis_toolbar = ttk.Frame(master=self.master_toolbar, style="TFrame", padding=5)
        self.vis_toolbar.grid(row=r, columnspan=2, sticky="W,E")
        self.vis_label = ttk.Label(master=self.vis_toolbar, text="Visualization", style="Italic.TLabel")
        self.vis_label.grid(row=0,column=0, sticky="W")

        self.togglecellbutton = ttk.Checkbutton(master=self.vis_toolbar, variable=self.show_cell_bdry_var, text="Show cell boundary", command=self.toggle_cell_boundary)
        self.togglecellbutton.grid(row=1,column=0, sticky="W")
        self.toggledetbutton = ttk.Checkbutton(master=self.vis_toolbar, variable=self.show_detections_var, text="Show centrosomes", command=self.toggle_dets)
        self.toggledetbutton.grid(row=1,column=1, sticky="W")

        # self.togglechannel1button = tkinter.Checkbutton(master=self.vis_toolbar, variable=self.show_channel1_var, text="Show channel 1", command=self.toggle_channel1)
        # self.togglechannel1button.grid(row=2,column=0, sticky="W")
        # self.togglechannel2button = tkinter.Checkbutton(master=self.vis_toolbar, variable=self.show_channel2_var, text="Show channel 2", command=self.toggle_channel2)
        # self.togglechannel2button.grid(row=2,column=1, sticky="W")
        # self.togglechannel3button = tkinter.Checkbutton(master=self.vis_toolbar, variable=self.show_channel3_var, text="Show channel 3", command=self.toggle_channel3)
        # self.togglechannel3button.grid(row=2,column=2, sticky="W")
        # self.togglechannel4button = tkinter.Checkbutton(master=self.vis_toolbar, variable=self.show_channel3_var, text="Show channel 3", command=self.toggle_channel3)
        # self.togglechannel4button.grid(row=2,column=3, sticky="W")

        self.display_choose_label = ttk.Label(master=self.vis_toolbar, text="Display:")
        self.display_choose_label.grid(row=2,column=1,sticky="E")
        self.display_choose = ttk.OptionMenu(self.vis_toolbar, self.display_image_var, self.display_image_var.get(),*self.display_choices)
        self.display_choose.grid(row=2,column=2,sticky="W")

        self.redraw_button = ttk.Button(master=self.vis_toolbar, text="Update display", command=self.redraw, style="TButton")
        self.redraw_button.grid(row=2,column=0,sticky="W")


    def create_parameter_toolbar(self,r):
        self.parameter_toolbar = ttk.Frame(master=self.master_toolbar, style="TFrame", padding=5)
        self.parameter_toolbar.grid(row=r, columnspan=2,sticky="W,E")
        self.parameter_label = ttk.Label(master=self.parameter_toolbar, text="Parameters", style="Italic.TLabel")
        self.parameter_label.grid(row=0,column=0, sticky="W")
        self.numdetlabel = ttk.Label(master=self.parameter_toolbar, text="Centrosomal foci threshold")
        self.numdetlabel.grid(row=1,column=0, columnspan=2, sticky="W")
        self.numdetentry = ttk.Entry(master=self.parameter_toolbar, textvariable=self.det_thresh_var, width=5)
        self.numdetentry.grid(row=1,column=2, sticky="W")
        self.numdetset = ttk.Button(master=self.parameter_toolbar, text="Set", command=self.set_det_thresh, style="TButton")
        self.numdetset.grid(row=1,column=3, sticky="W")

        self.cellthreshlabel = ttk.Label(master=self.parameter_toolbar, text="Cell threshold")
        self.cellthreshlabel.grid(row=2,column=0, columnspan=2,sticky="W")
        self.cellthreshentry = ttk.Entry(master=self.parameter_toolbar, textvariable=self.cell_thresh_var, width=5)
        self.cellthreshentry.grid(row=2,column=2, sticky="W")
        self.cellthreshset = ttk.Button(master=self.parameter_toolbar, text="Set", command=self.set_cell_thresh, style="TButton")
        self.cellthreshset.grid(row=2,column=3, sticky="W")
        config=self.parameter_toolbar.config()


        self.analysislabel = ttk.Label(self.parameter_toolbar, text="Which type of analysis?")
        self.analysislabel.grid(row=1,column=5,sticky="E")
        self.analysisentry = ttk.OptionMenu(self.parameter_toolbar, self.analysis_var, self.analysis_var.get(),*self.analysis_choices)
        self.analysisentry.grid(row=1,column=6,sticky="W")
        self.scalelabel = ttk.Label(self.parameter_toolbar, text='1 pixel = how many microns?')
        self.scalelabel.grid(row=2,column=5, sticky="E")
        self.scaleentry = ttk.Entry(self.parameter_toolbar, textvariable=self.scalevar, width=5)
        self.scaleentry.grid(row=2, column=6, sticky="W")
        self.scaleset = ttk.Button(master=self.parameter_toolbar, text="Set", command=self.set_scale, style="TButton")
        self.scaleset.grid(row=2,column=7, sticky="w")

        self.radiiframe = ttk.Frame(master=self.parameter_toolbar, padding=5,style="BW.TFrame")
        self.radiiframe.grid(row=3,columnspan=8, sticky="W,E")
        self.eb3label = ttk.Label(self.radiiframe, text="EB3/Microtubule density analysis:", style='ItalicU.TLabel')
        self.eb3label.grid(row=0,columnspan=4,sticky="W")
        self.eb3radiuslabel = ttk.Label(self.radiiframe, text="Max radius (microns)=")
        self.eb3radiuslabel.grid(row=1,column=0,sticky="E")
        self.eb3radiusentry = ttk.Entry(self.radiiframe, textvariable=self.eb3radiusvar, width=5)
        self.eb3radiusentry.grid(row=1, column=1, sticky="W")
        self.eb3steplabel = ttk.Label(self.radiiframe, text="Step (microns)=")
        self.eb3steplabel.grid(row=1,column=2,sticky="E")
        self.eb3stepentry = ttk.Entry(self.radiiframe, textvariable=self.eb3stepvar, width=5)
        self.eb3stepentry.grid(row=1, column=3, sticky="W")
        self.rel_thresh_label = ttk.Label(self.radiiframe, text="Relative EB3 foci intensity threshold:")
        self.rel_thresh_label.grid(row=2, column=0,columnspan=3, sticky="W")
        self.rel_thresh_entry = ttk.Entry(self.radiiframe, textvariable=self.rel_thresh_var, width=5)
        self.rel_thresh_entry.grid(row=2, column=3,sticky="W")
        self.abs_thresh_label = ttk.Label(self.radiiframe, text="Absolute intensity threshold for cell area:")
        self.abs_thresh_label.grid(row=3, column=0,columnspan=3, sticky="W")
        self.abs_thresh_entry = ttk.Entry(self.radiiframe, textvariable=self.abs_thresh_var, width=5)
        self.abs_thresh_entry.grid(row=3, column=3,sticky="W")
        self.length_thresh_label = ttk.Label(self.radiiframe, text="Length threshold for EB3 foci:")
        self.length_thresh_label.grid(row=4, column=0,columnspan=3, sticky="W")
        self.length_thresh_entry = ttk.Entry(self.radiiframe, textvariable=self.length_thresh_var, width=5)
        self.length_thresh_entry.grid(row=4, column=3,sticky="W")
        self.toggleeb3button = ttk.Checkbutton(master=self.radiiframe, variable=self.save_eb3_vis_var, text="Save EB3 visualization")
        self.toggleeb3button.grid(row=5,column=0, columnspan=3, sticky="W")


        self.pcmlabel = ttk.Label(self.radiiframe, text="PCM density analysis:", style='ItalicU.TLabel')
        self.pcmlabel.grid(row=0,column=4, columnspan=4,sticky="W")
        self.pcmradiuslabel = ttk.Label(self.radiiframe, text="Max radius (microns)=")
        self.pcmradiuslabel.grid(row=1,column=4,sticky="E")
        self.pcmradiusentry = ttk.Entry(self.radiiframe, textvariable=self.pcmradiusvar, width=5)
        self.pcmradiusentry.grid(row=1, column=5, sticky="W")
        self.pcmsteplabel = ttk.Label(self.radiiframe, text="Step (microns)=")
        self.pcmsteplabel.grid(row=1,column=6,sticky="E")
        self.pcmstepentry = ttk.Entry(self.radiiframe, textvariable=self.pcmstepvar, width=5)
        self.pcmstepentry.grid(row=1, column=7, sticky="W")

        self.analyzeandsavebutton = ttk.Button(master=self.radiiframe, text="Analyze and save", command=self._save)
        self.analyzeandsavebutton.grid(row=2, column=4, sticky="W")






    def create_channel_toolbar(self,r):
        self.channel_toolbar = ttk.Frame(master=self.master_toolbar, style="TFrame", padding=5)
        self.channel_toolbar.grid(row=r,columnspan=2, sticky="W,E")
        self.channel_label = ttk.Label(master=self.channel_toolbar, text="Channels",style="Italic.TLabel")
        self.channel_label.grid(row=0,sticky="W")

        self.pcm_choose_label = ttk.Label(master=self.channel_toolbar, text="PCM:")
        self.pcm_choose_label.grid(row=1,column=0,sticky="E")
        self.pcm_choose = ttk.OptionMenu(self.channel_toolbar, self.pcm_channel_var, self.pcm_channel_var.get(), *self.channel_choices)
        self.pcm_choose.grid(row=1,column=1,sticky="W")

        self.centrin_choose_label = ttk.Label(master=self.channel_toolbar, text="Centrin:")
        self.centrin_choose_label.grid(row=1,column=2,sticky="E")
        self.centrin_choose = ttk.OptionMenu(self.channel_toolbar, self.centrin_channel_var, self.centrin_channel_var.get(), *self.channel_choices)
        self.centrin_choose.grid(row=1,column=3,sticky="W")

        self.mt_choose_label = ttk.Label(master=self.channel_toolbar, text="MT:")
        self.mt_choose_label.grid(row=1,column=4,sticky="E")
        self.mt_choose = ttk.OptionMenu(self.channel_toolbar, self.mt_channel_var, self.mt_channel_var.get(), *self.channel_choices)
        self.mt_choose.grid(row=1,column=5,sticky="W")

        self.eb3_choose_label = ttk.Label(master=self.channel_toolbar, text="EB3:")
        self.eb3_choose_label.grid(row=2,column=0,sticky="E")
        self.eb3_choose = ttk.OptionMenu(self.channel_toolbar, self.eb3_channel_var, self.eb3_channel_var.get(),*self.channel_choices)
        self.eb3_choose.grid(row=2,column=1,sticky="W")

        self.dapi_choose_label = ttk.Label(master=self.channel_toolbar, text="DAPI:")
        self.dapi_choose_label.grid(row=2,column=2,sticky="Es")
        self.dapi_choose = ttk.OptionMenu(self.channel_toolbar, self.dapi_channel_var, self.dapi_channel_var.get(),*self.channel_choices)
        self.dapi_choose.grid(row=2,column=3,sticky="W")


    def get_all_boundaries(self):
        cell_map = self.cell_map
        unique_cells = np.unique(cell_map[cell_map!=0])
        self.boundaries = {}
        for l in unique_cells:
            D = distance_transform_edt(cell_map==l)
            L = ((D>0) & (D<2))
            self.boundaries[l] = np.where(L)


    def render_boundary_helper(self, bmap, color, points, margin):

        for i in range(-margin, margin):
            for j in range(-margin,margin):
                I = points[0] + i
                J = points[1] + j
                idx = (I>=0) & (I<bmap.shape[0]) & (J>=0) & (J<bmap.shape[1])
                for c in range(3):
                    bmap[I[idx], J[idx], c] = color[c]
        return bmap

    def render_boundary(self):
        labelids = self.labelids
        amplified = self.amplified
        bmap = np.zeros((self.cell_map.shape[0], self.cell_map.shape[1],3))
        for i,l in enumerate(labelids):
            if l==0:
                continue
            if amplified[i]:
                margin=15
            else:
                margin=5
            bmap = self.render_boundary_helper(bmap, self.colors[i,:],self.boundaries[l],margin)

        return bmap

    def load_foci_model(self):
        foci_model_file = askopenfilename(initialdir='.')
        if foci_model_file!='':
            self.foci_model = centrosome_analysis_backend.load_foci_model(foci_model_file)
            messagebox.showinfo('','Centrosome detection model loaded')

    def load_cell_model(self):
        cell_model_file = askopenfilename(initialdir='.')
        if cell_model_file!='':
            self.ml_cell_segmentation_model = centrosome_analysis_backend.load_cell_model(cell_model_file)
            messagebox.showinfo('','Cell segmentation model loaded')

    def run_detection_model(self, img):
        if self.foci_model is None:
            messagebox.showerror('Error','Foci detection model not loaded')
            return
        foci, foci_scores = centrosome_analysis_backend.run_detection_model(img, self.foci_model, self.mean, self.std)
        return foci, foci_scores
    def run_cell_model(self, img):
        if self.ml_cell_segmentation_model is None:
            messagebox.showerror('Error','Cell segmentation model not loaded')
            return
        cell_prob, cell_bmap = centrosome_analysis_backend.run_cell_model(img, self.ml_cell_segmentation_model, self.mean, self.std)
        return cell_prob, cell_bmap

    def _open(self):
        self.open_file= askopenfilename()
        imgname = self.open_file
        img = tifffile.imread(imgname)
        self.img = img
        img_stem = os.path.splitext(imgname)[0]
        self.json_file = img_stem + '-corrected.json'
        self.cell_map_file = img_stem + '-corrected.npz'
        detname = self.json_file
        cellname = self.cell_map_file

        if not os.path.isfile(detname):
            detname = img_stem+'.json'
        if os.path.isfile(detname) and self.should_load:
            with open(detname, 'r') as f:
                dets = json.load(f)
            foci = np.array(dets['foci'])
            foci_scores = np.array(dets['foci_scores'])
            foci = np.round(foci).astype(int)
        else:
            foci = np.zeros((0,2))
            foci_scores = np.array([])
            #foci, foci_scores = self.run_detection_model(img)

        if not os.path.isfile(cellname):
            cellname = img_stem+'.npz'
        if os.path.isfile(cellname) and self.should_load:
            a = np.load(cellname)
            cellprobs = a['cell_probs']
            cell_bmap = a['cell_bmap']
        else:
            cellprobs = np.zeros((img.shape[1],img.shape[2]))
            cell_bmap = np.zeros((img.shape[1], img.shape[2]))
            #cellprobs, cell_bmap = self.run_cell_model(img)

        self.detections = foci
        self.detection_scores = foci_scores
        self.new_points = None
        self.labels = np.ones(foci.shape[0])
        self.cell_probabilities = cellprobs
        self.cell_bmap = cell_bmap


        self.ml_model_has_run = False
        self.analysis_outcomes = None
        #temp = ChooseChannelOrder(self.master)
        #channel_order = temp.result

        #img = img[channel_order,:,:]




        self.xlim = [0, self.img.shape[2]]
        self.ylim = [0, self.img.shape[1]]

        self.display_channels, self.display_falsecolor = get_imgs_for_display(img)
        #self.imgdisp = self.imgdisp_orig.copy()

        self.draw(redraw_img=True)


    def run_ml_models(self):
        chosen=True
        if self.pcm_channel_var.get()=='None':
            tkinter.messagebox.showerror('Error','Please choose channel containing PCM!')
            chosen=False
        if self.centrin_channel_var.get()=='None':
            tkinter.messagebox.showerror('Error', 'Please choose channel containing Centrin!')
            chosen=False
        if self.dapi_channel_var.get()=='None':
            tkinter.messagebox.showerror('Error','Please choose channel containing DAPI!')
            chosen=False

        if not chosen:
            return
        pcm_channel = int(self.pcm_channel_var.get())-1
        centrin_channel = int(self.centrin_channel_var.get())-1
        dapi_channel = int(self.dapi_channel_var.get())-1
        #img_analysis = self.img[[centrin_channel, pcm_channel, dapi_channel],:,:]
        img_analysis = self.img[[pcm_channel, centrin_channel, dapi_channel],:,:]
        self.modevar.set('Running foci detection')
        foci, foci_scores = self.run_detection_model(img_analysis)
        self.modevar.set('Estimating cell bodies')
        cellprobs, cell_bmap = self.run_cell_model(img_analysis)
        self.detections = foci
        self.detection_scores = foci_scores
        self.new_points = None
        self.labels = np.ones(foci.shape[0])
        self.cell_probabilities = cellprobs
        self.cell_bmap = cell_bmap
        self.ml_model_has_run=True
        self.modevar.set('Segmenting cells')
        self.do_cell_analysis()
        self.redraw()


    def do_cell_analysis(self, reestimate_boundaries=True):
        if not self.ml_model_has_run:
            tkinter.messagebox.showerror('Error','Run ML model first!')
            return

        if reestimate_boundaries:
            self.cell_map = centrosome_analysis_backend.get_cell_map(self.cell_probabilities, self.cell_bmap, self.cell_probability_thresh)
            self.get_all_boundaries()
        self.cell_labels = centrosome_analysis_backend.get_cell_labels(self.cell_map, self.detections)

        idx = (self.detection_scores>self.det_thresh) & (self.labels==1)
        self.final_detections = self.detections[idx,:]
        self.final_detection_scores = self.detection_scores[idx]
        self.final_cell_labels = self.cell_labels[idx]
        self.amplified, self.chosen_for_analysis, self.labelids = centrosome_analysis_backend.cell_analysis(self.final_detections, self.final_cell_labels)
        

    def redraw(self):
        self.draw(redraw_img=True)

    def draw(self, redraw_img=False):

        if self.zoom_bbox is not None:
            xmin, ymin, xmax, ymax = self.zoom_bbox
            w = xmax-xmin+1
            h = ymax-ymin+1
            if self.zoom_rect is None:
                self.zoom_rect = Rectangle((xmin,ymin),w,h, fill=False, edgecolor='w')
                self.ax.add_patch(self.zoom_rect)
            else:
                self.zoom_rect.set_xy((xmin, ymin))
                self.zoom_rect.set_width(w)
                self.zoom_rect.set_height(h)
        if redraw_img:
            self.ax.clear()


            index = self.display_choices.index(str(self.display_image_var.get()))
            if index==4:
                self.imgdisp = self.display_falsecolor
            elif index==5 or index==6:
                if self.eb3_channel_var.get()=='Absent':
                    tkinter.messagebox.showerror('Error','Choose Eb3 channel first!')
                else:
                    eb3_channel = int(self.eb3_channel_var.get())-1
                    eb3 = self.img[eb3_channel,:,:]
                    rel_thresh = float(self.rel_thresh_var.get())
                    abs_thresh = float(self.abs_thresh_var.get())
                    eb3, labeled_segments = centrosome_analysis_backend.eb3_analysis.preprocess(eb3, rel_thresh)
                    if index==5:
                        self.imgdisp = get_single_channel_img(eb3>abs_thresh)
                    else:
                        self.imgdisp = get_single_channel_img(labeled_segments!=0)
            else:
                if index>=len(self.display_channels):
                    tkinter.messagebox.showerror('Channel {:d} does not exist!'.format(index+1))
                self.imgdisp = self.display_channels[index]
            # channels_to_show=[]
            # if self.show_channel1:
            #     channels_to_show.append(0)
            # if self.show_channel2:
            #     channels_to_show.append(1)
            # if self.show_channel3:
            #     channels_to_show.append(2)
            # if len(channels_to_show)==1:
            #     for i in range(3):
            #         self.imgdisp[:,:,i] = self.imgdisp_orig[:,:,channels_to_show[0]]
            # else:
            #     for c in channels_to_show:
            #         self.imgdisp[:,:,c] = self.imgdisp_orig[:,:,c]




            #self.ax.imshow(self.imgdisp, zorder=-500000)


            if self.ml_model_has_run:


                self.bmap = self.render_boundary()

                if self.show_cell_boundaries:
                    img_to_show = np.minimum(1, self.imgdisp.astype(float)/255. + self.bmap)
                else:
                    img_to_show = self.imgdisp
                self.ax.imshow(img_to_show, zorder=-500000)


                F = self.final_detections
                C = self.chosen_for_analysis
                labelids = self.labelids
                if self.show_dets:
                    for i, k in enumerate(labelids):
                        G = F[(self.final_cell_labels==k) & C,:]
                        self.ax.scatter(G[:,0], G[:,1], 15, edgecolor=self.colors[i,:], facecolor='None')
                        G = F[(self.final_cell_labels==k) & ~C,:]
                        self.ax.scatter(G[:,0], G[:,1], 15, facecolor=self.colors[i,:], marker='+')
            else:
                self.ax.imshow(self.imgdisp)







        self.ax.set_xlim(self.xlim[0],self.xlim[1])
        self.ax.set_ylim(self.ylim[1],self.ylim[0])
        self.ax.set_xlabel('Pixels')
        self.ax.set_ylabel('Pixels')
        #scalebar
        pixelscale = float(self.scalevar.get())
        micron = 5./pixelscale
        scalebarwidth = micron
        scalebarheight = float(self.ylim[1]-self.ylim[0])*0.01
        scalebarx = self.xlim[1]-scalebarwidth-0.05*(self.xlim[1]-self.xlim[0])
        scalebary = self.ylim[1]-float(self.ylim[1]-self.ylim[0])*0.05
        scalebar = Rectangle((scalebarx,scalebary),scalebarwidth,scalebarheight, fill=True, facecolor='w')
        self.ax.add_patch(scalebar)
        #QWERTY
        self.canvas.draw()


    def final_analysis(self):
        pixelscale = float(self.scalevar.get())
        if not self.ml_model_has_run:
            tkinter.messagebox.showerror('Error','Run ML model first!')
            return

        analysis_type = str(self.analysis_var.get())
        if analysis_type==self.analysis_choices[0]:
            mt_max_radius = float(self.eb3radiusvar.get())
            mt_step = float(self.eb3stepvar.get())
            radii = np.arange(mt_step, mt_max_radius, mt_step)
            radii_pixels = radii/pixelscale
            bgradii_pixels = self.bgradii/pixelscale
            #density on MT
            mt_channel = int(self.mt_channel_var.get())-1
            img_for_analysis = self.img[mt_channel,:,:]
            intensities, areas, densities = centrosome_analysis_backend.intensity_profile(img_for_analysis,
                                                self.final_detections, self.final_cell_labels, self.chosen_for_analysis,
                                                self.cell_map,radii_pixels, bgradii_pixels)
            self.analysis_outcomes = (intensities, areas, densities, radii)
        elif analysis_type==self.analysis_choices[1]:
            pcm_max_radius = float(self.pcmradiusvar.get())
            pcm_step = float(self.pcmstepvar.get())
            radii = np.arange(pcm_step, pcm_max_radius, pcm_step)
            radii_pixels = radii/pixelscale
            bgradii_pixels = self.bgradii/pixelscale
            #density on pcm
            pcm_channel = int(self.pcm_channel_var.get())-1
            img_for_analysis = self.img[pcm_channel,:,:]
            intensities, areas, densities = centrosome_analysis_backend.intensity_profile(img_for_analysis,
                                                self.final_detections, self.final_cell_labels, self.chosen_for_analysis,
                                                self.cell_map,radii_pixels, bgradii_pixels)
            self.analysis_outcomes = (intensities, areas, densities, radii)
        elif analysis_type==self.analysis_choices[2]:
            mt_max_radius = float(self.eb3radiusvar.get())
            mt_step = float(self.eb3stepvar.get())
            radii = np.arange(mt_step, mt_max_radius, mt_step)
            radii_pixels = radii/pixelscale

            #eb3
            eb3_channel = int(self.eb3_channel_var.get())-1
            centrin_channel = int(self.centrin_channel_var.get())-1

            eb3 = self.img[eb3_channel,:,:]
            centrin = self.img[centrin_channel,:,:]
            rel_thresh = float(self.rel_thresh_var.get())
            abs_thresh = float(self.abs_thresh_var.get())
            length_thresh = float(self.length_thresh_var.get())
            vis = bool(self.save_eb3_vis_var.get())
            vis_dir = os.path.splitext(self.open_file)[0]+'_eb3_vis'
            if vis and not os.path.isdir(vis_dir):
                os.makedirs(vis_dir)
            eb3, labeled_segments = centrosome_analysis_backend.eb3_analysis.preprocess(eb3, rel_thresh)
            counts, areas, densities = centrosome_analysis_backend.eb3_count_density(labeled_segments, eb3,
                                            self.final_detections, self.final_cell_labels, self.chosen_for_analysis,
                                            radii_pixels, length_thresh, abs_thresh, centrosome=centrin, vis_dir=vis_dir, vis=vis)
            self.analysis_outcomes = (counts, areas, densities, radii)

    def _save_foci_cellmap(self):
        self.json_file = asksaveasfilename(defaultextension='json')
        self.cell_map_file = os.path.splitext(self.json_file)[0]+'.npz'
        print(self.json_file, self.cell_map_file)

        with open(self.json_file, 'w') as f:
            json.dump(dict(foci=[self.final_detections.tolist()],foci_scores=self.final_detection_scores.tolist()),f)
        np.savez(self.cell_map_file, cell_probs=self.cell_probabilities, cell_bmap=self.cell_bmap)

    def _save(self):
        if self.analysis_outcomes is None:
            self.final_analysis()

        if self.save_file is None:
            self.save_file = asksaveasfilename()



        intensities, areas, densities, radii = self.analysis_outcomes
        centrosome_analysis_backend.save_to_csv(self.open_file, self.save_file, intensities, areas, densities, self.amplified, radii)
        tkinter.messagebox.showinfo("Finished", "Analysis results have been saved.")


    def _zoom(self, x, y):

        xmin, ymin, xmax, ymax = self.zoom_bbox
        xmax = x
        ymax = y
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymax, ymin)
        self.zoom_rect.remove()
        self.zoom_rect=None
        self.zoom_bbox=None
        self.xlim[0] = xmin
        self.xlim[1] = xmax
        self.ylim[0] = ymin
        self.ylim[1] = ymax
        self.draw()

    def _zoomcluster(self, x,y):
        xmin = x-5
        xmax = x+5
        ymin = y-5
        ymax = y+5
        cell_id = self.cell_map[x,y]
        foci = self.detections[:self.num_detections,:]
        foci = foci[self.cell_labels==cell_id,:]
        if foci is not None:
            xmin = min(xmin, np.minimum(foci[:,0]))
            ymin = min(ymin, np.minimum(foci[:,1]))
            xmax = max(xmax, np.maximum(foci[:,0]))
            ymax = max(ymax, np.maximum(foci[:,1]))





        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymax, ymin)
        self.xlim[0] = xmin
        self.xlim[1] = xmax
        self.ylim[0] = ymin
        self.ylim[1] = ymax



        self.draw()

    def _remove_cell(self, x, y):
        if not self.ml_model_has_run:
            tkinter.messagebox.showerror('Error','Run the ML model first!')
            return
        y = min(max(int(y),0),self.cell_map.shape[0]-1)
        x = min(max(int(x),0),self.cell_map.shape[1]-1)

        labelid = self.cell_map[y,x]
        self.labels[self.cell_labels==labelid] = 0
        self.do_cell_analysis(reestimate_boundaries=False)
        self.draw(redraw_img=True)
    def _correct(self, x, y):
        if not self.ml_model_has_run:
            tkinter.messagebox.showerror('Error','Run the ML model first!')
            return
        F = self.detections[self.detection_scores>self.det_thresh,:]

        dist = np.sqrt((x-F[:,0])**2 + (y-F[:,1])**2)
        dist[self.labels[self.detection_scores>self.det_thresh]==0]=np.inf
        minidx = np.argmin(dist)
        if dist[minidx]<=5:
            self.labels[minidx]=0
        self.do_cell_analysis(reestimate_boundaries=False)
        self.draw(redraw_img=True)

    def _addpoint(self, x,y):
        #if self.new_points is None:
        #    self.new_points = np.array([[x,y]])
        #else:
        #    self.new_points = np.concatenate((self.new_points, np.array([[x,y]])))
        if not self.ml_model_has_run:
            tkinter.messagebox.showerror('Run the ML model first!')
            return
        self.detections = np.concatenate((np.array([[x,y]]),self.detections),axis=0)
        self.detection_scores = np.concatenate((np.inf*np.ones(1), self.detection_scores),axis=0)
        self.labels = np.concatenate((np.ones(1),self.labels), axis=0)
        self.do_cell_analysis(reestimate_boundaries=False)
        self.draw(redraw_img=True)

    def draw_cell_boundary(self):
        if not self.ml_model_has_run:
            tkinter.messagebox.showerror('Run the ML model first!')
            return
        bmap = np.zeros_like(self.cell_probabilities)

        for i, p in enumerate(self.line[:-1]):
            coords = bresenham_line(np.array(p), np.array(self.line[i+1]))
            bmap[coords[:,1],coords[:,0]] = 1
        D = distance_transform_edt(bmap==0)
        self.cell_probabilities[D<15] = 0
        self.cell_bmap[D<=1] = 1
        #self.cell_bmap = cell_segmentation.compute_cell_bmap(self.cell_probabilities)
        self.do_cell_analysis()
        self.draw(redraw_img=True)

    def merge_cells(self):
        if not self.ml_model_has_run:
            tkinter.messagebox.showerror('Run the ML model first!')
            return
        cell_ids =self.to_merge
        label = self.cell_map
        Ds = []
        D_fin = np.zeros(self.cell_probabilities.shape)
        coords = []
        for c in cell_ids:
            D = distance_transform_edt(label!=c)
            D_fin = D + D_fin
            Dinside = distance_transform_edt(label==c)
            y, x = np.where(Dinside==np.max(Dinside))
            z = np.concatenate((y.reshape((-1,1)), x.reshape((-1,1))),axis=1)
            coords.append(z)
        coords = np.concatenate(coords, axis=0)



        for p in np.arange(np.min(D_fin),np.max(D_fin),20):
            cell_probs = self.cell_probabilities.copy()
            cell_probs[D_fin<=p] = 1
            cell_bmap = cell_segmentation.compute_cell_bmap(cell_probs)
            cellmap = centrosome_analysis_backend.get_cell_map(cell_probs, cell_bmap, self.cell_probability_thresh)
            if len(np.unique(cellmap))==1:
                continue
            if len(np.unique(cellmap[coords[:,0],coords[:,1]]))==1:

                cell_id = np.unique(cellmap[coords[:,0],coords[:,1]])[0]

                self.cell_probabilities = cell_probs

                self.cell_bmap = cell_segmentation.compute_cell_bmap(cell_probs)
                break

        self.do_cell_analysis()
        self.draw(redraw_img=True)
        self.to_merge = []







    def mark_as_amplified(self, x, y):
        if not self.ml_model_has_run:
            tkinter.messagebox.showerror('Run the ML model first!')
            return
        labelid = self.cell_map[int(y), int(x)]

        if labelid in self.labelids:
            index = np.where(self.labelids==labelid)[0][0]
            self.amplified[index] = 1
            self.draw(redraw_img=True)

    def mark_as_not_amplified(self, x, y):
        if not self.ml_model_has_run:
            tkinter.messagebox.showerror('Run the ML model first!')
            return
        labelid = self.cell_map[int(y), int(x)]
        if labelid in self.labelids:
            index = np.where(self.labelids==labelid)[0][0]
            self.amplified[index] = 0
            self.draw(redraw_img=True)





    def on_button_release(self, event):


        if event.button!=1:
            return
        mode = self.mode
        if mode=='zoom':
            self._zoom(event.xdata, event.ydata)
        elif mode=='correct':
            self._correct(event.xdata, event.ydata)
        elif mode=='remove_cell':
            self._remove_cell(event.xdata,event.ydata)
        elif mode=='add':
            self._addpoint(event.xdata, event.ydata)
        elif mode=='choose':
            self._zoomcluster(event.xdata, event.ydata)
        elif mode=='draw':
            dist = max(np.abs(event.xdata-self.line[-1][0]), np.abs(event.ydata-self.line[-1][1]))
            if dist>1:
                self.line.append((event.xdata,event.ydata))

            self.draw_cell_boundary()
            self.line = []

        elif mode=='merge':
            cellid = self.cell_map[int(event.ydata), int(event.xdata)]
            self.to_merge.append(cellid)
        elif mode=='amplified':
            self.mark_as_amplified(event.xdata, event.ydata)
        elif mode=='not amplified':
            self.mark_as_not_amplified(event.xdata, event.ydata)





    def on_mouse_move(self,event):
        if event.button!=1:
            return
        mode = self.mode
        if mode=='zoom':
            self.zoom_bbox[2] = event.xdata
            self.zoom_bbox[3] = event.ydata
            self.draw()
        elif mode=='draw':
            dist = max(np.abs(event.xdata-self.line[-1][0]), np.abs(event.ydata-self.line[-1][1]))
            self.plot_line.extend(self.ax.plot([self.line[-1][0], event.xdata], [self.line[-1][1], event.ydata], color='w'))
            self.canvas.draw()
            if dist>1:
                self.line.append((event.xdata,event.ydata))


    def on_button_press(self,event):

        if event.button!=1:
            return
        if self.mode=='zoom':
            self.zoom_bbox = [event.xdata, event.ydata, event.xdata, event.ydata]
            self.draw()
        elif self.mode=='draw':
            self.line = [(event.xdata,event.ydata)]
            self.plot_line = []


    def set_zoom_mode(self):

        self.mode='zoom'
        self.modevar.set("Zoom mode")

    def set_correct_mode(self):
        if not self.ml_model_has_run:
            tkinter.messagebox.showerror('Error','Run ML model first!')
            return
        self.mode='correct'
        self.modevar.set("Removing wrong detections")

    def set_remove_cell_mode(self):
        if not self.ml_model_has_run:
            tkinter.messagebox.showerror('Error','Run ML model first!')
            return
        self.mode='remove_cell'
        self.modevar.set("Removing cells from analysis")

    def set_add_mode(self):
        if not self.ml_model_has_run:
            tkinter.messagebox.showerror('Error','Run ML model first!')
            return
        self.mode='add'
        self.modevar.set("Adding new detections")

    def set_merge_mode(self):
        if not self.ml_model_has_run:
            tkinter.messagebox.showerror('Error','Run ML model first!')
            return

        self.mode='merge'
        self.modevar.set('Click cells to merge')
        self.to_merge = []



    def toggle_cell_boundary(self):
        self.show_cell_boundaries = int(self.show_cell_bdry_var.get())
        self.draw(redraw_img=True)

    def toggle_dets(self):
        self.show_dets = int(self.show_detections_var.get())
        self.draw(redraw_img=True)

    def toggle_channel1(self):
        self.show_channel1 = int(self.show_channel1_var.get())
        self.draw(redraw_img=True)

    def toggle_channel2(self):
        self.show_channel2 = int(self.show_channel2_var.get())
        self.draw(redraw_img=True)

    def toggle_channel3(self):
        self.show_channel3 = int(self.show_channel3_var.get())
        self.draw(redraw_img=True)



    def _set_num(self,num):
        self.det_thresh=num
        self.do_cell_analysis(reestimate_boundaries=False)
        self.draw(redraw_img=True)

    def reset_zoom(self):
        self.xlim = [0,self.imgdisp.shape[1]]
        self.ylim = [0, self.imgdisp.shape[0]]
        self.draw()

    def set_det_thresh(self):
        if not self.ml_model_has_run:
            tkinter.messagebox.showerror('Error','Run ML model first!')
            return
        self.det_thresh = float(self.det_thresh_var.get())
        self.do_cell_analysis(reestimate_boundaries=False)
        self.draw(redraw_img=True)

    def set_cell_thresh(self):
        if not self.ml_model_has_run:
            tkinter.messagebox.showerror('Error','Run ML model first!')
            return
        self.cell_probability_thresh = float(self.cell_thresh_var.get())
        self.do_cell_analysis()
        self.draw(redraw_img=True)

    def set_scale(self):
        self.draw(redraw_img=True)

    def set_draw_mode(self):
        if not self.ml_model_has_run:
            tkinter.messagebox.showerror('Error','Run ML model first!')
            return
        self.mode = 'draw'
        self.modevar.set('Drawing new boundary')

    def set_amplified_mode(self):
        if not self.ml_model_has_run:
            tkinter.messagebox.showerror('Error','Run ML model first!')
            return
        self.mode ='amplified'
        self.modevar.set('Click on cell to mark as amplified')

    def set_notamplified_mode(self):
        if not self.ml_model_has_run:
            tkinter.messagebox.showerror('Error','Run ML model first!')
            return
        self.mode ='not amplified'
        self.modevar.set('Click on cell to mark as not amplified')



    def _quit(self):
        self.master.quit()     # stops mainloop
        self.master.destroy()  # this is necessary on Windows to prevent
                        # Fatal Python Error: PyEval_RestoreThread: NULL tstate




def do_start():
    root = tkinter.Tk()
    root.wm_title("Analysis")
    mygui = AnalysisGUI(root)
if __name__=='__main__':

    do_start()
    #tkinter.mainloop()
    #detections, cell_labels, cell_map = mygui.get_detections()
    #intensities, areas, densities = do_analysis(detections, cell_labels, cell_map)
