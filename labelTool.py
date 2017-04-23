# -*- coding:utf-8 -*-  
# -------------------------------------------------------------------------------
# Name:        COLLECTOR BRUSH DEEP-LEARNING DATA PRODUCT
# Purpose:     LABEL BRUSH CONDITION
# Author:      Wang Zhipeng
# Created:     03/05/2017
# -------------------------------------------------------------------------------
from __future__ import division
from tkinter import *
from PIL import Image, ImageTk
import os
import glob
import cv2
import shutil
import random

w0 = 1;  # 图片原始宽度
h0 = 1;  # 图片原始高度

# colors for the bboxes  
COLORS = ['red', 'blue', 'yellow', 'pink', 'cyan', 'green', 'black']

# image sizes for the examples  
SIZE = 256, 256


# 指定缩放后的图像大小
# DEST_SIZE = 500, 500


class LabelTool():
    '''
    该工具实现某文件夹下图片文件的读取和加载
    将加载图片人工标示区域和区域代表的位置
    将标示出来的区域提取出来与lable单独存储
    '''

    def __init__(self, master):
        #类变量：
        self.img = 0#图片
        self.img_cv2 = 0#切割图片
        # 初始化界面
        # set up the main frame
        self.parent = master
        self.parent.title("Collector Brush Img Lable Tool")
        self.frame = Frame(self.parent)
        self.frame.pack(fill=BOTH, expand=1)
        self.parent.resizable(width=TRUE, height=TRUE)

        # initialize global state  
        self.imageDir = ''
        self.imageList = []
        self.egDir = ''
        self.egList = []
        self.outDir = ''
        self.cur = 0
        self.total = 0
        self.category = 0
        self.imagename = ''
        self.labelfilename = ''
        self.tkimg = None

        # initialize mouse state  
        self.STATE = {}
        self.STATE['click'] = 0
        self.STATE['x'], self.STATE['y'] = 0, 0

        # reference to bbox  
        self.bboxIdList = []
        self.bboxId = None
        self.bboxList = []
        self.bboxList_tmp = []
        self.hl = None
        self.vl = None

        # add radio
        self.allowed_or_not = [('0', '报警'), ('1', '预警'), ('2', '状态良好')]

        # ----------------- GUI stuff ---------------------
        # dir entry & load  
        self.label = Label(self.frame, text="Image Dir:")
        self.label.grid(row=0, column=0, sticky=E)
        self.entry = Entry(self.frame)
        self.entry.grid(row=0, column=1, sticky=W + E)
        self.ldBtn = Button(self.frame, text="Load", command=self.loadDir)
        self.ldBtn.grid(row=0, column=2, sticky=W + E)

        # main panel for labeling
        self.mainPanel = Canvas(self.frame, cursor='tcross')
        self.mainPanel.bind("<ButtonPress-1>", self.mouseClick)
        self.mainPanel.bind("<ButtonRelease-1>", self.mouseClick)
        self.mainPanel.bind("<Motion>", self.mouseMove)
        self.parent.bind("<Escape>", self.cancelBBox)  # press <Espace> to cancel current bbox
        self.parent.bind("s", self.cancelBBox)
        self.parent.bind("a", self.prevImage)  # press 'a' to go backforward
        self.parent.bind("d", self.nextImage)  # press 'd' to go forward
        self.mainPanel.grid(row=1, column=1, rowspan=4, sticky=W + N)

        # showing bbox info & delete bbox
        self.lb1 = Label(self.frame, text='Bounding boxes:')
        self.lb1.grid(row=1, column=2, sticky=W + N)

        self.listbox = Listbox(self.frame, width=28, height=12)
        self.listbox.grid(row=2, column=2, sticky=N)

        self.btnDel = Button(self.frame, text='Delete', command=self.delBBox)
        self.btnDel.grid(row=3, column=2, sticky=W + E + N)
        self.btnClear = Button(self.frame, text='ClearAll', command=self.clearBBox)
        self.btnClear.grid(row=4, column=2, sticky=W + E + N)

        # control panel for image navigation
        self.BrushLabel = Label(self.frame, text="Brush")
        self.BrushLabel.grid(row=8, column=0, sticky=W)

        self.brushSignal = IntVar()
        Radiobutton(self.frame, variable=self.brushSignal, value=self.allowed_or_not[0][0],
                    text=self.allowed_or_not[0][1]).grid(row=8, column=1, sticky=E)
        Radiobutton(self.frame, variable=self.brushSignal, value=self.allowed_or_not[1][0],
                    text=self.allowed_or_not[1][1]).grid(row=8, column=1,
                                                         sticky=N)
        Radiobutton(self.frame, variable=self.brushSignal, value=self.allowed_or_not[2][0],
                    text=self.allowed_or_not[2][1]).grid(row=8, column=1,
                                                         sticky=W)

        self.ctrPanel = Frame(self.frame)
        self.ctrPanel.grid(row=5, column=1, columnspan=2, sticky=W + E)
        self.prevBtn = Button(self.ctrPanel, text='<< Prev', width=10, command=self.prevImage)
        self.prevBtn.pack(side=LEFT, padx=5, pady=3)
        self.nextBtn = Button(self.ctrPanel, text='Next >>', width=10, command=self.nextImage)
        self.nextBtn.pack(side=LEFT, padx=5, pady=3)
        self.progLabel = Label(self.ctrPanel, text="Progress:     /    ")
        self.progLabel.pack(side=LEFT, padx=5)
        self.tmpLabel = Label(self.ctrPanel, text="Go to Image No.")
        self.tmpLabel.pack(side=LEFT, padx=5)
        self.idxEntry = Entry(self.ctrPanel, width=5)
        self.idxEntry.pack(side=LEFT)
        self.goBtn = Button(self.ctrPanel, text='Go', command=self.gotoImage)
        self.goBtn.pack(side=LEFT)

        # example pannel for illustration
        self.egPanel = Frame(self.frame, border=10)
        self.egPanel.grid(row=2, column=0, rowspan=5, sticky=N)
        self.tmpLabel2 = Label(self.egPanel, text="Examples:")
        self.tmpLabel2.pack(side=TOP, pady=5)

        self.egLabels = []
        for i in range(3):
            self.egLabels.append(Label(self.egPanel))
        self.egLabels[-1].pack(side=TOP)

        # display mouse position
        self.disp = Label(self.ctrPanel, text='')
        self.disp.pack(side=RIGHT)

        self.frame.columnconfigure(1, weight=1)
        self.frame.rowconfigure(4, weight=1)

    def loadDir(self, dbg=False):
        if not dbg:
            s = self.entry.get()
            # print(s)
            self.parent.focus()
            if s == "":
                self.category = 0
            else:
                Img_load_Path = s
                print("图片加载路径:", Img_load_Path)

        else:
            # s = r'C:\\PicCut_DeepLearning'
            Img_load_Path = r'./'
            print("need a path")

        # self.imageDir = os.path.join(Img_load_Path, '%03d' % (self.category))

        self.imageDir = Img_load_Path
        # print(self.imageDir)
        self.imageList = glob.glob(os.path.join(self.imageDir, '*.png'))
        self.imageList = self.imageList + glob.glob(os.path.join(self.imageDir, '*.jpg'))
        if len(self.imageList) == 0:
            print('没找到图片文件')
            return
        else:
            print('共有 %d 张图片' % (len(self.imageList)))

        # default to the 1st image in the collection
        self.cur = 1
        self.total = len(self.imageList)

        # set up output dir
        self.outDir = os.path.join(r'./labels')
        # print(self.outDir)
        if not os.path.exists(self.outDir):
            os.mkdir(self.outDir)

        # load example bboxes
        self.egDir = os.path.join(r'./Examples')
        # print(self.egDir)

        # 暂未发现有什么用
        # filelist = glob.glob(os.path.join(self.egDir, '*.png'))
        # self.tmp = []
        # self.egList = []
        # random.shuffle(filelist)
        # for (i, f) in enumerate(filelist):
        #     if i == 3:
        #         break
        #     im = Image.open(f)
        #     r = min(SIZE[0] / im.size[0], SIZE[1] / im.size[1])
        #     new_size = int(r * im.size[0]), int(r * im.size[1])
        #     self.tmp.append(im.resize(new_size, Image.ANTIALIAS))
        #     self.egList.append(ImageTk.PhotoImage(self.tmp[-1]))
        #     self.egLabels[i].config(image=self.egList[-1], width=SIZE[0], height=SIZE[1])

        self.loadImage()
        # print('%d images loaded from %s' % (self.total, s))

    def loadImage(self):
        # load image  
        imagepath = self.imageList[self.cur - 1]
        pil_image = Image.open(imagepath)

        # 获取图像的原始大小
        global w0, h0
        w0, h0 = pil_image.size

        # 缩放到指定大小（暂不需要)
        # pil_image = pil_image.resize((DEST_SIZE[0], DEST_SIZE[1]), Image.ANTIALIAS)
        self.img = pil_image
        self.img_cv2 = cv2.imread(imagepath)

        self.tkimg = ImageTk.PhotoImage(pil_image)
        self.mainPanel.config(width=max(self.tkimg.width(), 400), height=max(self.tkimg.height(), 400))
        self.mainPanel.create_image(0, 0, image=self.tkimg, anchor=NW)
        self.progLabel.config(text="%04d/%04d" % (self.cur, self.total))

        # load labels  
        self.clearBBox()
        self.imagename = os.path.split(imagepath)[-1].split('.')[0]
        labelname = self.imagename + '.txt'
        self.labelfilename = os.path.join(self.outDir, labelname)
        bbox_cnt = 0
        if os.path.exists(self.labelfilename):
            with open(self.labelfilename) as f:
                for (i, line) in enumerate(f):
                    # print("i: ", i)
                    if i == 0:
                        bbox_cnt = int(line.strip())
                        continue

                    tmp = [(t.strip()) for t in line.split()]

                    self.bboxList.append(tuple(tmp))
                    self.bboxList_tmp.append(tuple(tmp))
                    tx0 = float(tmp[0])
                    ty0 = float(tmp[1])
                    tx1 = float(tmp[2])
                    ty1 = float(tmp[3])
                    brush = int(tmp[4])

                    # 还原大小
                    # tx0 = int(tx0 * DEST_SIZE[0])
                    # ty0 = int(ty0 * DEST_SIZE[1])
                    # tx1 = int(tx1 * DEST_SIZE[0])
                    # ty1 = int(ty1 * DEST_SIZE[1])

                    tmpId = self.mainPanel.create_rectangle(tx0, ty0, tx1, ty1, \
                                                            width=2, \
                                                            outline=COLORS[(len(self.bboxList) - 1) % len(COLORS)])

                    self.bboxIdList.append(tmpId)
                    self.listbox.insert(END, '(%.2f,%.2f)-(%.2f,%.2f) %d' % (tx0, ty0, tx1, ty1, brush))
                    self.listbox.itemconfig(len(self.bboxIdList) - 1,
                                            fg=COLORS[(len(self.bboxIdList) - 1) % len(COLORS)])

    def saveImage(self):
        # print(self.bboxList)
        self.syncImg()
        with open(self.labelfilename, 'w') as f:
            f.write('%d\n' % len(self.bboxList))
            i = 0
            for bbox in self.bboxList:
                f.write(' '.join(map(str, bbox)) + '\n')
                tmp = bbox
                tx0 = int(tmp[0])
                ty0 = int(tmp[1])
                tx1 = int(tmp[2])
                ty1 = int(tmp[3])
                brush = int(tmp[4])
                # 还原大小
                # tx0 = int(tx0 * DEST_SIZE[0])
                # ty0 = int(ty0 * DEST_SIZE[1])
                # tx1 = int(tx1 * DEST_SIZE[0])
                # ty1 = int(ty1 * DEST_SIZE[1])
                # print(tmp)
                # print(tx0, tx1, ty0, ty1)
                image_path = self.imageDir + os.sep + str(tmp[4]) + os.sep + self.imagename
                print("image_path is ", image_path)
                if os.path.isdir(image_path):
                    print("exists")
                else:
                    os.makedirs(image_path)

                image_path = self.imageDir + os.sep + str(tmp[4]) + os.sep + self.imagename + os.sep + str(i)
                # print(image_path)
                i += 1
                img_new = self.img_cv2[ty0:ty1, tx0:tx1]
                # cv2.imshow('1', img_new)
                # cv2.waitKey(0)
                cv2.imwrite(image_path + '.jpg', img_new)
                # print('Image No. %d saved' % (self.cur))

    def mouseClick(self, event):
        if self.STATE['click'] == 0:
            self.STATE['x'], self.STATE['y'] = event.x, event.y
            self.STATE['click'] = 1
        else:
            # print("mouse click")
            x1, x2 = min(self.STATE['x'], event.x), max(self.STATE['x'], event.x)
            y1, y2 = min(self.STATE['y'], event.y), max(self.STATE['y'], event.y)
            # print(x1, x2, y1, y2)

            # x1, x2 = x1 / DEST_SIZE[0], x2 / DEST_SIZE[0]
            # y1, y2 = y1 / DEST_SIZE[1], y2 / DEST_SIZE[1]

            brush = self.brushSignal.get()
            self.bboxList.append((x1, y1, x2, y2, brush))
            self.bboxIdList.append(self.bboxId)
            self.bboxId = None
            self.listbox.insert(END, '(%.2f, %.2f)-(%.2f, %.2f)' % (x1, y1, x2, y2))
            self.listbox.itemconfig(len(self.bboxIdList) - 1, fg=COLORS[(len(self.bboxIdList) - 1) % len(COLORS)])
            self.STATE['click'] = 1 - self.STATE['click']

    def mouseMove(self, event):
        # self.disp.config(text='x: %.2f, y: %.2f' % (event.x / DEST_SIZE[0], event.y / DEST_SIZE[1]))
        self.disp.config(text='x: %.2f, y: %.2f' % (event.x, event.y))
        if self.tkimg:
            if self.hl:
                self.mainPanel.delete(self.hl)
            self.hl = self.mainPanel.create_line(0, event.y, self.tkimg.width(), event.y, width=2)
            if self.vl:
                self.mainPanel.delete(self.vl)
            self.vl = self.mainPanel.create_line(event.x, 0, event.x, self.tkimg.height(), width=2)
        if 1 == self.STATE['click']:
            if self.bboxId:
                self.mainPanel.delete(self.bboxId)
            self.bboxId = self.mainPanel.create_rectangle(self.STATE['x'], self.STATE['y'], \
                                                          event.x, event.y, \
                                                          width=2, \
                                                          outline=COLORS[len(self.bboxList) % len(COLORS)])

    def cancelBBox(self, event):
        if 1 == self.STATE['click']:
            if self.bboxId:
                self.mainPanel.delete(self.bboxId)
                self.bboxId = None
                self.STATE['click'] = 0

    def delBBox(self):
        sel = self.listbox.curselection()
        if len(sel) != 1:
            return
        idx = int(sel[0])
        self.mainPanel.delete(self.bboxIdList[idx])
        self.bboxIdList.pop(idx)
        self.bboxList.pop(idx)
        self.listbox.delete(idx)

    def clearBBox(self):
        for idx in range(len(self.bboxIdList)):
            self.mainPanel.delete(self.bboxIdList[idx])
        self.listbox.delete(0, len(self.bboxList))
        self.bboxIdList = []
        self.bboxList = []

    def prevImage(self, event=None):
        self.saveImage()
        if self.cur > 1:
            self.cur -= 1
            self.loadImage()

    def nextImage(self, event=None):
        self.saveImage()
        if self.cur < self.total:
            self.cur += 1
            self.loadImage()

    def gotoImage(self):
        idx = int(self.idxEntry.get())

        if 1 <= idx and idx <= self.total:
            self.saveImage()
            self.cur = idx
            self.loadImage()

            # def imgresize(w, h, w_box, h_box, pil_image):
            #     '''''
            #     resize a pil_image object so it will fit into
            #     a box of size w_box times h_box, but retain aspect ratio
            #     '''
            #     f1 = 1.0 * w_box / w  # 001.0 forces float division in Python2
            #     f2 = 1.0 * h_box / h
            #     factor = min([f1, f2])
            #     # print(f1, f2, factor) # test
            #     # use best down-sizing filter
            #     width = int(w * factor)
            #     height = int(h * factor)
            #     return pil_image.resize((width, height), Image.ANTIALIAS)

    def syncImg(self):
        i = 0
        for b in self.bboxList_tmp:
            print(b)
            if b in self.bboxList:
                print("此条无变化")
            else:
                tmp = b
                tx0 = int(tmp[0])
                ty0 = int(tmp[1])
                tx1 = int(tmp[2])
                ty1 = int(tmp[3])
                brush = int(tmp[4])

                image_path = self.imageDir + os.sep + str(tmp[4]) + os.sep + self.imagename + os.sep + str(i) + '.jpg'
                print("删除", image_path)
                if os.path.exists(image_path):
                    os.remove(image_path)
            i += 1


if __name__ == '__main__':
    root = Tk()
    tool = LabelTool(root)
    root.mainloop()

