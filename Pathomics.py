#!/usr/bin/env python3
#  -*- coding: utf-8 -*-
#
# Support module generated by PAGE version 8.0
#  in conjunction with Tcl version 8.6
#    May 15, 2024 12:31:02 AM +0300  platform: Windows NT

# Standard library modules.
import tkinter
import tkinter.font
import tkinter.filedialog
import tkinter.ttk
from tkinter.constants import *

# Local modules.
import gui

##############################
# CONSTANTS                  #
##############################
ASSETS_PATH  = r".\Assets"
CHOOSE_MODEL = False

##############################
# MAIN                       #
##############################
def main():
    global root
    root = tkinter.Tk()
    root.protocol('WM_DELETE_WINDOW', root.destroy)

    # Creates a toplevel widget.
    global _top1, _w1
    _top1 = root
    _w1 = gui.MainPage(_top1, assets=ASSETS_PATH)
    _w1.choose_model = CHOOSE_MODEL

    # Run GUI.
    root.mainloop()

if __name__ == '__main__':
    main()